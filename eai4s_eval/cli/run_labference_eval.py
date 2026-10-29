"""Serial, resumable evaluation for the featured four and optional L/R tasks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import yaml

from eai4s_eval.eval.batch_results import FEATURED, SCORE_VERSION, normalize_result, read_json, write_json, write_reports
from eai4s_eval.eval.task_suite import load_task_catalog, load_task_suite

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {'weighing_hard': ('weighing_v1', 'hard'),
            'device_operation_hard': ('device_operation_v1', 'hard'),
            'centrifuge_hard': ('centrifuge_v2', 'hard'),
            'rack_same_cap': ('rack_recall_v4', 'hard_same_cap')}
OPTIONAL_FAMILIES = {'weighing_easy': ('weighing_v1', 'easy'),
                     'centrifuge_easy': ('centrifuge_v2', 'easy'),
                     'rack_easy': ('rack_recall_v4', 'easy'),
                     'rack_distinct_cap': ('rack_recall_v4', 'hard_distinct_cap')}


def definitions():
    tasks = {}
    for task_id, (family, variant) in {**FAMILIES, **OPTIONAL_FAMILIES}.items():
        reference = ROOT/f'eai4s_eval/{family}/configs/{variant}.json'
        native = reference.with_name('native_'+reference.name)
        tasks[task_id] = dict(task_id=task_id, group='special', family=family,
                              reference_config=str(reference), native_config=str(native),
                              max_steps=read_json(native).get('evaluation', {}).get('max_steps', 10000))
        if task_id == 'device_operation_hard':
            # This task uses BenchmarkTaskEvalRunner's action clock, not the
            # legacy collection controller's physics-frame budget.
            spec = yaml.safe_load((ROOT/'eai4s_eval/configs/tasks/level4_L09_hard_device_operation.yaml').read_text())
            tasks[task_id]['max_steps'] = spec['max_steps']
    catalog = load_task_catalog(load_task_suite('eai4s_benchmark_longhorizon_reasoning_v1'))
    for task in catalog['tasks']:
        tasks[task['task_id']] = dict(task_id=task['task_id'], group='ordinary',
                                     task_spec=task['task_spec'], max_steps=task.get('max_steps') or 500)
    return tasks


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'summarize', 'preflight'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--group', choices=('special', 'ordinary', 'both'), default='special')
    p.add_argument('--tasks', help='Comma-separated IDs; ordinary tasks are optional.')
    p.add_argument('--seeds', default='101', help='Explicit comma-separated unique seed list.')
    p.add_argument('--mode', choices=('native', 'reference'), default='native')
    p.add_argument('--model-name', default=None)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--split', choices=('sim_clean', 'sim_dev', 'sim_randomized'), default='sim_dev')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--refresh-records', action='store_true', help='With summarize, reread preserved worker evidence using current normalization.')
    p.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--max-steps', type=int, help='Explicit override; otherwise task-owned budget.')
    p.add_argument('--task-timeout-seconds', type=float, default=1200)
    p.add_argument('--vla-host', default='127.0.0.1')
    p.add_argument('--vla-port', type=int, default=None)
    p.add_argument('--policy-provider', choices=('openpi', 'gr00t'), default='openpi')
    p.add_argument('--policy-backend', choices=('websocket', 'http', 'zmq'), default=None)
    p.add_argument('--policy-profile', default='eai4s_eval/configs/policies/openpi_franka_ee_pos_v1.yaml')
    p.add_argument('--policy-timeout', type=float, default=30)
    return p


def make_plan(args):
    args.model_name = args.model_name or ('reference-engineering' if args.mode == 'reference' else 'OpenPI')
    tasks = definitions()
    chosen = [v.strip() for v in args.tasks.split(',')] if args.tasks else [k for k, v in tasks.items()
        if (args.group == 'both' or v['group'] == args.group) and (v['group'] != 'special' or k in FEATURED)]
    if not chosen or len(set(chosen)) != len(chosen) or any(k not in tasks for k in chosen):
        raise ValueError('tasks must be unique registered IDs')
    if args.group != 'both' and any(tasks[k]['group'] != args.group for k in chosen):
        raise ValueError('selected tasks do not match --group')
    seeds = [int(v) for v in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('seeds must be unique uint32 values')
    if args.mode == 'native' and not args.checkpoint:
        raise ValueError('native evaluation requires --checkpoint')
    if (not math.isfinite(args.task_timeout_seconds) or not math.isfinite(args.policy_timeout)
            or args.task_timeout_seconds <= 0 or args.policy_timeout <= 0
            or args.max_steps is not None and args.max_steps <= 0):
        raise ValueError('budgets must be positive')
    if args.mode == 'native' and args.policy_provider != 'openpi' and any(tasks[k]['group'] == 'special' for k in chosen):
        raise ValueError('special protocols currently require OpenPI; no implicit reference fallback')
    if args.mode == 'native' and args.policy_backend not in (None, 'websocket') and any(tasks[k]['group'] == 'special' for k in chosen):
        raise ValueError('special batch protocols require the OpenPI WebSocket action adapter')
    if args.mode == 'native' and args.policy_provider == 'gr00t' and args.policy_backend not in (None, 'zmq'):
        raise ValueError('GR00T requires the ZMQ adapter')
    if args.mode == 'native' and args.policy_provider == 'openpi' and args.policy_backend == 'zmq':
        raise ValueError('OpenPI requires WebSocket or HTTP, not the GR00T ZMQ adapter')
    args.checkpoint = args.checkpoint or 'reference-only'
    jobs = []
    for key in chosen:
        for seed in seeds:
            task = tasks[key]
            files = [Path(task['task_spec'])] if task['group'] == 'ordinary' else [Path(task['reference_config']), Path(task['native_config'])]
            for path in files:
                if not path.is_file():
                    raise FileNotFoundError(path)
            profile_path = args.policy_profile
            if profile_path == 'eai4s_eval/configs/policies/openpi_franka_ee_pos_v1.yaml':
                if task['group'] == 'ordinary' and args.policy_provider == 'gr00t':
                    profile_path = 'eai4s_eval/configs/policies/gr00t_franka_ee_pos_v1.yaml'
                elif task.get('family') == 'weighing_v1':
                    profile_path = 'eai4s_eval/configs/policies/openpi_weighing_ee_pos_v1.yaml'
                elif task.get('family') == 'rack_recall_v4':
                    profile_path = 'eai4s_eval/configs/policies/openpi_rack_recall_ee_pos_v1.yaml'
            if not Path(profile_path).is_absolute():
                profile_path = str(ROOT/profile_path)
            if not Path(profile_path).is_file():
                raise FileNotFoundError(profile_path)
            from eai4s_eval.policy import load_openpi_profile
            profile = load_openpi_profile(profile_path)
            if task['group'] == 'special' and args.mode == 'native':
                expected_hz = 20 if task.get('family') == 'rack_recall_v4' else 10
                if profile.action_space != 'ee_pos' or profile.control_hz != expected_hz:
                    raise ValueError(f'{key} requires an EE pose profile at {expected_hz} Hz')
                if task.get('family') == 'weighing_v1' and not {'display_rgb', 'weights_rgb'} <= set(profile.observation_map):
                    raise ValueError('feedback weighing requires the display and weights observation streams')
            jobs.append({**task, 'seed': seed, 'split': args.split, 'mode': args.mode,
                         'policy_profile': profile_path,
                         'max_steps': args.max_steps or task['max_steps'],
                         'model_name': args.model_name, 'checkpoint': args.checkpoint,
                         'config_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[Path(profile_path)]}})
    identity = dict(jobs=jobs, policy_profile=args.policy_profile, host=args.vla_host,
                    port=args.vla_port, provider=args.policy_provider, backend=args.policy_backend,
                    timeout=args.task_timeout_seconds, policy_timeout=args.policy_timeout,
                    headless=args.headless, save_video=args.save_video, score_version=SCORE_VERSION)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(schema_version='labference_eval_plan/v1', score_version=SCORE_VERSION,
                run_id=args.output.resolve().name, model_name=args.model_name,
                checkpoint=args.checkpoint, mode=args.mode, fingerprint=fingerprint, jobs=jobs)


def worker_command(args, job, directory):
    raw = directory/'raw'
    if job['group'] == 'ordinary' or job['task_id'] == 'device_operation_hard':
        task_spec = job.get('task_spec') or str(ROOT/'eai4s_eval/configs/tasks/level4_L09_hard_device_operation.yaml')
        cmd = [sys.executable, '-m', 'eai4s_eval.cli.run_benchmark_eval', '--task-spec', task_spec,
               '--output-root', str(raw), '--run-id', 'episode', '--num-episodes', '1',
               '--seeds', str(job['seed']), '--split', job['split'], '--model-name', job['model_name'],
               '--checkpoint', job['checkpoint'], '--max-steps', str(job['max_steps'])]
        if args.mode == 'reference':
            cmd.append('--reference')
        else:
            cmd += ['--policy-provider', args.policy_provider, '--policy-profile', job['policy_profile'],
                    '--vla-host', args.vla_host, '--vla-port', str(args.vla_port or (5555 if args.policy_provider == 'gr00t' else 8000)),
                    '--policy-timeout', str(args.policy_timeout)]
            cmd += ['--policy-backend', args.policy_backend or ('zmq' if args.policy_provider == 'gr00t' else 'websocket')]
    else:
        payload = read_json(Path(job['native_config'] if args.mode == 'native' else job['reference_config']))
        if args.mode == 'native':
            interface = payload['evaluation']['policy_interface']
            interface.update(host=args.vla_host, port=args.vla_port or 8000)
            from dataclasses import replace
            from eai4s_eval.oracle_runtime.contracts import PolicyProfile
            streams = ('front', 'wrist', 'display', 'weights') if job['family'] == 'weighing_v1' else ('front', 'wrist')
            profile = PolicyProfile.legacy(interface, streams)
            profile = replace(profile, action=replace(profile.action, control_hz=20 if job['family'] == 'rack_recall_v4' else 10,
                                                     representation='eef_pose', frame='world',
                                                     units='meters+quaternion_wxyz+finger_meter'))
            if job['family'] == 'weighing_v1':
                profile = replace(profile, input_mapping={'front': 'camera_1_rgb', 'wrist': 'camera_3_rgb',
                                                          'display': 'display_rgb', 'weights': 'weights_rgb'})
            payload['evaluation']['policy_profile'] = profile.public()
            payload['labbench_runtime'] = dict(max_action_steps=job['max_steps'], execution_horizon=None,
                action_space='ee_pos', policy_profile=job['policy_profile'],
                backend=args.policy_backend or 'websocket', policy_timeout=args.policy_timeout)
        config = directory/'protocol.json'
        write_json(config, payload)
        cmd = [sys.executable, '-m', f"eai4s_eval.{job['family']}.cli", 'run',
               '--config', str(config), '--output', str(raw), '--seed', str(job['seed']),
               '--max-steps', str(job['max_steps'])]
    if args.headless:
        cmd.append('--headless')
    if args.save_video:
        cmd.append('--save-video')
    return cmd


def supervise(command, directory, timeout):
    """Drain output without blocking timeout/heartbeats on silent workers."""
    lines = queue.Queue()
    started = time.monotonic()
    with (directory/'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace', start_new_session=True)
        def drain():
            for line in process.stdout:
                lines.put(line)
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        timed_out = False
        last_heartbeat = started
        try:
            while process.poll() is None or thread.is_alive() or not lines.empty():
                now = time.monotonic()
                if not timed_out and now-started >= timeout:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                if timed_out and now-started >= timeout+10:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    thread.join(timeout=2)
                    break
                if now-last_heartbeat >= 15:
                    print(f'[batch heartbeat] {directory.name}: {now-started:.0f}s', flush=True)
                    last_heartbeat = now
                try:
                    line = lines.get(timeout=.2)
                    log.write(line)
                    log.flush()
                except queue.Empty:
                    pass
            return process.wait(), timed_out
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if process.stdout is not None and not thread.is_alive():
                process.stdout.close()


def records(root):
    return [read_json(p) for p in sorted((root/'results').glob('*.json'))]


def check_policy_services(args, plan):
    from eai4s_eval.policy import JointPolicyAdapter, Gr00tPolicyClient, load_openpi_profile
    for path in sorted({j['policy_profile'] for j in plan['jobs']}):
        client = (Gr00tPolicyClient(host=args.vla_host, port=args.vla_port or 5555)
                  if args.policy_provider == 'gr00t' else None)
        policy = JointPolicyAdapter(host=args.vla_host, port=args.vla_port or (5555 if client else 8000),
                                    profile=load_openpi_profile(path), client=client,
                                    backend='gr00t_zmq' if client else args.policy_backend or 'websocket',
                                    timeout_sec=args.policy_timeout)
        try:
            policy.health()
        finally:
            policy.close()


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    if args.command == 'summarize':
        plan = read_json(args.output/'plan.json')
        if not plan:
            raise ValueError('missing evaluation plan')
        if args.refresh_records:
            for job in plan['jobs']:
                key = f"{job['task_id']}_{job['split']}_{job['seed']}"
                target = args.output/'results'/f'{key}.json'
                previous = read_json(target)
                if not previous:
                    continue
                raw = Path(previous.get('raw_dir', '')).resolve()
                if not raw.is_relative_to(args.output/'attempts'):
                    raise ValueError('worker evidence must remain within this batch')
                refreshed = normalize_result(job, raw, int(previous.get('returncode', 127)),
                                             timed_out=previous.get('termination_reason') == 'worker_timeout')
                write_json(target, refreshed)
        result = write_reports(args.output, plan, records(args.output))
        print(json.dumps(result['groups'], indent=2))
        return 0
    plan = make_plan(args)
    if args.command == 'preflight':
        print(json.dumps(plan, indent=2))
        return 0
    old = read_json(args.output/'plan.json')
    if old and (not args.resume or old.get('fingerprint') != plan['fingerprint']):
        raise ValueError('existing output requires --resume with identical configuration')
    if args.output.exists() and not old and any(args.output.iterdir()):
        raise ValueError('output is non-empty and has no batch plan')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'plan.json', plan)
    if args.mode == 'native':
        try:
            check_policy_services(args, plan)
        except Exception as exc:
            write_json(args.output/'preflight_error.json', {'error_type': type(exc).__name__, 'message': str(exc)})
            write_reports(args.output, plan, records(args.output))
            print(f'Policy preflight failed: {type(exc).__name__}: {exc}', flush=True)
            return 2
    for job in plan['jobs']:
        key = f"{job['task_id']}_{job['split']}_{job['seed']}"
        destination = args.output/'results'/f'{key}.json'
        prior = read_json(destination)
        if prior.get('evaluation_valid') is True:
            continue
        attempts = args.output/'attempts'/key
        attempt = attempts/f'{len(list(attempts.glob("*"))):04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        print(f'[batch] {key} mode={args.mode}', flush=True)
        try:
            command = worker_command(args, job, attempt)
            write_json(attempt/'command.json', {'argv': command})
            code, expired = supervise(command, attempt, args.task_timeout_seconds)
            row = normalize_result(job, attempt/'raw', code, timed_out=expired)
        except Exception as exc:
            row = normalize_result(job, attempt/'raw', 127)
            row['error'] = f'{type(exc).__name__}: {exc}'
        write_json(destination, row)
        write_reports(args.output, plan, records(args.output))
        print(f"[batch] {key}: valid={row['evaluation_valid']} success={row['success']} reason={row['termination_reason']}", flush=True)
    report = write_reports(args.output, plan, records(args.output))
    return 0 if all(g['complete'] for g in report['groups'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
