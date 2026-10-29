import json
import sys
import pytest

from eai4s_eval.cli.run_labference_eval import parser, make_plan, worker_command, main, supervise
from eai4s_eval.eval.batch_results import normalize_result, summarize, write_json, write_reports
from eai4s_eval.public_site import build_featured_leaderboard, build_public_snapshot
from eai4s_eval.server.app import create_app, _scan_release_runs
from eai4s_eval.cli.export_public_site import export_public_site


def plan(root, group='special', mode='native', seeds='101,102', tasks=None):
    argv = ['preflight', '--output', str(root), '--group', group, '--checkpoint', 'test-v1',
            '--seeds', seeds, '--mode', mode]
    if tasks:
        argv += ['--tasks', tasks]
    args = parser().parse_args(argv)
    return args, make_plan(args)


def result(job, success=True, valid=True):
    return {**job, 'evaluation_valid': valid, 'success': success if valid else None,
            'ranking_eligible': valid and job['mode'] == 'native', 'termination_reason': 'finished'}


def test_hand_calculated_four_task_macro_and_coverage(tmp_path):
    _, p = plan(tmp_path)
    values = [True, True, True, False, False, False, True, True]
    rows = [result(j, v) for j, v in zip(p['jobs'], values)]
    report = write_reports(tmp_path, p, rows)
    assert report['groups']['special']['final_score_100'] == 62.5
    assert report['groups']['special']['ranking_eligible']
    partial = summarize(p, rows[:-1])['groups']['special']
    assert partial['final_score_100'] is None and partial['provisional_score_100'] is not None
    rows[-1] = result(p['jobs'][-1], valid=False)
    assert summarize(p, rows)['groups']['special']['final_score_100'] is None
    assert (tmp_path/'report.md').is_file() and (tmp_path/'task_scores.csv').is_file()


def test_missing_duplicate_identity_and_reference_safety(tmp_path):
    _, p = plan(tmp_path, mode='reference', seeds='101')
    assert summarize(p, [])['groups']['special']['final_score_100'] is None
    rows = [result(j) for j in p['jobs']]
    assert summarize(p, rows)['groups']['special']['final_score_100'] == 100
    assert not summarize(p, rows)['groups']['special']['ranking_eligible']
    with pytest.raises(ValueError, match='duplicate'):
        summarize(p, rows + rows[:1])
    with pytest.raises(ValueError, match='identity'):
        summarize(p, [{**rows[0], 'checkpoint': 'different'}])


def test_ordinary_subset_macro_local_only(tmp_path):
    _, p = plan(tmp_path, group='ordinary', tasks='L01,R01', seeds='101')
    summary = summarize(p, [result(p['jobs'][0]), result(p['jobs'][1], False)])
    assert summary['groups']['ordinary']['final_score_100'] == 50
    assert summary['groups']['ordinary']['scope'] == 'selected_tasks'
    assert summary['groups']['ordinary']['tracks'] == {'L': 100, 'R': 0}
    assert summary['ordinary_results_public'] is False


def test_normalization_does_not_turn_missing_or_invalid_into_failure(tmp_path):
    _, p = plan(tmp_path/'out', seeds='101')
    job = p['jobs'][0]
    assert normalize_result(job, tmp_path, 0)['success'] is None
    write_json(tmp_path/'run.json', {'seed': 101, 'runtime_status': 'finished'})
    write_json(tmp_path/'scores.json', {'evaluation_valid': False, 'valid_success': True})
    assert normalize_result(job, tmp_path, 0)['success'] is None
    write_json(tmp_path/'scores.json', {'evaluation_valid': True, 'valid_success': False})
    assert normalize_result(job, tmp_path, 0)['success'] is False
    assert normalize_result(job, tmp_path, 1)['evaluation_valid'] is False
    assert normalize_result(job, tmp_path, 124, timed_out=True)['termination_reason'] == 'worker_timeout'


def test_invalid_action_is_failure_but_connection_error_is_invalid(tmp_path):
    _, p = plan(tmp_path/'out', group='ordinary', seeds='101', tasks='L01')
    for reason, expected in [('policy_output_invalid', True), ('policy_request_failed', False), ('timeout', True)]:
        (tmp_path/'episode_metrics.jsonl').write_text(json.dumps({'seed': 101, 'success': False, 'valid_scene': True, 'termination_reason': reason})+'\n')
        row = normalize_result(p['jobs'][0], tmp_path, 0)
        assert row['evaluation_valid'] is expected
        assert row['success'] == (False if expected else None)


def test_batch_public_score_matches_local_and_rejects_duplicates(tmp_path):
    _, p = plan(tmp_path, seeds='101')
    write_json(tmp_path/'plan.json', p)
    rows = []
    for j in p['jobs']:
        r = result(j, j['task_id'] != 'rack_same_cap')
        rows.append(r)
        write_json(tmp_path/'results'/f"{j['task_id']}_{j['split']}_{j['seed']}.json", r)
    scanned = _scan_release_runs(tmp_path)
    board = build_featured_leaderboard(scanned)
    assert board['models'][0]['overall_score_100'] == summarize(p, rows)['groups']['special']['final_score_100'] == 75
    assert board['models'][0]['rank'] == 1
    assert build_featured_leaderboard(scanned + scanned)['models'][0]['rank'] is None


def test_public_privacy_and_static_exports(tmp_path):
    assert build_public_snapshot([{'model_name': 'ORDINARY_PRIVATE_MODEL'}])['leaderboards'] == {}
    client = create_app(outputs_root=tmp_path, console_enabled=False).test_client()
    for route in ('/api/runs', '/api/leaderboard', '/api/benchmark/leaderboard', '/api/task_scores'):
        assert client.get(route).status_code == 404
    assert client.get('/api/public/leaderboard').status_code == 410
    assert client.get('/tasks/').status_code == 200
    assert 'id="ordinary-leaderboard"' not in client.get('/leaderboard/').get_data(as_text=True)
    export_public_site(output_dir=tmp_path/'site', outputs_root=tmp_path/'runs', base_url='/Labference')
    assert not (tmp_path/'site/data/leaderboard-easy.json').exists()
    assert not (tmp_path/'site/data/leaderboard-hard.json').exists()
    assert json.loads((tmp_path/'site/data/public-site.json').read_text())['leaderboards'] == {}


def test_worker_commands_and_reject_unknown_contracts(tmp_path):
    args, p = plan(tmp_path)
    for job in p['jobs']:
        directory = tmp_path/job['task_id']
        directory.mkdir(exist_ok=True)
        cmd = worker_command(args, job, directory)
        assert '--seed' in cmd or '--seeds' in cmd
        if job['task_id'] != 'device_operation_hard':
            config = json.loads((directory/'protocol.json').read_text())
            assert config['evaluation']['policy_profile']['action']['representation'] == 'eef_pose'
        else:
            assert 'eai4s_eval.cli.run_benchmark_eval' in cmd
    args.tasks = 'not-a-task'
    with pytest.raises(ValueError):
        make_plan(args)
    args.tasks = None
    args.policy_provider = 'gr00t'
    with pytest.raises(ValueError, match='OpenPI'):
        make_plan(args)


def test_timeout_does_not_wait_for_stdout(tmp_path):
    code, expired = supervise([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path, .2)
    assert expired and code != 0


def test_resume_plan_is_identity_checked(tmp_path, monkeypatch):
    argv = ['run', '--output', str(tmp_path), '--group', 'ordinary', '--tasks', 'L01',
            '--mode', 'reference', '--checkpoint', 'test-v1']
    from eai4s_eval.cli import run_labference_eval as cli
    def fake_supervise(command, directory, timeout):
        raw = directory/'raw'
        raw.mkdir()
        (raw/'episode_metrics.jsonl').write_text(json.dumps({'seed': 101, 'success': True, 'valid_scene': True})+'\n')
        return 0, False
    monkeypatch.setattr(cli, 'supervise', fake_supervise)
    assert main(argv) == 0
    assert main(argv + ['--resume']) == 0
    assert len(list((tmp_path/'attempts/L01_sim_dev_101').iterdir())) == 1
    with pytest.raises(ValueError, match='identical'):
        main(argv + ['--resume', '--checkpoint', 'different'])


def test_all28_definitions_and_task_specific_policy_profiles(tmp_path):
    args, p = plan(tmp_path, group='both', seeds='101')
    assert len(p['jobs']) == 28
    jobs = {j['task_id']: j for j in p['jobs']}
    assert set(jobs) >= {'L01', 'L12', 'R01', 'R12'}
    assert jobs['device_operation_hard']['max_steps'] == 500
    from eai4s_eval.policy import load_openpi_profile
    assert load_openpi_profile(jobs['rack_same_cap']['policy_profile']).control_hz == 20
    assert set(load_openpi_profile(jobs['weighing_hard']['policy_profile']).observation_map) == {
        'camera_1_rgb', 'camera_3_rgb', 'display_rgb', 'weights_rgb'}


def test_native_protocol_configs_validate_without_simulator(tmp_path):
    args, p = plan(tmp_path, seeds='101')
    from eai4s_eval.centrifuge_v2.cli import load_evaluation_config
    from eai4s_eval.weighing_v1.cli import load_config as weighing_config
    from eai4s_eval.rack_recall_v4.cli import load_config as rack_config
    validators = {'weighing_hard': weighing_config, 'centrifuge_hard': load_evaluation_config,
                  'rack_same_cap': rack_config}
    for job in p['jobs']:
        if job['task_id'] not in validators:
            continue
        folder = tmp_path/job['task_id']
        folder.mkdir()
        worker_command(args, job, folder)
        validators[job['task_id']](str(folder/'protocol.json'))


def test_websocket_adapter_with_mock_service():
    """Exercise the real codec and adapter; this is not a trained policy."""
    import threading
    import numpy as np
    from websockets.sync.server import serve
    from eai4s_eval.policy import JointPolicyAdapter, load_openpi_profile, PolicyOutputInvalid, PolicyClientError
    from eai4s_eval.policy.openpi_websocket_client import OpenPIMessagePackCodec
    codec = OpenPIMessagePackCodec()
    seen = []
    mode = ['valid']
    def handler(connection):
        connection.send(codec.pack({'provider': 'mock', 'action_dim': 8}))
        try:
            payload = codec.unpack(connection.recv())
            seen.append(payload)
            if mode[0] == 'disconnect':
                connection.close()
            elif mode[0] == 'invalid':
                connection.send(codec.pack({'actions': np.zeros((1, 7), dtype=np.float32)}))
            else:
                connection.send(codec.pack({'actions': np.array([[.3, 0, .8, 1, 0, 0, 0, 1]], dtype=np.float32)}))
        except Exception:
            pass
    with serve(handler, '127.0.0.1', 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for case in ('valid', 'invalid', 'disconnect'):
            mode[0] = case
            adapter = JointPolicyAdapter(host='127.0.0.1', port=server.socket.getsockname()[1],
                profile=load_openpi_profile('eai4s_eval/configs/policies/openpi_weighing_ee_pos_v1.yaml'),
                backend='websocket', timeout_sec=.2)
            state = {'agent_pose': np.zeros(8, dtype=np.float32), 'language_instruction': 'test',
                     'camera_data': {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in
                                     ('camera_1_rgb', 'camera_3_rgb', 'display_rgb', 'weights_rgb')}}
            try:
                if case == 'valid':
                    assert adapter.predict(state).joint_positions.shape == (1, 8)
                else:
                    with pytest.raises((PolicyOutputInvalid, PolicyClientError)):
                        adapter.predict(state)
            finally:
                adapter.close()
        server.shutdown()
        thread.join(timeout=2)
    assert len(seen) == 3
    assert {'observation/front_image', 'observation/wrist_image', 'observation/display_image',
            'observation/weights_image'} <= set(seen[0])


def test_console_batch_report_is_private(tmp_path):
    _, p = plan(tmp_path, group='ordinary', tasks='L01', seeds='101', mode='reference')
    write_reports(tmp_path, p, [result(p['jobs'][0])])
    internal = create_app(outputs_root=tmp_path).test_client()
    assert 'L01' in internal.get('/console/batches').get_data(as_text=True)
    public = create_app(outputs_root=tmp_path, console_enabled=False).test_client()
    assert public.get('/console/batches').status_code == 404


def test_reference_first_step_does_not_query_uninitialized_controller():
    from types import SimpleNamespace
    from evaluation.runtime import BenchmarkRuntime
    runtime = BenchmarkRuntime.__new__(BenchmarkRuntime)
    class Controller:
        initialized = False
        def is_success(self):
            assert self.initialized
            return False
        def step(self, state):
            self.initialized = True
            return None, False, False
    runtime.controller = Controller()
    runtime.world = SimpleNamespace(step=lambda **kw: None, is_playing=lambda: True)
    runtime.task = SimpleNamespace(step=lambda: {}, frame_idx=0, need_reset=lambda: False)
    runtime.render = False
    runtime._terminal_hold = False
    runtime._step_index = 0
    runtime._apply_observation_contract = lambda state: None
    runtime.build_snapshot = lambda state: state
    assert runtime.step().state == {'sim_step': 0}


def test_policy_preflight_failure_writes_incomplete_report(tmp_path, monkeypatch):
    from eai4s_eval.cli import run_labference_eval as cli
    def unavailable(*args):
        raise ConnectionError('mock unavailable')
    monkeypatch.setattr(cli, 'check_policy_services', unavailable)
    code = main(['run', '--output', str(tmp_path), '--checkpoint', 'test'])
    assert code == 2
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['groups']['special']['final_score_100'] is None
    assert (tmp_path/'preflight_error.json').is_file()


def test_orphan_worker_pipe_cannot_bypass_timeout(tmp_path):
    command = [sys.executable, '-c',
               "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"]
    _, expired = supervise(command, tmp_path, .3)
    assert expired


def test_gr00t_batch_selects_zmq_and_matching_profile(tmp_path):
    args = parser().parse_args(['preflight', '--output', str(tmp_path), '--group', 'ordinary',
                               '--tasks', 'L01', '--checkpoint', 'groot-v1', '--policy-provider', 'gr00t'])
    p = make_plan(args)
    cmd = worker_command(args, p['jobs'][0], tmp_path)
    assert cmd[cmd.index('--policy-backend')+1] == 'zmq'
    assert cmd[cmd.index('--vla-port')+1] == '5555'
    assert cmd[cmd.index('--policy-profile')+1].endswith('gr00t_franka_ee_pos_v1.yaml')


def test_explicit_evaluator_errors_override_success_flag(tmp_path):
    _, p = plan(tmp_path/'out', seeds='101')
    write_json(tmp_path/'run.json', {'seed': 101, 'runtime_status': 'error', 'ranking_eligible': True})
    write_json(tmp_path/'scores.json', {'evaluation_valid': True, 'valid_success': True})
    assert normalize_result(p['jobs'][0], tmp_path, 0)['success'] is None


def test_nonfinite_budgets_are_rejected(tmp_path):
    args = parser().parse_args(['preflight', '--output', str(tmp_path), '--mode', 'reference',
                               '--task-timeout-seconds', 'nan'])
    with pytest.raises(ValueError, match='budgets'):
        make_plan(args)


def test_legacy_suite_merges_seed_runs_instead_of_last_run_winning(tmp_path):
    from eai4s_eval.eval.task_suite import TaskSuiteAggregator
    catalog = {'tasks': [{'task_id': 'L01', 'status': 'implemented'}]}
    run1 = {'manifest': {'run_id': 'a'}, 'task_summary': [{'task_id': 'L01', 'split': 'sim_dev',
            'num_trials': 2, 'num_valid_trials': 2, 'num_success': 2, 'success_rate': 1}]}
    run2 = {'manifest': {'run_id': 'b'}, 'task_summary': [{'task_id': 'L01', 'split': 'sim_dev',
            'num_trials': 3, 'num_valid_trials': 3, 'num_success': 0, 'success_rate': 0}]}
    rows = TaskSuiteAggregator(tmp_path)._task_score_table(catalog, [run1, run2, run1])
    assert rows[0]['num_trials'] == 5
    assert rows[0]['task_success_rate'] == .4


def test_ordinary_scene_position_ranges_have_ordered_bounds():
    from pathlib import Path
    import yaml
    from eai4s_eval.cli.run_labference_eval import ROOT, definitions

    for task in definitions().values():
        if task['group'] != 'ordinary':
            continue
        spec = yaml.safe_load(Path(task['task_spec']).read_text())
        config = ROOT / 'config' / (spec['runtime']['env_config'] + '.yaml')
        scene = yaml.safe_load(config.read_text())
        for obj in scene.get('task', {}).get('obj_paths', []):
            for axis, bounds in obj.get('position_range', {}).items():
                assert bounds[0] <= bounds[1], (config, obj['path'], axis, bounds)
