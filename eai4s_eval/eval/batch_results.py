"""Auditable success-rate summaries; no simulator or policy imports."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

FEATURED = ('weighing_hard', 'device_operation_hard', 'centrifuge_hard', 'rack_same_cap')
SCORE_VERSION = 'labference_success_macro/v2'


def _finite_tree(value):
    if isinstance(value, dict):
        return {k: _finite_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_tree(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, value: dict) -> None:
    """Replace complete records atomically, including when a worker crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def normalize_result(job: dict, raw: Path, returncode: int, *, timed_out: bool = False) -> dict:
    row = {k: job[k] for k in ('task_id', 'group', 'seed', 'split', 'mode', 'model_name', 'checkpoint')}
    row.update(evaluation_valid=False, success=None, termination_reason='missing_result',
               returncode=returncode, raw_dir=str(raw), ranking_eligible=False, diagnostics={})
    # Each batch worker is exactly one planned seed. Never combine other runs.
    ordinary_files = sorted(raw.rglob('episode_metrics.jsonl'))
    candidates = []
    for path in ordinary_files:
        try:
            for line in path.read_text().splitlines():
                item = json.loads(line)
                if item.get('seed') == job['seed']:
                    candidates.append((path, item))
        except (OSError, ValueError):
            row['termination_reason'] = 'corrupt_result'
            return row
    if candidates:
        # Retries can record invalid initialization attempts. Select the last
        # event for the seed, not a success from a previous invalid attempt.
        path, item = candidates[-1]
        reason = item.get('termination_reason') or item.get('failure_type') or 'finished'
        valid = item.get('valid_scene') is True and isinstance(item.get('success'), bool)
        if reason in {'policy_request_failed', 'service_error', 'runtime_error'}:
            valid = False
        row.update(evaluation_valid=valid, success=item.get('success') if valid else None,
                   termination_reason=reason, evidence=str(path),
                   steps=item.get('episode_steps'), duration_seconds=item.get('duration_sec'))
        # Diagnostics must be measurements, never aliases of overall success.
        row['diagnostics'] = {key: item[key] for key in (
            'goal_pass', 'sequence_pass', 'terminal_pass', 'stability_pass',
            'safety_valid', 'quality_metrics', 'failure_category') if key in item}
    else:
        pairs = [p for p in raw.rglob('run.json') if p.with_name('scores.json').is_file()]
        pairs = [p for p in pairs if read_json(p).get('seed') == job['seed']]
        if len(pairs) == 1:
            path = pairs[0]
            manifest, scores = read_json(path), read_json(path.with_name('scores.json'))
            valid = scores.get('evaluation_valid', manifest.get('evaluation_valid')) is True
            # v1 centrifuge reports did not serialize evaluation_valid. Its
            # measured replay flag is an independently checked validity
            # predicate, not an inference from controller/task completion.
            if (job['task_id'].startswith('centrifuge_') and 'evaluation_valid' not in scores
                    and manifest.get('runtime_status') in {'finished', 'time_limit', 'stopped_after_violation'}
                    and scores.get('physical_replay_verified') is True
                    and scores.get('raw_measurement_error_count') == 0):
                valid = True
                row['validity_source'] = 'legacy_verified_measurement_replay'
            success = scores.get('valid_success')
            valid = valid and isinstance(success, bool)
            if manifest.get('runtime_status') in {'error', 'service_error', 'initializing', 'running'}:
                valid = False
            row.update(evaluation_valid=valid, success=success if valid else None,
                       termination_reason=manifest.get('runtime_status', 'finished'), evidence=str(path))
            row['source_ranking_eligible'] = manifest.get('ranking_eligible') is True
            row['diagnostics'] = {k: v for k, v in scores.items()
                                  if k not in {'valid_success', 'evaluation_valid'}
                                  and (isinstance(v, bool) or isinstance(v, (int, float))
                                       and math.isfinite(v))}
        elif len(pairs) > 1:
            row['termination_reason'] = 'ambiguous_result'
    if returncode != 0 or timed_out:
        reason = row['termination_reason']
        row.update(evaluation_valid=False, success=None,
                   termination_reason=('worker_timeout' if timed_out else reason if reason in
                                       {'runtime_error', 'policy_request_failed', 'invalid_scripted_precondition', 'corrupt_result'}
                                       else 'worker_error'))
    if row['success'] is None:
        errors = sorted(raw.rglob('reference_error.json'))
        if errors:
            error = read_json(errors[-1])
            row.update(termination_reason='runtime_error', error=error.get('message'),
                       error_type=error.get('error_type'))
    row['ranking_eligible'] = (job['mode'] == 'native' and row['evaluation_valid']
                               and row.get('source_ranking_eligible', True))
    return _finite_tree(row)


def summarize(plan: dict, rows: list[dict]) -> dict:
    """Use only planned terminal records of one identity; never merge history."""
    identities = {(j['task_id'], j['seed'], j['split']): j for j in plan['jobs']}
    selected = {}
    for row in rows:
        key = (row.get('task_id'), row.get('seed'), row.get('split'))
        job = identities.get(key)
        if not job or any(row.get(k) != job[k] for k in ('model_name', 'checkpoint', 'mode', 'group')):
            raise ValueError('record identity is outside this evaluation plan')
        if key in selected:
            raise ValueError('duplicate terminal record for task/seed/split')
        selected[key] = row
    tasks = []
    for task_id in dict.fromkeys(j['task_id'] for j in plan['jobs']):
        jobs = [j for j in plan['jobs'] if j['task_id'] == task_id]
        items = [r for key, r in selected.items() if key[0] == task_id]
        valid = [r for r in items if r.get('evaluation_valid') is True and isinstance(r.get('success'), bool)]
        successes = sum(r['success'] for r in valid)
        tasks.append(dict(task_id=task_id, group=jobs[0]['group'], planned=len(jobs),
                          completed=len(items), valid=len(valid), invalid=len(items)-len(valid),
                          successes=successes, failures=len(valid)-successes,
                          success_rate=successes/len(valid) if valid else None,
                          score_100=100*successes/len(valid) if valid else None,
                          complete=len(valid) == len(jobs),
                          failure_reasons={reason: sum(r.get('termination_reason') == reason for r in items)
                                           for reason in sorted({str(r.get('termination_reason')) for r in items
                                                                 if r.get('success') is not True})}))
    groups = {}
    for group in ('special', 'ordinary'):
        members = [t for t in tasks if t['group'] == group]
        if not members:
            continue
        evaluated = [t for t in members if t['score_100'] is not None]
        provisional = sum(t['score_100'] for t in evaluated)/len(evaluated) if evaluated else None
        complete = all(t['complete'] for t in members)
        full_special = group != 'special' or set(t['task_id'] for t in members) == set(FEATURED)
        groups[group] = dict(planned_tasks=len(members), evaluated_tasks=len(evaluated),
                             complete=complete, provisional_score_100=provisional,
                             final_score_100=provisional if complete else None,
                             featured_final_score_100=provisional if complete and full_special and group == 'special' else None,
                             scope='featured_four' if group == 'special' and full_special else 'selected_tasks',
                             ranking_eligible=(group == 'special' and full_special and complete and plan['mode'] == 'native'
                                               and all(r.get('ranking_eligible') is True for r in selected.values()
                                                       if r['group'] == 'special')))
        if group == 'ordinary':
            if {t['task_id'] for t in members} == {f'{prefix}{n:02d}' for prefix in ('L', 'R') for n in range(1, 13)}:
                groups[group]['scope'] = 'ordinary_24'
            groups[group]['tracks'] = {}
            for track in ('L', 'R'):
                values = [t['score_100'] for t in evaluated if t['task_id'].startswith(track)]
                groups[group]['tracks'][track] = sum(values)/len(values) if values else None
    return dict(schema_version='labference_batch_summary/v1', score_version=SCORE_VERSION,
                run_id=plan['run_id'], model_name=plan['model_name'], checkpoint=plan['checkpoint'],
                mode=plan['mode'], tasks=tasks, groups=groups, ordinary_results_public=False)


def write_reports(root: Path, plan: dict, rows: list[dict]) -> dict:
    result = summarize(plan, rows)
    write_json(root/'summary.json', result)
    (root/'episode_metrics.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows), encoding='utf-8')
    with (root/'task_scores.csv').open('w', newline='', encoding='utf-8') as stream:
        keys = ('task_id', 'group', 'planned', 'completed', 'valid', 'invalid', 'successes',
                'failures', 'success_rate', 'score_100', 'complete')
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['tasks'])
    lines = ['# Labference evaluation', '', f"Run: {plan['run_id']} · {plan['mode']}", '',
             'Reference/smoke runs are not model submissions. Ordinary scores are local only.', '',
             '| Task | Valid / planned | Successes | Score | Complete |', '|---|---:|---:|---:|---|']
    for t in result['tasks']:
        score = 'N/A' if t['score_100'] is None else f"{t['score_100']:.2f}"
        lines.append(f"| {t['task_id']} | {t['valid']}/{t['planned']} | {t['successes']} | {score} | {t['complete']} |")
    for name, group in result['groups'].items():
        lines.extend(['', f"{name}: final={group['final_score_100']}, provisional={group['provisional_score_100']}, scope={group['scope']}"])
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
