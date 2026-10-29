"""Audit published special-task HDF5 streams without starting Isaac Sim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def audit(path: Path) -> dict:
    with h5py.File(path) as h5:
        leaves = {}
        h5.visititems(lambda name, node: leaves.update({name.rsplit('/', 1)[-1]: node})
                      if isinstance(node, h5py.Dataset) else None)
        props = json.loads(leaves['task_properties'][()])
        required = ('actions', 'actions_ee', 'action_gripper', 'camera_1_rgb', 'camera_3_rgb')
        lengths = {key: len(leaves[key]) for key in required}
        errors = []
        if not props.get('physical_success'):
            errors.append('not physically successful')
        if len(set(lengths.values())) != 1 or min(lengths.values()) < 2:
            errors.append('empty or unaligned streams')
        if leaves['actions_ee'].shape[1:] != (8,):
            errors.append('EE action must be xyz+wxyz+finger')
        for key in ('actions', 'actions_ee', 'action_gripper'):
            if not np.isfinite(leaves[key][:]).all():
                errors.append(f'{key} contains nonfinite values')
        cameras = {}
        for key, node in leaves.items():
            if not key.endswith('_rgb'):
                continue
            if len(node) != lengths['actions']:
                errors.append(f'{key} is unaligned')
            std = float(np.std(node[0])) if len(node) else 0.0
            cameras[key] = {'shape': list(node.shape), 'first_frame_std': std}
            rgb_shape = len(node.shape) == 4 and (node.shape[-1] == 3 or node.shape[1] == 3)
            if std < 1 or not rgb_shape:
                errors.append(f'{key} is blank or malformed')
        return {'path': str(path), 'passed': not errors, 'errors': errors,
                'lengths': lengths, 'cameras': cameras, 'task_properties': props}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    paths = sorted(p for p in args.root.glob('**/dataset/episode_*.h5')
                   if 'attempts' not in p.relative_to(args.root).parts)
    rows = [audit(path) for path in paths]
    print(json.dumps({'passed': bool(rows) and all(row['passed'] for row in rows),
                      'episode_count': len(rows), 'episodes': rows}, indent=2))
    return 0 if rows and all(row['passed'] for row in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
