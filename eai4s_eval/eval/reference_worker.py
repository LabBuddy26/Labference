"""Engineering-only controller execution judged by the normal physical evaluator."""
from __future__ import annotations


class DiscardRecorder:
    """Exercise reference actions without writing or accepting collection data."""
    episode_count = 0

    def cache_step(self, *args, **kwargs):
        pass

    def clear_cache(self, *args, **kwargs):
        pass

    def write_cached_data(self, *args, **kwargs):
        self.episode_count += 1

    def set_task_properties(self, *args, **kwargs):
        pass

    def close(self):
        pass


class ReferenceBackend:
    backend_id = 'reference_physical_validation'

    def __init__(self, runtime, control_hz, physics_hz):
        self.runtime = runtime
        self.stride = physics_hz // control_hz
        self.frames = 0

    def reset_episode(self, **kwargs):
        self.frames = 0
        return self.runtime.reset_episode(**kwargs)

    def step(self):
        result = self.runtime.step()
        if result.state is not None:
            self.frames += 1
        return result

    def policy_stats(self):
        return dict(executed_action_steps=self.frames // self.stride, policy_calls=0,
                    sim_steps=self.frames, latencies_ms=[])

    def actual_state(self):
        return self.runtime.actual_state()

    def enter_terminal_hold(self):
        self.runtime.enter_terminal_hold()

    def close(self):
        self.runtime.close()


def run_reference(args):
    from pathlib import Path
    from eai4s_eval.eval.runner import load_task_spec
    from eai4s_eval.cli.run_benchmark_eval import _resolve_task_spec
    # Imported only after parsing/preflight; no external model connection.
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    backend = None
    try:
        import hydra
        from omegaconf import open_dict
        import controllers.base_controller as base
        from evaluation.runtime import BenchmarkRuntime
        from eai4s_eval.eval.benchmark_runner import BenchmarkTaskEvalRunner
        spec = load_task_spec(str(_resolve_task_spec(args.task_spec)), 'franka', '', args.max_steps, None)
        spec['max_steps'] = args.max_steps
        config_name = args.config_name or spec['runtime']['env_config']
        with hydra.initialize_config_dir(config_dir=str(Path(args.benchmark_root).resolve()/'config'), version_base=None):
            cfg = hydra.compose(config_name=config_name)
        with open_dict(cfg):
            cfg.mode = 'collect'
            cfg.max_episodes = 1
            cfg.multi_run.run_dir = str(Path(args.output_root)/str(args.run_id)/'discarded_collection')
        base.create_collector = lambda *a, **k: DiscardRecorder()
        runtime = BenchmarkRuntime(cfg=cfg, simulation_app=app, backend=args.backend, render=True)
        backend = ReferenceBackend(runtime, int(spec.get('control_hz', 10)), int(spec.get('physics_hz', 60)))
        args.model_name = args.model_name or 'reference-engineering'
        args.checkpoint = args.checkpoint or 'reference-only'
        runner = BenchmarkTaskEvalRunner(args=args, backend=backend, task_spec=spec,
                                        policy_metadata={'provider': 'reference', 'profile_id': 'reference-only',
                                                         'action_space': 'joint', 'ranking_eligible': False})
        runner.run()
        return 0
    except BaseException as exc:
        import traceback
        from eai4s_eval.eval.batch_results import write_json
        # Kit shutdown may terminate stdio before Python prints an uncaught
        # exception. Persist/print it before closing the simulator.
        write_json(Path(args.output_root)/str(args.run_id)/'reference_error.json',
                   {'error_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        raise
    finally:
        if backend:
            backend.close()
        app.close()
