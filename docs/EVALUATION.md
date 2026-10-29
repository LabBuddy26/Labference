# Evaluation

## Policy service

Start OpenPI or GR00T separately. OpenPI uses MessagePack over WebSocket and
the evaluator expects front RGB, wrist RGB, an 8-D robot state and a language
prompt. Returned actions must be finite `float32[T,8]` values.

## Official suite

Always provide both model and checkpoint names so records cannot be attributed
to the wrong model:

```bash
python -m eai4s_eval.cli.run_benchmark_suite_eval \
  --task-suite official_evaluation_suites_v1 \
  --model-name <model-name> \
  --checkpoint <checkpoint-name> \
  --policy-provider openpi \
  --policy-backend websocket \
  --vla-host 127.0.0.1 --vla-port 8000 \
  --required-splits sim_clean \
  --num-episodes 50 --headless
```

Results are written under `outputs/eval_runs/`. Local results do not appear on
the official leaderboard until an administrator validates the submission and
adds its exact run IDs to the verified submission registry.

## Reproducibility

- `sim_clean`: nominal conditions;
- `sim_dev`: development perturbations within configured collection ranges;
- `sim_randomized`: broader position, yaw, lighting and distractor variation.

Use fixed suite seeds and retain the generated manifests with every reported
result.
