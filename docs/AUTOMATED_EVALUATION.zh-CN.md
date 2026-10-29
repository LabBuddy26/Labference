# Labference 自动评测与成绩汇总

## 计分与发布范围

- 默认四精选任务：`weighing_hard`、`device_operation_hard`、`centrifuge_hard`、`rack_same_cap`。
  各任务成功率乘100后等权平均，每项25%，版本为 `labference_success_macro/v2`。
- 必须四项全部计划seed都有有效结果才有最终分及榜单名次；未完成或存在无效评测时，
  仅报告暂定均分和覆盖率，不把缺失项算0，也不宣称取得完整总分。
- 普通任务 L01–L12、R01–R12 支持选跑/全跑，单独输出成功率和L/R分组宏平均，
  不与特殊任务合成总分。普通成绩不进入公开网页、API或静态JSON。
- 物理失败、策略非法动作和动作预算用尽属于有效失败（0分）；环境启动失败、缺失证据、
  服务断连或进程墙钟超时属于无效评测。后者不进入成功率分母，阻止产生最终完整分数。
- 成功与否由任务物理谓词决定，不能使用控制器阶段结束或采集文件存在代替。
- 选择、执行、质量、安全和恢复等是独立诊断，缺失保持缺失；不为填满表格复制总成功标记。

## 环境与预检查

```bash
cd /path/to/Labference
conda activate env_isaaclab
python -m eai4s_eval.cli.run_labference_eval preflight \
  --group both --mode reference --seeds 101 \
  --output outputs/eval_validation/check28
```

`preflight` 离线检查所选任务和策略文件，打印计划，不运行仿真。
`run --mode native` 在启动子进程前连接策略服务并校验已声明的策略契约。
没有真实模型服务时只能验证参考执行与模拟服务，不代表模型评测成功。

## 四个特殊任务正式模型评测

先启动实际 OpenPI 服务，再执行（模型名称、检查点改为真实值）：

```bash
python -m eai4s_eval.cli.run_labference_eval run \
  --group special --mode native \
  --model-name OpenPI --checkpoint checkpoint_30000 \
  --vla-host 127.0.0.1 --vla-port 8000 \
  --seeds 101,102,103,104,105 --split sim_dev \
  --headless --save-video \
  --output outputs/eval_runs/openpi_ckpt30000_special_v2
```

每个seed是一个独立仿真子进程，串行执行，默认墙钟超时1200秒。
动作预算按任务配置读取，`--max-steps` 是显式覆盖。设备操作使用其评测TaskSpec的500动作步，
不误用采集协议中的120000帧。实际是否能在预算内完成应由评测记录反映。
可使用 `--task-timeout-seconds` 调整墙钟上限；墙钟超时是无效执行，不是模型动作预算超时。

默认称重策略输入四路图像（front/wrist/display/weights），控制频率10Hz；
试管架默认使用20Hz策略配置，其余精选任务10Hz。自定义profile必须满足该契约。
特殊批量评测当前支持OpenPI WebSocket末端8维动作；不支持的provider/backend会明确报错，
不会降级到参考控制器。普通任务保留OpenPI和GR00T既有适配。

原批量入口也可调用新流程：

```bash
python -m eai4s_eval.cli.run_benchmark_suite_eval \
  --task-suite labference_special_featured_v1 \
  --model-name OpenPI --checkpoint checkpoint_30000 \
  --seeds 101,102 --num-episodes 2 --headless \
  --run-id openpi_special_v2 --output-root outputs/eval_runs
```

该版本化批次使用显式seed列表和单个split；多split分开创建批次。
旧入口不带 `--evaluation-group`、不选择新套件时仍保持原有调用方式。
带认证的既有单任务接口不在此版本化批量桥接范围，不会静默忽略API key。
该桥接当前只支持numpy后端，并使用启动命令所在的Python环境；
非默认`--eval-python`或GPU后端会明确拒绝，请直接使用目标环境的Python启动。

本次实际仿真范围、结果与限制见[2026-10-28验证记录](EVALUATION_VALIDATION_20261028.zh-CN.md)。

## 普通任务选跑与全跑

```bash
# 只评测选定的普通任务，成绩仅本地可见
python -m eai4s_eval.cli.run_labference_eval run \
  --group ordinary --tasks L01,L07,R01,R10 \
  --mode native --model-name OpenPI --checkpoint checkpoint_30000 \
  --seeds 101,102 --headless \
  --output outputs/eval_runs/openpi_ordinary_subset

# 全24项：不指定 --tasks
python -m eai4s_eval.cli.run_labference_eval run \
  --group ordinary --mode native \
  --model-name OpenPI --checkpoint checkpoint_30000 \
  --seeds 101,102 --headless \
  --output outputs/eval_runs/openpi_ordinary24
```

`--group both` 一次调度四精选任务和全部24普通任务，仍然分别计分，不合成联合分数。
`--tasks` 中的任务必须属于所选group；如混选两组，使用 `--group both`。

其他特殊难度可显式选择 `weighing_easy`、`centrifuge_easy`、`rack_easy`、`rack_distinct_cap`。
其结果是所选协议成绩，不混入默认四精选总分；历史八协议汇总函数仍保留旧含义。

## 不依赖模型的工程验证

```bash
python -m eai4s_eval.cli.run_labference_eval run \
  --group both --mode reference --seeds 101 --headless \
  --task-timeout-seconds 1200 \
  --output outputs/eval_validation/reference28_seed101
```

普通任务及设备操作使用原参考控制器执行，由现有独立物理评测器判定；不写采集HDF5。
其余特殊任务调用自身参考协议。工程验证记录明确标记为reference，不进入模型榜单。
参考任务失败仍记录真实失败，不能据此声称控制器已经全部调通。

进度与日志：

```bash
tail -f outputs/eval_validation/reference28_seed101/attempts/weighing_hard_sim_dev_101/0000/worker.log
```

控制台每15秒输出心跳，防止无日志时无法分辨进程是否仍受监控。

## 续跑、离线重汇总

续跑时使用原命令和原输出路径，加 `--resume`。有效成功和有效失败都不重复运行，
只重试无效/尚未完成的轮次。修改seed、模型、检查点、任务或策略文件后，应创建新输出目录；
配置指纹不一致时拒绝混用。

```bash
# 仅按保存的计划与终态记录汇总，不启动仿真
python -m eai4s_eval.cli.run_labference_eval summarize \
  --output outputs/eval_runs/openpi_ckpt30000_special_v2

# 修复读取器后，重新读取保留的原始证据再汇总（不改写原始证据）
python -m eai4s_eval.cli.run_labference_eval summarize \
  --output outputs/eval_runs/openpi_ckpt30000_special_v2 --refresh-records
```

旧版离心机评分缺少evaluation_valid时，仅在测量回放已验证、无原始测量错误且退出状态正常的
条件下兼容识别有效性，并写明 `validity_source=legacy_verified_measurement_replay`。
不根据任务是否成功推断评测有效性。

## 文件与查看方式

| 文件 | 内容 |
| --- | --- |
| `plan.json` | 固定任务/seed、模型身份、配置指纹和预算 |
| `results/*.json` | 每个计划轮次唯一的终态记录，避免重复统计 |
| `attempts/*/*/worker.log` | 各次启动日志，重试不覆盖旧证据 |
| `episode_metrics.jsonl` | 当前终态记录快照 |
| `task_scores.csv` | 单任务计划/完成/有效/失败数量和分数 |
| `summary.json` | 两组分数、覆盖率、是否完整、是否可排名 |
| `report.md` | 可读报告 |

`run` 退出码0表示所有计划轮次有效完成，不代表任务全部成功；退出码2表示仍有无效或缺失轮次。
`summarize` 是否有完整最终分应查看JSON字段，不能仅凭命令退出码判断。

内部控制台：

```bash
python -m eai4s_eval.server.app --host 127.0.0.1 --port 8767 \
  --outputs-root outputs/eval_runs
```

访问 `http://127.0.0.1:8767/console/batches` 查看批次报告。
内部控制台无对外认证配置时只绑定localhost，不应直接公开。
使用 `--disable-console` 时，内部API及批次页面均不可访问。

公开页面只发布完整四精选模型成绩，部分成绩显示Incomplete且无排名。
普通公开榜单接口 `/api/public/leaderboard` 已退役（410），普通成绩JSON不再静态导出。
部署时应使用已有带 `--delete` 且保护 `.git`/`CNAME` 的同步流程清理旧静态成绩文件。
本次实现不自动推送、部署，也不删除已发布历史分支中的记录。
