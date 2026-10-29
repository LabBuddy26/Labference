# 自动评测验证记录（2026-10-28）

## 范围与结论

四个特殊长程任务与24个普通任务均已通过一次限步参考仿真检查：seed为101，
每项最多120动作步，子进程墙钟上限300秒。28项均产生有效终态记录，
能够读取物理判定、区分成功与失败、计算各自组内分数并输出汇总文件。

这是评测链路的工程验证，不能据此宣称28项控制器全部能够完成完整任务，
也不是模型成绩。当前没有连接真实模型服务；模型正式评测仍需使用真实检查点运行。
参考执行和模拟策略服务的结果均不能进入模型榜单。

## 可审计输出

运行目录：`outputs/eval_validation/20261028/bounded28_seed101/`。

- `plan.json`：任务、seed、模式、预算和配置身份。
- `attempts/`：每次独立子进程的日志及原始结果，包括修复前失败的尝试。
- `results/`：每个任务seed的归一化终态记录。
- `summary.json`、`task_scores.csv`、`report.md`：分组分数、有效覆盖与失败原因。
- `episode_metrics.jsonl`：用于汇总的终态记录快照。

本次四个特殊任务在120步预算内均未完成，参考组分数为0；普通任务有5项成功
（R01、R03、R04、R05、R12），普通组宏平均为20.8333分。两组独立统计，
不计算28项联合总分。以上数值仅用于核对计算链路，不作为模型能力结论。

另有较长参考运行：`L01_smoke` 的L01和 `all28_v2` 的称重任务得到物理成功。
`all28_v2` 的整批运行曾被中断，不能作为完整28项测试结果。

## 仿真中修复的问题

1. L03、L05：首步控制器尚未初始化state时，运行时提前调用成功查询，导致异常。
   改为执行首步控制后获取结果；终态保持阶段才单独查询成功状态。
2. R06：第二根试管Y范围写为`[-0.2, -0.3]`，场景重置拒绝此范围。
   改为`[-0.3, -0.2]`，并对普通任务坐标区间增加回归检查。
3. 离心机：旧版测量回放评分未输出`evaluation_valid`，导致物理失败与证据无效混淆。
   现在根据测量回放验证与测量错误数独立输出有效性。
4. 参考子进程：Kit退出可能掩盖Python异常；现在退出前保存`reference_error.json`，
   汇总时展示具体错误，不将其计为模型失败。

## 自动测试

使用`env_isaaclab`环境执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/eai4s_eval tests/test_eval_mode_config.py tests/test_public_release.py \
  tests/test_centrifuge_v2.py tests/test_centrifuge_online_v2.py --tb=short
```

结果：261 passed、10 skipped、129 subtests passed。
覆盖手算分数、完整性约束、重试、身份隔离、超时、模拟WebSocket策略契约、
服务断连、非法动作、公开页面/API/静态导出隐私以及旧版多seed合并。
跳过的测试未计入通过数。

正式评测指令与使用说明见[自动评测说明](AUTOMATED_EVALUATION.zh-CN.md)。
