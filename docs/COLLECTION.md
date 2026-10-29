# 数据采集

环境与资产准备见[安装说明](INSTALLATION.md)。从仓库根目录运行：

```bash
python main.py --config-name level4_weighing_feedback_hard --mode collect --backend numpy --headless
python main.py --config-name level4_L09_hard_DeviceOperation --mode collect --backend numpy --headless
```

批量采集使用`python -m utils.collect_driver start --help`核对参数，通过task-list选择任务、task-targets设置目标目录及数量；使用success-episodes、consecutive-failures和task-timeout-minutes控制停止条件。目标目录应为本机可写位置，不复用作者机器路径。

检查运行报告、COLLECT_RESULT和HDF5文件共同确认采集情况。文件包含相机、机器人状态、动作、末端动作及夹爪字段；具体命名以实际collector输出为准。只有满足任务物理判定的轨迹才计为成功。不要将评测输出混入训练目录；失败日志应保留供排查。

四特殊任务和普通24项的评测见[自动评测说明](AUTOMATED_EVALUATION.zh-CN.md)。
