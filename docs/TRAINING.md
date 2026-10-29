# 模型训练、推理与评测

本仓库提供仿真采集、策略适配与评测。训练使用独立的上游OpenPI或GR00T工程，不包含已验证的一键训练管线。

## 数据准备

按episode划分训练与验证，避免同一轨迹泄漏。明确相机顺序、CHW/HWC、采样频率、末端位置与wxyz四元数及gripper语义。采集HDF5不等于上游训练格式；根据固定版本上游要求转换，保存数据统计及归一化参数。当前未提供覆盖所有任务的已验证转换器，不能直接把HDF5路径作为训练参数。

## OpenPI

本地检查到上游工程提交为`f6042bd`，包含`scripts/train.py`；这只是入口检查，不代表训练已验证。GR00T本地提交为`2b5c671`，未发现旧版`scripts/gr00t_finetune.py`，不得套用旧版本命令。

使用 https://github.com/Physical-Intelligence/openpi 的独立环境，固定实际使用的提交。注册数据配置、观测键和动作转换，按该版本训练文档启动训练。记录训练配置名和检查点目录，供本仓库服务加载。未确认上游版本前不提供可能失效的训练命令。

```bash
python -m eai4s_eval.policy_servers.openpi_server --help
```

服务所需config必须是上游训练配置注册表中的真实名称，checkpoint应包含匹配的模型及统计文件。

## GR00T

使用 https://github.com/NVIDIA/Isaac-GR00T 的独立环境，固定版本并按其文档准备数据、embodiment和modality配置。不同版本训练入口及数据格式不同；本仓库提供ZMQ适配，不保证所有上游检查点兼容。训练和转换目前未进行端到端验证。

## 推理契约与验收

参考`eai4s_eval/configs/policies/`的对应profile。普通任务支持OpenPI和GR00T；新版特殊批量评测当前支持OpenPI WebSocket末端8维动作。称重需要额外display/weights图像，试管回忆使用20Hz，不能直接复用普通双相机10Hz配置。

先加载模型验证一次观测请求，再执行[正式评测](AUTOMATED_EVALUATION.zh-CN.md)。记录上游提交、数据版本、训练配置、模型revision和评测版本；参考执行结果不能替代模型成绩。HF模型发布模板见[hugging face发布说明](hugging%20face发布说明.md)。
