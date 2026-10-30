# Labference

[论文](https://arxiv.org/abs/2610.35252) · [官网](https://labbuddy26.github.io/Labference/)

## 使用文档

- [环境与资产安装](docs/INSTALLATION.md)
- [单任务与批量采集](docs/COLLECTION.md)
- [OpenPI / GR00T训练及推理](docs/TRAINING.md)
- [四特殊任务及24普通任务自动评测](docs/AUTOMATED_EVALUATION.zh-CN.md)
- [Hugging Face发布](docs/hugging%20face发布说明.md)

本次发布程序源码，不附带仿真资产、原始轨迹或模型权重。训练数据转换及端到端训练尚未验证，参考仿真不代表正式模型成绩。

[English](README.md)

Labference 面向机器人实验室中的推理—动作连接，评估实际物理结果如何指导后续操作选择，
以及这些选择如何转化为可靠动作。技术报告重点展开离心机装载、反馈称重和试管位置回忆，
通过受控证据变化及匹配的 Oracle 辅助续跑，分别分析操作选择、执行与完成。
附录 A 描述十二个实验室工作流和十二个动作中推理任务；网页另保留设备操作的精选展示。

> **预发布状态：** 当前属于 `0.x` 系列。在 `v1.0.0` 前，接口、任务定义、指标和
> 数据集内容仍可能调整。

## 当前内容

- Long-Horizon Procedures 与 Reasoning in Action 共 24 个公开任务；
- 三类论文重点场景及额外精选的设备操作工作流；
- 固定的 `sim_clean`、`sim_dev`、`sim_randomized` 评测分集；
- OpenPI WebSocket 与 GR00T 策略适配；
- 文件化评测记录、官方提交校验和网页榜单；
- Flask 管理控制台及可静态导出的公开网站。

## 快速开始

Labference 需要 NVIDIA Isaac Sim Python 环境：

```bash
git clone https://github.com/LabBuddy26/Labference
cd Labference
python -m pip install -r eai4s_eval/requirements-dashboard.txt
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
```

公共源码仓库和网页不提供实验室运行资产。启动仿真前，需要从已授权的内部或上游
来源安装资产，并阅读[资产许可说明](docs/ASSET_LICENSING.md)。

采集单个任务：

```bash
python main.py --config-name level1_pick --mode collect --headless
```

通过 OpenPI WebSocket 服务运行官方评测套件：

```bash
python -m eai4s_eval.cli.run_benchmark_suite_eval \
  --task-suite official_evaluation_suites_v1 \
  --model-name OpenPI \
  --checkpoint <checkpoint-name> \
  --policy-provider openpi \
  --policy-backend websocket \
  --vla-host 127.0.0.1 --vla-port 8000 \
  --all-splits --num-episodes 50 --headless
```

详细流程见 `docs/INSTALLATION.md` 和 `docs/EVALUATION.md`。

## 许可证

EAI4S Contributors 拥有权利的软件代码采用 MIT License。明确作为 Labference 数据集
发布的数据文件采用 CC BY 4.0，完整说明见 `DATASET_LICENSE.md`。USD、机器人模型、
纹理、渲染图片、视频、模型权重和第三方产品不包含在公共发布目录中。详情见
`NOTICE.md`、`ASSETS_AND_LICENSES.md` 和 `THIRD_PARTY_NOTICES.md`。

## 贡献方式

内部开发仓库是唯一事实源。公开仓库中的修改先回流内部仓库，通过测试后再由生成的
同步 PR 发布。准备较大修改前请先创建 issue。
