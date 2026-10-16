# EAI4S LabBench

[English](README.md)

LabBench 是面向视觉—语言—动作智能体的化学实验室操作数据集和评测基准，包含
Long-Horizon Procedures 与 Reasoning in Action 两类共 24 个普通任务，以及
4 个精选优化的特殊长程实验室任务。通过同步观测、机器人状态、关节动作、末端执行器
动作和夹爪记录，支持数据采集与模型评测；特殊任务与普通任务分别展示评测榜单。

## 四个特殊长程任务

| 任务 | 内容简介 | 当前公开展示状态 |
| --- | --- | --- |
| 反馈称重（`weighing_v1`） | 根据天平实时反馈调整标准砝码，使目标重量稳定达成。 | 已展示成功采集视频 |
| 设备操作（`device_operation_v1`） | 完成设备与烧杯的多阶段操作，应对设备门的干扰变化。 | 已展示成功采集视频 |
| 离心机装载（`centrifuge_v2`） | 将试管插入转子相对孔位，并保证放置稳定。 | 待实现（公开展示状态） |
| 试管架位置回忆（`rack_recall_v4`） | 在遮挡前后记忆试管孔位，并从试管架取回正确试管。 | 待实现（公开展示状态） |

首页优先展示这四个任务，目前仅播放称重、设备操作两个任务的成功主视角视频，
另外两个保留待实现标记。[Tasks 页面](https://zhaochunhai.github.io/LabBench/tasks/) 先展示四个
特殊任务，再展示 24 个普通任务及其演示视频。
[评测榜单](https://zhaochunhai.github.io/LabBench/leaderboard/) 将四个特殊任务与普通任务分别展示，
不合并为一个总分。成功采集演示不等同于模型评测成绩，也不代表四个任务均已完成。

## 公开预览

- **项目主页：** [打开 LabBench 基准主页](https://zhaochunhai.github.io/LabBench/)
- **数据集：** [通过百度网盘下载](https://pan.baidu.com/s/1zyJL3zxHqI_j67hUkUF6lw?pwd=gtc4)
- **代码仓库：** [https://github.com/zhaochunhai/LabBench](https://github.com/zhaochunhai/LabBench)

网页公开任务目录、物体元数据、数据集统计、已验证榜单、WebP 物体预览和 MP4
任务演示。USD、网格、纹理、仿真资产、模型权重和内部评测记录不会对外分发。

## 源码发布状态

当前实现正在进行源码、依赖和再分发审查，本预览仓库暂不包含程序源码。代码实现和
审查全部完成后，将在该仓库发布经过清理的源码快照，不导入内部 Git 历史。

## 许可证

仓库文档以及后续发布的第一方软件采用 MIT License。明确作为 LabBench 数据集发布的
数据文件采用 CC BY 4.0，详见 [DATASET_LICENSE.md](DATASET_LICENSE.md)。网页中的
预览媒体不代表其底层仿真资产源文件可以下载或再分发，详见 [NOTICE.md](NOTICE.md)。

## 引用

正式论文信息发布前，请引用公开仓库地址、数据集版本和公开发布标识。机器可读信息见
[CITATION.cff](CITATION.cff)。
