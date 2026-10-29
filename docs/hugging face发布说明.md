# Labference Hugging Face 发布说明

正式官网继续使用 https://zhaochunhai.github.io/Labference/ ，Hugging Face负责数据、仿真资源与模型分发。论文：https://arxiv.org/abs/2610.35252 。本文账号、路径和版本均需替换为真实值。

## 页面与仓库设计

| 资源 | 类型 | 建议名称 |
|---|---|---|
| 项目统一入口 | Collection | Labference |
| HDF5轨迹 | Dataset | Labference-Dataset |
| 环境配置与可分发资产 | Dataset | Labference-Simulation |
| 每种架构的策略权重 | Model | Labference-模型名 |

Collection收录以上仓库、论文和官网链接。暂不需要Space；以后可用Space展示榜单与视频，共享官网同一份审核成绩，避免重复维护。

数据集卡依次介绍项目、四特殊与24普通任务、真实数量与容量、训练验证划分、相机及action/gripper字段、读取示例、下载方法、许可和引用。保留HDF5格式，不能承诺HF自动展示逐帧预览。

仿真资源卡说明Isaac Sim/Isaac Lab/Python版本、代码提交、资产下载位置、路径配置、采集及评测命令。GitHub承载代码，不上传整个Conda环境或Isaac Sim安装目录。当前资产许可清单存在pending及不可再分发记录；只能打包已确认允许分发的文件，其余说明获取方法。不要直接上传整个私有仓库。

每种模型架构分别建Model仓库。记录上游base_model及revision、训练数据版本、训练方法、检查点、归一化统计、图像处理器、相机顺序、动作维度与坐标系、频率和gripper约定。微调策略称baseline或fine-tuned policy，不宣称从头训练的基础模型。Adapter需列出所需上游权重及加载方法；完整权重和adapter不可混淆。沿用真实框架格式，不自动转为safetensors或宣称支持Transformers。

## 本地整理

在源码仓库外准备三个独立发布目录，各含README.md、实际许可文件及manifest清单。数据按任务组织HDF5；环境按configs/assets组织；模型保留推理必需文件。清单包含相对路径、文件大小、SHA256与版本。数据版本、环境版本、代码提交和模型revision应互相关联。

模板见本文同级 `hf_templates/`。将占位符全部替换，统计真实数量，审核资产及上游模型许可；发布目录不要包含Token、内部路径、采集日志、优化器状态或未审核文件。

## 创建与上传

1. 登录HF，选择个人或组织命名空间，创建上述两个Dataset仓库及实际需要的Model仓库，初始设为Private。
2. 建立独立上传环境，创建具有目标仓库写权限的Token，交互登录，不将Token写入文档。

```bash
conda create -n env_hf_release python=3.11 -y
conda activate env_hf_release
pip install -U huggingface_hub
hf auth login

HF_NAMESPACE="替换为真实账号或组织名"
hf upload "$HF_NAMESPACE/Labference-Dataset" /path/to/dataset-release --repo-type dataset
hf upload "$HF_NAMESPACE/Labference-Simulation" /path/to/simulation-release --repo-type dataset
hf upload "$HF_NAMESPACE/Labference-模型名" /path/to/model-release --repo-type model
```

大目录可使用`hf upload-large-folder --help`核对当前CLI用法，使用其任务缓存恢复上传；重试时保留本地发布目录及缓存。上传前核对HF当前容量和文件数量限制，必要时申请存储额度；不承诺任意容量免费。不要将所有任务打成一个不可按任务下载的大包。

## 下载验收与公开

```bash
hf download "$HF_NAMESPACE/Labference-Dataset" --repo-type dataset --local-dir /path/to/check/data
hf download "$HF_NAMESPACE/Labference-Simulation" --repo-type dataset --local-dir /path/to/check/simulation
hf download "$HF_NAMESPACE/Labference-模型名" --repo-type model --local-dir /path/to/check/model
```

正式复现需增加`--revision`指定审核的commit或tag。核对SHA256，在新目录读取一个任务所有必要字段，安装环境后启动一个任务，加载下载的模型启动服务并完成一次真实评测。HDF5读取示例应覆盖实际相机、actions_ee与gripper字段，不把采集文件存在当成物理成功。

验收通过后，在各仓库Settings改为Public，加入Collection，并用未登录浏览器检查链接可访问性。若采用Gated访问，文档需明确申请方式，不能宣称匿名可直接下载。

## 官网与榜单同步

在`eai4s_eval/configs/site/public_site_v1.json`的links中填写：

- huggingface：Collection真实地址。
- hf_dataset：数据集仓库地址。
- hf_simulator：仿真资源仓库地址。
- hf_models：模型Collection地址。

空地址显示Coming soon。保留百度网盘入口，不公开提取码。榜单沿用四特殊任务等权评分、不同split分别排名；部分结果不排名，24普通任务成绩仅内部可见。参考控制器及模拟服务不发布为模型成绩。

```bash
cd /path/to/Labference
conda activate env_isaaclab
mkdir -p outputs/public_release/empty_eval
python -m eai4s_eval.cli.export_public_site \
  --output-dir outputs/public_release/site-dist \
  --outputs-root outputs/public_release/empty_eval \
  --release-outputs-root outputs/public_release/empty_eval \
  --base-url /Labference --include-public-media
```

空评测目录将不包含模型成绩；保留正式榜单时替换为审核结果目录。网站预览后同步到独立LabBench-pages仓库并推送gh-pages，不能把构建输出目录设为发布Git仓库根目录。源码main使用单独的build/verify流程，不能将网站文件覆盖源码。

官方参考：[Collections](https://huggingface.co/docs/hub/collections)、[数据集](https://huggingface.co/docs/hub/datasets-adding)、[Model Cards](https://huggingface.co/docs/hub/model-cards)、[上传](https://huggingface.co/docs/huggingface_hub/guides/upload)、[CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)、[存储限制](https://huggingface.co/docs/hub/storage-limits)。
