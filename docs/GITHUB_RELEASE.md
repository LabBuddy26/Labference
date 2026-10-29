# GitHub源码发布

由内部源码仓库维护者执行。网站gh-pages与源码main独立，不直接推送内部开发分支。

```bash
python -m utils.public_release build --output /path/to/Labference-source-dist --source-ref HEAD
python -m utils.public_release verify --root /path/to/Labference-source-dist
git -C /path/to/LabBench-public status -sb
git -C /path/to/LabBench-public pull --ff-only origin main
rsync -ani --delete --exclude .git --exclude CNAME --exclude .claude/ /path/to/Labference-source-dist/ /path/to/LabBench-public/
```

确认目标是独立公开仓库、远程地址正确且没有未保存改动后，把rsync的`-ani`改为`-ai`。该操作删除目标中过期文件，但保护Git目录。提交公开仓库变更并`git push origin main`。保留源提交和PUBLIC_RELEASE.json的内容哈希。不使用force push。

内部构建工具不随源码快照发布。普通用户无需执行这些维护命令；按安装、采集及评测文档使用代码。公开仓库不携带内部Git历史、原始数据、模型权重或未确认可分发的资产。
