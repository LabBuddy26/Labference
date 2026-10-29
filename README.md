# Labference

[Paper](https://arxiv.org/abs/2610.35252) · [Website](https://lab-buddy.github.io/Labference/)

## Usage guides

- [Environment and assets](docs/INSTALLATION.md)
- [Data collection](docs/COLLECTION.md)
- [OpenPI / GR00T training and inference](docs/TRAINING.md)
- [Four special tasks and 24 ordinary tasks: automated evaluation](docs/AUTOMATED_EVALUATION.zh-CN.md)
- [Hugging Face resource publication](docs/hugging%20face发布说明.md)

This is a source release without simulator assets or model weights. Training conversion and end-to-end training are not yet verified. Reference smoke checks are not model scores.

[中文说明](README.zh-CN.md)

Labference benchmarks the reasoning–action connection in robotic laboratories:
how observed physical outcomes guide subsequent choices and how those choices
translate into actions. The Technical Report develops centrifuge loading,
feedback weighing and placement recall, with controlled evidence variations,
independent choice/execution/completion measurements and matched Oracle-assisted
continuations. Appendix A provides twelve laboratory workflows and twelve
reasoning-in-action tasks. The website additionally highlights device operation.

> **Pre-release status:** Labference is currently in the `0.x` series. APIs,
> task definitions, metrics and dataset contents may change before `v1.0.0`.

## What is included

- 24 public task definitions across Long-Horizon Procedures and Reasoning in Action;
- three detailed report settings and an additional selected device-operation workflow;
- fixed evaluation splits (`sim_clean`, `sim_dev`, `sim_randomized`);
- OpenPI WebSocket and GR00T policy adapters;
- file-based evaluation records and official-submission validation;
- a Flask console and an exportable read-only website.

## Quick start

Labference requires an NVIDIA Isaac Sim Python environment. Clone the public
repository, install the dashboard dependencies, then verify Isaac Sim:

```bash
git clone https://github.com/Lab-Buddy/Labference
cd Labference
python -m pip install -r eai4s_eval/requirements-dashboard.txt
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
```

Runtime laboratory assets are not included in the public repository or website.
Install them from an authorized internal or upstream source before attempting
simulation tasks. See [Asset licensing](docs/ASSET_LICENSING.md).

Collect one task:

```bash
python main.py --config-name level1_pick --mode collect --headless
```

Evaluate the official suite with an OpenPI WebSocket service:

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

See [Installation](docs/INSTALLATION.md) and
[Evaluation](docs/EVALUATION.md) for the complete workflow.

## Public website

The repository contains the source for the task catalogue, object catalogue,
dataset summary and leaderboard. Public links are configured at release time;
Paper and GitHub links that are not yet published appear as “Coming soon”.
Configure `links.paper`, `links.code`, `links.homepage` and `links.dataset` in
`eai4s_eval/configs/site/public_site_v1.json`. An empty homepage defaults to the
site root, including the static export path prefix. Institution logos are
configured in `partners` and included when exporting public media.

## Licences

Software owned by EAI4S Contributors is licensed under the MIT License. Data
files explicitly published as a Labference dataset release are licensed under
[CC BY 4.0](DATASET_LICENSE.md). USD files, robot models, textures, rendered
images, videos, checkpoints and third-party products are not included in the
public release. See
[NOTICE.md](NOTICE.md), [ASSETS_AND_LICENSES.md](ASSETS_AND_LICENSES.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

Paper and archival citation information will be added when available. Until
then, cite the repository version and source commit recorded in
`PUBLIC_RELEASE.json`.

## Contributing

The internal development repository is the source of truth. Public changes
are reviewed, integrated internally, and republished through a generated sync
pull request. Please open an issue before preparing a substantial change.
