# Labference

[中文说明](README.zh-CN.md)

**Benchmarking the Reasoning–Action Connection in Robotic Laboratories**

Labference evaluates how observed physical outcomes guide subsequent choices and
how those choices translate into robotic laboratory actions. Controlled evidence
variations test outcome-dependent behavioral adaptation, while operation selection,
physical execution and task completion are measured independently. Matched
Oracle-assisted continuations from shared checkpoints diagnose which support
improves subsequent performance. The scope is robotic laboratories and Embodied
AI for Science, not chemistry manipulation alone.

The Technical Report (October 21, 2026) develops three settings: centrifuge loading,
feedback weighing and placement recall. Appendix A describes twelve laboratory
workflows (L01–L12) and twelve reasoning-in-action tasks (R01–R12). This website
retains four selected workflow cards, including device operation as an additional
catalogue workflow. Initial experimental findings concern feedback weighing, not
validated performance across the full catalogue.

## Four special long-horizon workflows

| Workflow | Description | Public demonstration status |
| --- | --- | --- |
| Feedback weighing (`weighing_v1`) | Adjust standard weights using live scale feedback to reach a stable target. | Successful collection demo available |
| Device operation (`device_operation_v1`) | Manipulate a laboratory device and beakers through a multi-stage procedure with door disturbances. | Successful collection demo available |
| Centrifuge loading (`centrifuge_v2`) | Insert tubes into opposite rotor slots while preserving placement stability. | Simulator scene preview available |
| Rack placement recall (`rack_recall_v4`) | Remember tube locations through occlusion and retrieve the correct tube from a rack. | Simulator scene preview available |

The homepage prioritizes these four workflows and shows successful main-camera
videos only for weighing and device operation. undefined The [Tasks page](https://zhaochunhai.github.io/Labference/tasks/) presents the special
workflows first, followed by the 24 ordinary tasks and their demonstrations.
The [Leaderboard](https://zhaochunhai.github.io/Labference/leaderboard/) reports the four special
workflows separately from the ordinary benchmark. Successful collection demos
are not model evaluation scores and do not imply all four workflows are complete.

## Public preview

- **Project website:** [Open the Labference benchmark website](https://zhaochunhai.github.io/Labference/)
- **Dataset:** [Download from Baidu Netdisk](https://pan.baidu.com/s/1zyJL3zxHqI_j67hUkUF6lw)
- **Repository:** [https://github.com/zhaochunhai/Labference](https://github.com/zhaochunhai/Labference)

The website publishes the task catalogue, object metadata, dataset summary,
verified leaderboard results, rendered WebP object previews and MP4 task
demonstrations. USD files, meshes, textures, simulator assets, model
checkpoints and internal evaluation records are not distributed.

## Source release status

The implementation is undergoing source, dependency and redistribution review.
Program source code is not part of this preview repository. An audited source
snapshot will be published here when implementation and review are complete;
internal Git history will not be imported.

## Licences

Repository documentation and first-party software, when released, use the MIT
License. Data files explicitly identified as a Labference dataset release use
CC BY 4.0; see [DATASET_LICENSE.md](DATASET_LICENSE.md). Website previews do
not grant access to or redistribution rights for their underlying simulator
asset source files. See [NOTICE.md](NOTICE.md).

## Citation

Until archival paper metadata is available, cite the repository URL, dataset
version and public release identifier. Machine-readable metadata is provided
in [CITATION.cff](CITATION.cff).
