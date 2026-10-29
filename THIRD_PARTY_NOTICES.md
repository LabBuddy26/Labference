# Third-party notices

This file records third-party software and material known to be referenced by
LabBench. It is not a substitute for the upstream licence text.

| Component | Relationship | Distribution in the public repository |
|---|---|---|
| NVIDIA Isaac Sim / Omniverse | External simulator runtime | Not bundled |
| Franka robot assets | Runtime robot representation | Pending provenance review; not bundled by the public release builder |
| OpenPI | External policy service | Not bundled |
| NVIDIA GR00T | External policy service | Not bundled |
| Flask | Dashboard dependency | Installed separately from PyPI |
| NumPy, OpenCV, PyTorch, Hydra, OmegaConf, h5py | Runtime dependencies | Installed separately; not vendored |

Some internal asset files contain NVIDIA copyright notices or have incomplete
provenance metadata. They remain available only in the internal development
workspace until redistribution rights are verified and documented in the
machine-readable asset licence registry.
