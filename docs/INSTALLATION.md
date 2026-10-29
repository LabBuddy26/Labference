# Installation

## Requirements

- Linux with a supported NVIDIA driver and GPU;
- NVIDIA Isaac Sim and its Python environment;
- Python dependencies used by LabBench (`hydra-core`, `omegaconf`, `numpy`,
  `opencv-python`, `h5py`, and the dashboard requirements);
- a separately installed, licence-approved LabBench asset package.

No model checkpoint or simulator asset is downloaded implicitly.

## Setup

```bash
git clone https://github.com/zhaochunhai/Labference
cd Labference
python -m pip install hydra-core omegaconf numpy opencv-python h5py
python -m pip install -r eai4s_eval/requirements-dashboard.txt
python -c "from isaacsim import SimulationApp; print('Isaac Sim OK')"
python -c "import hydra, omegaconf, h5py; print('Dependencies OK')"
```

Install the asset package at the paths documented by that package. Assets
marked `pending` or `NOASSERTION` are intentionally unavailable from the
public repository.

## Dashboard

```bash
python -m eai4s_eval.server.app \
  --eval-python /path/to/isaac/python \
  --outputs-root outputs/eval_runs \
  --host 127.0.0.1 --port 5000
```

Keep the service bound to localhost unless access control and a reverse proxy
have been configured.
