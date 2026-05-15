# PhysicsNeMo Projects and Examples

- This repository contains runnable PhysicsNeMo projects and examples. It does not need to be installed as a Python package.

- Instead, you install PhysicsNeMo itself in a separate environment, activate that environment, and then run the examples from this repository inside it.

- This readme includes how to install PhysicsNeMo.

## Usage

If PhysicsNeMo is already installed and synced in your environment, use this path:

```bash
cd <your-projects-dir>/physicsnemo
source .venv/bin/activate
cd <your-projects-dir>/physicsnemo-projects/legacy_sym_examples/wave_equation
python wave_equation_1d.py
```
If you are setting up for the first time, continue with the full steps below.

## Repository layout

- `legacy_sym_examples/`: archived and migrated Sym-based example projects ([legacy_sym_examples/README.md](./legacy_sym_examples/README.md))
- `projects/`: space for your own working projects built on top of the same PhysicsNeMo environment ([README.md](./README.md) for setup and usage instructions)

## Troubleshooting

**cuBLAS/CUDA context warning:**
The userwarning about cuBLAS and CUDA context is harmless. If you still want to avoid it, you can add this line at the top of your script (after imports):

```python
import torch; torch.cuda.init()
```


<img src="./Deltares_Deep_Learning.png" alt="Deltares logo" align="left" width="500" />

<br clear="all" />

# Installing PhysicsNeMo on Zbook with Windows 11

There are several ways to install physicsnemo. The general installation guide can be found here: https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html.

Below is a tried-n-tested install for the following (common) user setup:
- Zbook laptop with Windows 11 and an NVIDIA GPU
- You have installed WLS2 (Windows Subsystem for Linux) - [Install WSL](https://learn.microsoft.com/windows/wsl/install)

<br>

## Step 0: Verify that the GPU is visible in WSL2

Open WSL command promt (windows key, then type wsl)

Run:

```bash
nvidia-smi
```

If this works and shows your NVIDIA GPU, WSL2 can see the GPU correctly.

Why this matters:
On a ZBook, WSL2 uses the NVIDIA GPU rather than accidentally falling back to the integrated Intel graphics path that can cause confusion or GPU-related errors.

## Step 1: Install system prerequisites

Run:

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y build-essential git curl python3 python3-venv python3-pip
```

This installs the basic tools needed for cloning repositories and creating the Python environment.

## Step 2: Install the latest `uv` in WSL2

Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec bash
uv --version
```

If `uv --version` prints a version number, `uv` is installed correctly.

## Step 3: Clone PhysicsNeMo inside the WSL filesystem

Do **not** clone into a Windows-mounted path such as `/mnt/c/...`.
That works on first sight, but it is extremely slow and can cause file-system related issues.

Run:

```bash
cd <your-projects-dir>
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo
```

## Step 4: Install PhysicsNeMo and the required extras

For this setup, install PhysicsNeMo with these extras:

- `cu12`: CUDA 12 support (used here for NVIDIA RTX A1000, Ampere, compute capability 8.6)
- `nn-extras`: additional neural-network layers and tooling
- `sym`: symbolic PDEs, PINNs, geometry, and constraints
- `gnns`: graph and mesh support (adding this takes several minutes more to install)

**Important:** The CUDA version (`cu12` or `cu13`) depends on your GPU's compute capability. If you have a newer GPU (Ada, Hopper, etc.), you may need `cu13` instead of `cu12`. Check your GPU's compute capability here: https://developer.nvidia.com/cuda-gpus

Run:

```bash
uv sync --extra cu12 --extra nn-extras --extra sym --extra gnns
```
## Step 5: Verify PhysicsNeMo install

First activate the environment and check python version.

```bash
source .venv/bin/activate
python --version
```

Then verify if the sym imports work.

```bash
python - <<'EOF'
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
print("Navier-Stokes PDE loaded")
EOF
```

## Ready to go!

Everything is set up and working. Jump to **Usage** at the top and start running your own projects and examples in this repository.



