# judo
[![build](https://github.com/bdaiinstitute/judo/actions/workflows/build.yml/badge.svg)](https://github.com/bdaiinstitute/judo/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/bdaiinstitute/judo/graph/badge.svg?token=3GGYCZM2Y2)](https://codecov.io/gh/bdaiinstitute/judo)
[![docs](https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml/badge.svg)](https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml)
[![Static Badge](https://img.shields.io/badge/documentation-latest-8A2BE2)](https://bdaiinstitute.github.io/judo)

> Disclaimer: This code is released as a research prototype and is not
> production-quality software. Please note that the code may contain
> missing features and potential bugs. As part of this release, the
> RAI Institute does not offer maintenance or support for the software.
> While we may accept PRs implementing features and bugfixes, we cannot
> guarantee timely responses to issues.


## Judo: A Unified Framework for Agile Robot Control, Learning, and Data Generation via Sampling-Based MPC

![judo](docs/source/_static/images/judo.gif)

While sampling-based MPC has enjoyed success for both real-time control and as an underlying planner
for model-based RL, recent years have seen promising results for applying these controllers directly
in-the-loop as planners for difficult tasks such as [quadrupedal](https://lecar-lab.github.io/dial-mpc/)
and [bipedal](https://johnzhang3.github.io/mujoco_ilqr/) locomotion as well as
[dexterous manipulation](caltech-amber.github.io/drop/).

`judo` is a simple, `pip`-installable framework for designing and deploying new tasks and algorithms
for sampling-based controllers. It consists of three components:
1. A visualizer based on [`viser`](https://github.com/nerfstudio-project/viser) which enables
3D visualization of system states and GUI callbacks for real-time, interactive parameter tuning.
2. A multi-threaded C++ rollout of `mujoco` physics which can be used as the underlying engine
for sampling-based controllers.
3. `numpy`-based implementations of standard sampling-based MPC algorithms from literature and
canonical tasks (cartpole, acrobot) as `mujoco` tasks.

We also release a simple app that deploys simulated tasks and controllers for better study of how the
algorithms perform in closed-loop.

These utilities are being released with the hope that they will be of use to the broader community,
and will facilitate more research into and benchmarking of sampling-based MPC methods in coming years.

### Installation
Install judo
```
uv pip install -e .
```

The package contains custom C++ extensions for multi-threaded physics rollouts. These
should be compiled as part of the "typical" python installation, but you may need to
install `cmake` if it is not available on your system:
```
sudo apt install cmake
```

If you are not using `uv` you will need to install the extensions manually with
```
pip install -e ./src/mujoco_extensions
```

### Getting started
To run the application, simply run:
```
uv run viser-app
```
Then open the visualizer in your browser by clicking on the link in the terminal.
```
http://localhost:8008/
```

### Run tests locally
In the virtual environment
```
pip install -e .[dev]
python -m pytest
```
you might have to
```
unset PYTHONPATH
```

### Contributing

We welcome contributions to `judo`!

Before submitting a PR, please ensure:
1.  Your changes pass all pre-commit checks.
2.  All unit tests pass, including any new tests for your changes.

Thank you for helping improve this project!

### Limitations
The code as released has several limitations:
* The inter-process communication (IPC) is implemented in Python `multiprocessing`
    and cannot communicate with processes in other languages.
* Coverage of canonical gradient-free and first-order methods (e.g., GAs, iLQR)
    is low and could be improved.
* The system architecture must contain `Physics`, `Controller` and `Visualizer` objects,
    without exception, which is a somewhat limited API.
* `Controllers` must output splines even if (like RL controllers) they are instantaneous
    feedback policies.
