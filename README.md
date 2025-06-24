# Judo 🥋

<p align="center">
  <img src="docs/source/_static/images/banner.gif" alt="task dropdown" width="640">
</p>

[![Python](https://img.shields.io/badge/Python-3.10|3.11|3.12|3.13-blue?logo=python&logoColor=white)](https://github.com/bdaiinstitute/judo)
&nbsp;
[![Test Status](https://github.com/bdaiinstitute/judo/actions/workflows/test.yml/badge.svg)](https://github.com/bdaiinstitute/judo/actions/workflows/test.yml)
&nbsp;
[![Docs Status](https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml/badge.svg)](https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml)
&nbsp;
[![Coverage Status](https://codecov.io/gh/bdaiinstitute/judo/graph/badge.svg?token=3GGYCZM2Y2)](https://codecov.io/gh/bdaiinstitute/judo)


`judo` is a `python` package inspired by [`mujoco_mpc`](https://github.com/google-deepmind/mujoco_mpc) that makes sampling-based MPC easy. Features include:
- 👩‍💻 A simple interface for defining custom tasks and controllers.
- 🤖 Automatic parsing of configs into a browser-based GUI, allowing real-time parameter tuning.
- 📬 Asynchronous interprocess communication using [`dora`](https://dora-rs.ai/) for easy integration with your hardware.
- 🗂️ Configuration management with [`hydra`](https://hydra.cc/docs/intro/) for maximum flexibility.

> ⚠️ **Disclaimer** ⚠️
>
> This code is released as a **research prototype** and is *not* production-quality software. It may contain missing features and potential bugs. The RAI Institute does **not** guarantee maintenance or support for this software. While we encourage contributions and may accept pull requests for new features or bugfixes, we **cannot guarantee** timely responses to issues.
>
> The current release is also in **alpha**. We reserve the right to make breaking changes to the API and configuration system in future releases. We will try to minimize these changes, but please be aware that they may occur.

# Quickstart
This section walks you through the installation and usage of `judo`. For more details, see [the docs](https://bdaiinstitute.github.io/judo).

## 1. Installation
### Using `pip`
We recommend installing `judo` using `pip` as follows:
```bash
pip install judo-rai  # if you want dev dependencies, use judo-rai[dev]
```

### Developers
#### Conda
For developers, run the following commands from this folder after cloning:
```bash
conda create -n judo python=3.13
conda activate judo
pip install -e .[dev]  # includes docs dependencies
pre-commit install
pybind11-stubgen mujoco -o typings/  # stops type checkers from complaining
```

#### Pixi
You can also use [`pixi`](https://pixi.sh/dev/) instead of `conda`, which has the added benefit of having an associated lock file that ensures complete reproducibility.

To install `pixi`, run the following:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
To create our environment (and activate it each time later), run the following in the repo root:
```bash
# every time you want to activate
pixi shell -e dev

# first time only
pre-commit install
pybind11-stubgen mujoco -o typings/
```

## 2. Run the `judo` app!
To start the simulator, you can simply run:
```bash
judo
```
This will start the stack and print a link in the terminal that will open the app in your browser, e.g.,
```
http://localhost:8080
```

We package `judo` with a few starter tasks and optimizers. If you want to start the simulator with one of these, you can run:
```bash
judo task=<task_name> optimizer=<optimizer_name>
```
where `task_name` is one of the following:
```
cylinder_push
cartpole
fr3_pick
leap_cube
leap_cube_down
caltech_leap_cube
```
and `optimizer_name` is one of the following:
```
cem
mppi
ps
```
This is not necessary, though, because you can use the dropdown menus to switch between tasks and optimizers after launching the app.
<p align="center">
  <img src="docs/source/_static/images/task_dropdown.png" alt="task dropdown" width="300">
   <img src="docs/source/_static/images/optimizer_dropdown.png" alt="optimizer dropdown" width="300">
</p>

You can also run the app programmatically from some other script or program.
```python
from judo.cli import app

if __name__ == "__main__":
    # do whatever you want here, like registering tasks/optimizers/overrides, etc.
    app()  # this runs the app from your own script
```

To see more information about the available tasks, please refer to [the task README](judo/tasks/README.md).

## 3. Running `judo` as a Dependency
You can easily install `judo` as a dependency in your own project. A few comments:
* You can still use the `judo` CLI command from anywhere, so long as you are working in an environment where `judo` is installed.
* If you do this, you should use the `hydra` configuration system to do things like registering custom tasks and optimizers, modifying the `dora` nodes in the sim stack, etc. See the [Configuration with `hydra`](#configuration-with-hydra) and [Config Registration](#config-registration) sections for more details.
* You can also run the app programmatically, as shown above.

## 4. Benchmarking
To benchmark all registered tasks and optimizers, simply run
```bash
benchmark
```
This will loop through all task/optimizer pairs and check the planning time over 100 samples. The end result will be printed to the console, showing useful statistics on your system.

Note that the benchmarking program runs the default task and optimizer parameters (subject to default task-specific overrides). If you want to benchmark with different settings, please read the information below, which explains how to change defaults.

# Docs
For developers, to build docs locally, run the following in your environment from the repo root. Note that asset paths will be broken locally that work correctly on Github Pages.
```bash
# using conda
pip install -e .[docs]  # dev also includes docs

# using pixi
pixi shell -e docs  # dev also includes docs

# building the docs (both conda and pixi)
sphinx-build docs/source docs/build -b dirhtml
python -m http.server --directory docs/build 8000
```

# 🤝 Contributing
We welcome contributions! See our [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started.


# Citation
If you use `judo` in your research, please use the following citation for our [paper](https://arxiv.org/abs/2506.17184):
```
@inproceedings{li2025_judo,
  title     = {Judo: A User-Friendly Open-Source Package for Sampling-Based Model Predictive Control},
  author    = {Albert H. Li and Brandon Hung and Aaron D. Ames and Jiuguang Wang and Simon Le Cleac'h and Preston Culbertson},
  booktitle = {Proceedings of the Workshop on Fast Motion Planning and Control in the Era of Parallelism at Robotics: Science and Systems (RSS)},
  year      = {2025},
  url       = {https://github.com/bdaiinstitute/judo},
}
```
