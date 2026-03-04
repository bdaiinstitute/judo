# `run_mpc/main.py`

Run batched MPC rollouts from a JSON config and save successful trajectories to an HDF5 dataset.

## Prerequisite

This script must be run in the `mjwarp` pixi environment:

```bash
pixi shell -e mjwarp
```

Run commands below from the repository root (`judo/`).

## Basic usage

```bash
python run_mpc/main.py --config-path run_mpc/configs/spot_tire_upright.json
```

By default, this writes output to:

- `run_mpc/configs/trajectories.h5`

## Common examples

Run 50 trajectories with 8 in parallel:

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/spot_tire_upright.json \
  --num-trajectories 50 \
  --num-parallel 8
```

Choose a custom output file:

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/spot_tire_upright.json \
  --dataset-output-path outputs/mpc/spot_tire_upright.h5
```

Enable visualization (no parallel, single env only):

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/spot_tire_upright.json \
  --num-parallel 1 \
  --visualize
```

## One-liner without entering shell

```bash
pixi run -e mjwarp python run_mpc/main.py --config-path run_mpc/configs/spot_tire_upright.json
```
