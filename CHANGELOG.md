# v0.0.6

## Added
* Separation between task-space controls and MuJoCo simulation controls
  * Added `task_to_sim_ctrl()` method to `Task` base class (default identity mapping)
  * Updated `RolloutBackend` to accept and use `task_to_sim_ctrl` function for control mapping
  * Updated `MJSimulation` to use `task_to_sim_ctrl` when setting controls
  * Tasks can now override `task_to_sim_ctrl()` to map between different control spaces

# v0.0.5

## Added
* Separation between middleware implementation and core functionality
* Support for different middleware implementations
* Sliders with enforced bounds to errors caused by unintended values
* Interface to create controllers from registered tasks and optimizers
* Abstract interface to implement different simulation backends
* Removed duplication of objects inside a node to ensure that all references to data point to the same object

## Fixed
* Fixed bug that would allow certain sliders to be set to arbitrary values that could crash the app
* Fixed bug that would allow NaN as a valid entry into the slider text box
* Fixed bug that caused the ControllerConfig to not be recreated when the `TaskConfig` was reset.

## Documentation
* Added specification between different slider types.
* Added guidelines on how to use the `Controller`. `Visualizer`, and `Simulation` objects

## Dev
* Modified release process so that any CODEOWNER can publish to pypi

# v0.0.4

## Added
* Added support for arm-based Macs (@johnzhang3, #87)
* Included functions for simpler model indexing for sensors and joints (@bhung-bdai, @alberthli, #76)

## Fixed
* Fixed bug after spec changes where unnamed geoms were causing `judo` to crash (@alberthli, @pculbertson, #72)

## Dev
* Bump version to 0.0.4 in pyproject.toml (@alberthli, #73)
* Cache meshes in CI to prevent CI failures when request limit is hit (@alberthli, #79)

# v0.0.3

## Added
* Action normalization (@yunhaif, #54)
    * Added three types of action normalizer: `none` (default, doing nothing), `min_max` (normalizing with actuator control range), and `running` (tracking running mean and std to normalize actions)
    * Added tests for the normalizers and their integration with the controller.
    * Users can change the action normalizer under the controller tab in the GUI.

## Documentation
* Updated the README with the arxiv link for the paper (@alberthli, #56)
* Created `docs` dependency group for even easier one-command installation of docs deps for both conda and pixi (@alberthli, #57)

## Dev
* Made `pixi-version` use `latest` instead of pinned version in all workflows (@alberthli, #59)
* Bumped Viser to 1.0.0 (@brentyi, #66)

# v0.0.2
This release contains bugfixes prior to RSS 2025.

## Added
* New `judo`-specific branding! (@slecleach, #42)

## Fixed
* Brandon's last name misspelling in citation in README.md (@pculbertson, #15)
* Fixed `max_opt_iters` not correctly being applied (@lujieyang and @tzhao-bdai, #14)
    * Added a test to check for this case (@alberthli, #16)
* Fix bug where if no `exclude_geom_substring` is passed to the model visualizer, all geoms are accidentally excluded (@alberthli, #17)
* Fix bug where when `max_opt_iters>1`, the shape of `nominal_knots` is not correct. **NEW BEHAVIOR:** in the controller, `self.nominal_knots` now has shape `(num_knots, nu)` instead of `(1, num_knots, nu)` (@yunhaif, #29).
* Update model loading so that textures appear correctly in the visualizer (@pculbertson, #32)
* Fix bug where leap cube task encountered division by 0 in axis normalization (@alberthli, #34)
* Fixed bug where changing tasks added accumulating grey lines to the GUI (@slecleach, #44)
* Fixed bug in FR3 task where the MJC distance sensors were flakily not reporting when cube was being lifted (@alberthli, #49).

## Documentation
* Changelog file to track changes in the repository (@alberthli, #43)
* Added contributor guidelines to the README (@alberthli, #43)
* Added information about the tasks in a task README (@alberthli, #43)
* Updated author order in the README citation (@pculbertson, #50)

## Dev
* Bump prefix-dev/setup-pixi from 0.8.8 to 0.8.10 (#18)
* Create workflow for manually publishing releases to PyPi (@alberthli, #33)
* Update a bunch of versions for pixi (@alberthli, #40)

# v0.0.1
Initial release!
