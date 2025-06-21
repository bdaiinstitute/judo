# Tasks
This README documents all of the tasks that are currently available in Judo.

| Task | Description |
| ---- | ----------- |
| [cylinder_push](#cylinder-push) | Push a cylinder to a target position with another actuated cylinder. |
| [cartpole](#cartpole) | Swing up a cartpole with actuator limits to a target angle. |
| [fr3_pick](#fr3-pick) | Pick up a cube with a Franka robot and place it in a target location. |
| [leap_cube](#leap-cube) | Rotate a cube to a target orientation using a LEAP hand. |
| [leap_cube_down](#leap-cube-down) | Rotate a cube to a target orientation using a LEAP hand, but with the palm facing down. |
| [caltech_leap_cube](#caltech-leap-cube) | Rotate a cube to a target orientation using a modified LEAP hand at Caltech. |

## Cylinder Push
The cylinder push task has the following weights:

| Weight | Description |
| ------ | ----------- |
| `w_pusher_proximity` | Penalizes the distance between the pusher and a point offset from the cart along the direction to the goal. |
| `w_pusher_velocity` | Penalizes the squared linear velocity of the pusher. |
| `w_cart_position` | Penalizes the distance from the cart to the goal. |

There are other tunable parameters:

| Parameter | Description |
| --------- | ----------- |
| `pusher_goal_offset` | The offset distance from the cart that defines the pusher's target. |
| `goal_pos` | The 2D target position for the cart. |

## Cartpole
The cartpole task has the following weights:

| Weight | Description |
| ------ | ----------- |
| `w_vertical` | Penalizes the distance between the pole's angle and its upright (vertical) position. |
| `w_centered` | Penalizes the distance of the cart from the origin. |
| `w_velocity` | Penalizes the squared linear and angular velocities of the cart and pole. |
| `w_control` | Penalizes the amount of control effort applied. |

---

There are other tunable parameters:

| Parameter | Description |
| --------- | ----------- |
| `p_vertical` | The threshold parameter for the smooth L1 norm used in the `w_vertical` penalty. |
| `p_centered` | The threshold parameter for the smooth L1 norm used in the `w_centered` penalty. |

## FR3 Pick
The FR3 Pick task involves a Franka Research 3 picking up an object and placing it at a goal location. The task is divided into four phases, the first three of which have reward weights, and also includes global reward weights applied throughout the task.

> ⚠️ **Disclaimer** ⚠️
>
> This task is currently configured to use up to 64 threads by default (depending on the optimizer)! For most users, this is too many, and will be very slow. You can adjust this in the GUI or update the default value.

### Reward Weights

#### Lift Phase Weights
These weights are active during the "LIFT" phase, when the object is being lifted from the table.

| Weight | Description |
| ------ | ----------- |
| `w_lift_close` | Penalizes the distance between the end-effector's grasp site and the object. Encourages the robot to grasp the object. |
| `w_lift_height` | Penalizes the squared difference between the object's current height and a predefined `pick_height`. Encourages lifting the object to a certain height. |

#### Move Phase Weights
These weights are active during the "MOVE" phase, when the object is being moved towards the goal position.

| Weight | Description |
| ------ | ----------- |
| `w_move_goal` | Penalizes the distance between the object's XY position and the target `goal_pos`. Encourages moving the object to the goal. |
| `w_move_close` | Penalizes the distance between the end-effector's grasp site and the object. Encourages maintaining a grasp on the object. |

#### Place Phase Weights
These weights are active during the "PLACE" phase, when the object is being placed on the table at the goal.

| Weight | Description |
| ------ | ----------- |
| `w_place_table` | Penalizes the distance between the object and the table. Encourages placing the object on the table. |
| `w_place_goal` | Penalizes the distance between the object's XY position and the target `goal_pos`. Encourages placing the object accurately at the goal. |

#### Home Phase Weights
Once placed, the system enters the "HOME" phase and homes the arm, but there is only one reward term in this phase, so no additional weights.

#### Global Weights
These weights are applied across all phases of the task.

| Weight | Description |
| ------ | ----------- |
| `w_upright` | Penalizes the deviation of the end-effector's Z-axis from a purely downward (upright) orientation. Encourages maintaining an upright end-effector. |
| `w_coll` | Rewards avoiding contact between the robot hand (fingers) and the table. |
| `w_qvel` | Penalizes the norm of the robot's joint velocities, with a decay over time. Encourages smoother, less erratic movements. |
| `w_open` | Penalizes the squared difference between the gripper position and an "open" position (0.04). Encourages the gripper to be open. |

### Other Tunable Parameters

| Parameter | Description |
| --------- | ----------- |
| `goal_pos` | The 2D (x, y) target position for the object to be placed. |
| `goal_radius` | The radius around `goal_pos` that defines the goal region for the object. |
| `pick_height` | The desired height the object should reach during the `LIFT` phase. |

## Leap Cube
The LEAP Cube Rotation task involves a robotic hand manipulating a cube to achieve a desired orientation while maintaining its position.

| Weight | Description |
| ------ | ----------- |
| `w_pos` | Penalizes the squared difference between the cube's current 3D position and a fixed `goal_pos`. Encourages the cube to stay at the target position. |
| `w_rot` | Penalizes the squared difference in orientation (using SO(3) distance) between the cube's current quaternion and a target `goal_quat`. Encourages the cube to achieve the desired rotation. |

## Leap Cube Down
This works the same way as the LEAP Cube task, but the palm is facing downwards.

> ⚠️ **Disclaimer** ⚠️
>
> This task is currently configured to use up to 64 threads by default (depending on the optimizer)! For most users, this is too many, and will be very slow. You can adjust this in the GUI or update the default value.

## Caltech Leap Cube
This works the same way as the LEAP Cube task, but uses a modified LEAP hand model from Caltech.

> ⚠️ **Disclaimer** ⚠️
>
> This task is currently configured to use up to 32 threads by default (depending on the optimizer)! For many users, this is too many, and will be very slow. You can adjust this in the GUI or update the default value.
