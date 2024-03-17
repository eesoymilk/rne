# HW1: Navigation with Known Map

## Path Tracking

### PID Control

PID (Proportional-Integral-Derivative) control is a popular control algorithm used in various applications, including robotics and autonomous vehicles. In path tracking, the PID controller adjusts the steering angle and/or velocity of the vehicle to minimize the deviation from a desired path.

#### Basic

In the basic kinematic model, the vehicle is controlled by adjusting its velocity (v) and angular velocity (ω). The PID controller calculates the error between the vehicle's current position and the target position on the path and uses this error to compute the steering command.

- Proportional (P): This term produces an output that is proportional to the current error. It helps the vehicle to steer towards the path quickly.
- Integral (I): This term accounts for the cumulative error over time. It helps to eliminate any steady-state error.
- Derivative (D): This term considers the rate of change of the error. It helps to dampen the oscillations and stabilize the response.

#### Differential Drive

In the differential drive model, the vehicle is controlled by adjusting the velocities of the left and right wheels. The PID controller is used to maintain the desired trajectory by controlling the angular velocity (ω) while maintaining a constant linear velocity (v).

- Proportional (P): Controls the angular velocity based on the deviation from the path.
- Integral (I): Compensates for any accumulated error in the system.
- Derivative (D): Provides damping to prevent overshooting and oscillations.

#### Bicycle

In the bicycle model, the vehicle is controlled by adjusting the acceleration (a) and the steering angle (δ). The PID controller is used to control the steering angle based on the error between the vehicle's current heading and the desired heading towards the target point on the path.

- Proportional (P): Adjusts the steering angle proportionally to the heading error.
- Integral (I): Reduces any persistent offset between the desired and actual headings.
- Derivative (D): Dampens the response to prevent oscillations around the desired heading.

### Pure Pursuit

Pure Pursuit is a path tracking algorithm that calculates the steering angle to follow a path based on a lookahead point. The lookahead distance dynamically changes with the vehicle's speed to ensure smooth and stable tracking.

#### Basic

In the basic kinematic model, the Pure Pursuit controller calculates the angular velocity (ω) to steer the vehicle towards a lookahead point on the path. The lookahead distance is adjusted based on the vehicle's speed to maintain stability.

- Lookahead Distance (Ld): The distance ahead of the vehicle where the target point on the path is chosen.
- Steering Command: The angular velocity is calculated to minimize the angle between the vehicle's heading and the direction to the lookahead point.

#### Differential Drive

In the differential drive model, the Pure Pursuit controller calculates the angular velocity (ω) while maintaining a constant linear velocity (v). The controller aims to minimize the angle between the vehicle's heading and the direction to the lookahead point.

- Lookahead Distance (Ld): Adjusted based on the vehicle's speed for stable tracking.
- Steering Command: The angular velocity is calculated to steer the vehicle towards the lookahead point.

#### Bicycle

In the bicycle model, the Pure Pursuit controller calculates the steering angle (δ) to follow the path. The steering angle is determined based on the geometry between the vehicle's position, its heading, and the lookahead point.

- Lookahead Distance (Ld): Dynamically adjusted with the vehicle's speed.
- Steering Command: The steering angle is calculated to align the vehicle's heading with the direction to the lookahead point.

### Stanley

The Stanley controller is another path tracking algorithm that combines both heading and cross-track error to compute the steering command. It is particularly effective in scenarios with sharp turns and dynamic changes in the path.

#### Bicycle

In the bicycle model, the Stanley controller adjusts the steering angle (δ) based on the heading error and the cross-track error.

- Heading Error: The difference between the vehicle's heading and the tangent angle of the path at the closest point.
- Cross-Track Error: The perpendicular distance from the vehicle to the path.
- Steering Command: The steering angle is calculated to minimize both the heading error and the cross-track error, ensuring that the vehicle follows the path accurately.

## Path Planning

### A\*

A* is a popular pathfinding algorithm used for grid-based maps. It combines features of Dijkstra's Algorithm (uniform cost search) and Greedy Best-First-Search, making it both complete and optimal.

- Heuristic Function (h): Estimates the cost to reach the goal from a node.
- Cost Function (g): The actual cost to reach a node from the start.
- F-Score: Sum of g and h, used to prioritize nodes in the open set.
- Open Set: A priority queue that holds nodes to be evaluated.
- Closed Set: Holds nodes that have already been evaluated.

The algorithm iteratively explores the most promising nodes (with the lowest F-score) until it reaches the goal. The path is reconstructed by tracing back from the goal node to the start node.

### RRT\*

Rapidly-exploring Random Trees Star (RRT*) is an improved version of the RRT algorithm that ensures asymptotic optimality. It builds a tree by randomly sampling points in the search space and connecting them to the nearest node in the tree, while also rewiring the tree to maintain the shortest path.

- Sampling: Randomly selects points in the search space.
- Nearest Node: Finds the nearest node in the tree to the sampled point.
- Steering: Moves from the nearest node towards the sampled point by a fixed distance (extend length).
- Collision Checking: Ensures that the path between nodes does not intersect with obstacles.
- Rewiring: Checks if the path to nearby nodes can be improved by going through the new node and updates the tree accordingly.

The algorithm continues to sample and extend the tree until it reaches the goal, and the path is reconstructed by tracing back from the goal node to the start node.

## Collision Handling

The collision handling logic in the navigation system is designed to respond to collisions between the simulated vehicle and obstacles in the environment. When a collision is detected, the vehicle backs up a short distance and then attempts to re-plan its path to the goal. This approach ensures that the vehicle can recover from collisions and continue navigating towards its destination.

- Collision Response

    Once a collision is detected, the vehicle responds by backing up a short distance to disengage from the obstacle. This is achieved by sending a reverse command to the simulator. The distance to back up can vary depending on the vehicle's kinematic model (e.g., basic, differential drive, bicycle).

- Recovery

    After backing up, the vehicle checks if it has moved a sufficient distance away from the point of contact. If so, it initiates re-path-planning to find a new path to the goal, avoiding the obstacle that caused the collision.

```python=
if collision_count > 0:
    if self.simulator_name in ("basic", "diff_drive"):
        backup_command = ControlState(self.simulator_name, -5, 0)  # Backward for a short distance
    elif self.simulator_name == "bicycle":
        backup_command = ControlState(self.simulator_name, -1, 0)  # Backward for a short distance
    else:
        raise NameError("Unknown simulator!!")

    self.simulator.step(backup_command)
    self._pose = (
        self.simulator.state.x,
        self.simulator.state.y,
        self.simulator.state.yaw,
    )

    if (
        np.hypot(
            self._pose[0] - contact_position[0],
            self._pose[1] - contact_position[1],
        )
        > 20
    ):
        self._set_path(self._goal)
        collision_count = 0
```


