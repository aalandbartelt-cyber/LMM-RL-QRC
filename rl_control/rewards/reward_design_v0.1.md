# Reward Design v0.1

Initial reward terms:

| Term | Purpose |
|---|---|
| velocity tracking | Follow commanded velocity |
| yaw tracking | Follow commanded yaw rate |
| base height | Keep body height stable |
| roll/pitch stability | Reduce body attitude oscillation |
| energy cost | Reduce torque or action magnitude |
| action smoothness | Avoid jittery actions |
| fall penalty | Penalize unsafe states |
| collision penalty | Avoid obstacles |
| goal reward | Encourage waypoint or task completion |
