# Metrics Schema

Each episode result should include:

| Field | Description |
|---|---|
| episode_id | Unique episode id |
| scene | Scene name |
| policy_name | Policy version |
| success | Whether the task completed |
| completion_time | Episode time in seconds |
| fall_count | Number of falls |
| collision_count | Number of collisions |
| safety_stop_count | Number of safety stops |
| velocity_tracking_error | Mean velocity tracking error |
| roll_error | Roll error summary |
| pitch_error | Pitch error summary |
| path_error | Path deviation |
| failure_reason | Failure category |
