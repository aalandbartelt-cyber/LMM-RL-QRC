# Safety Rules

Initial safety rules:

- Reject tasks with `max_speed` greater than the configured robot limit.
- Reject tasks with missing target points.
- Stop when obstacle distance is below threshold.
- Stop when robot fallen state is true.
- Stop when communication times out.
- Stop when episode timeout is reached.
