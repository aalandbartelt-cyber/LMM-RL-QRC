# Paper Reading List

This file is the team's curated paper map for the quadruped robot challenge. It focuses on papers that can directly shape our September screening prototype: simulation-first evidence, LLM decision, RL locomotion, Sim2Real, and two scenario applications.

## Reading Priority

| Priority | Paper | Link | Main Group | Why It Matters |
|---|---|---|---|---|
| P0 | DrEureka: Language Model Guided Sim-To-Real Transfer | https://arxiv.org/abs/2406.01967 | RL, Simulation | LLM-generated reward functions and domain randomization; strongest reference for making our simulation work look transferable to a real robot |
| P0 | Long-horizon Locomotion and Manipulation on a Quadrupedal Robot with Large Language Models | https://arxiv.org/abs/2404.05291 | LLM decision, RL | High-level LLM planning calls low-level RL skills; matches our "decision layer + skill library + controller" architecture |
| P0 | AINav: Large Language Model-Based Adaptive Interactive Navigation | https://arxiv.org/abs/2503.22942 | LLM decision, Simulation | Primitive skill tree, adaptive replanning, and obstacle interaction; useful for disaster-response and cluttered-environment scenes |
| P1 | SayTap: Language to Quadrupedal Locomotion | https://arxiv.org/abs/2306.07580 | LLM decision, RL | Uses foot contact patterns as the bridge between language and locomotion; lightweight enough to inspire a near-term interface |
| P1 | MoRE: Unlocking Scalability in Reinforcement Learning for Quadruped Vision-Language-Action Models | https://arxiv.org/abs/2503.08007 | LLM decision, RL | VLA + MoE + RL fine-tuning; suitable as an advanced architecture reference, not the first implementation target |

## Project Pages and Code

| Paper | Resource | Link | What to Check |
|---|---|---|---|
| DrEureka | Project page and code entry | https://eureka-research.github.io/dr-eureka/ | Reward examples, domain-randomization examples, safety instructions, sim-to-real workflow |
| Long-horizon robot | Project page | https://sites.google.com/view/long-horizon-robot | Skill API naming, multi-agent LLM decomposition, generated robot-code style, task examples |
| SayTap | Project page | https://saytap.github.io/ | Contact-pattern representation, prompts, and language-to-gait interface |

## How Each Group Should Use These Papers

### RL Group

Read first:

1. DrEureka: sections on reward generation, safety instruction, and domain randomization.
2. SayTap: contact-pattern interface and reward design.
3. Long-horizon robot: low-level RL skill library and how the high-level planner calls skills.
4. MoRE: only skim the architecture and experiment setting unless we later build VLA training.

Deliverables to extract:

- A first version of reward terms for flat walking, rough terrain, stairs/obstacles, stability, energy, and command tracking.
- A domain-randomization table: mass, friction, motor strength, latency, terrain height, sensor noise, push disturbance.
- A skill API list that the LLM group can call, such as `walk_to`, `turn_to`, `climb_step`, `recover`, `inspect_area`, `return_home`.
- A short "Sim2Real credibility" note explaining why our policy is not only a simulator demo.

### LLM Decision Group

Read first:

1. Long-horizon robot: semantic planner, parameter calculator, code generator, replanner.
2. AINav: primitive skill tree, advisor, arborist, adaptive replanning.
3. SayTap: language-to-contact-pattern interface as a possible lightweight gait-control bridge.
4. MoRE: VLA/MoE as advanced background for final presentation, not immediate implementation.

Deliverables to extract:

- A JSON task schema that converts natural-language commands into scenario, goal, constraints, and skill calls.
- A behavior-tree or state-machine template for patrol and disaster-response tasks.
- A safe failure protocol: unknown command, unreachable target, obstacle too risky, robot low battery, policy instability.
- Two complete demo commands that run from text instruction to simulated robot behavior.

### Simulation Group

Read first:

1. DrEureka: what simulator inputs are required for reward/DR design.
2. AINav: cluttered or blocked navigation scenes.
3. Long-horizon robot: long task scenes and task success metrics.

Deliverables to extract:

- Two scene definitions: park/community patrol and disaster-response reconnaissance.
- Terrain and obstacle parameter list that RL can randomize.
- Evaluation scripts for success rate, fall count, route completion, collision count, average speed, and recovery success.
- Short videos or screenshots showing the robot completing a scenario in simulation.

## Immediate Implementation Translation

The papers should become concrete repo work, not only reading notes:

| Paper Idea | Repo Target | Concrete Output |
|---|---|---|
| DrEureka reward generation | `rl_control/rewards/` | Add reward term checklist and first reward config |
| DrEureka domain randomization | `simulation/` and `rl_control/env/` | Add DR parameter table and training/evaluation ranges |
| Long-horizon skill API | `llm_decision/schemas/` and `llm_decision/behavior_tree/` | Add callable skill schema and example tasks |
| AINav primitive tree | `llm_decision/behavior_tree/` | Add patrol/rescue behavior tree with replanning states |
| SayTap contact patterns | `rl_control/policies/` | Define optional `contact_pattern` command field |
| MoRE VLA architecture | `docs/` | Use as advanced technical reference in proposal/presentation |

## Suggested One-Week Reading Split

| Day | RL Group | LLM Decision Group | Simulation Group |
|---|---|---|---|
| Day 1 | DrEureka abstract, method, reward/DR examples | Long-horizon abstract, system overview | AINav abstract, scenes, evaluation |
| Day 2 | DrEureka domain randomization table | Long-horizon skill API and prompts | DrEureka simulator assumptions |
| Day 3 | SayTap contact pattern and reward | AINav primitive tree and replanning | Long-horizon scene/task examples |
| Day 4 | Map reward/DR into our repo | Draft JSON task schema | Draft two simulation scene specs |
| Day 5 | Write RL skill API proposal | Write behavior-tree prototype | Write evaluation metric schema |
| Day 6 | Sync with LLM and Simulation groups | Sync with RL and Simulation groups | Sync with RL and LLM groups |
| Day 7 | Produce a short reading note and next action list | Produce a short reading note and next action list | Produce a short reading note and next action list |

## Notes for September Screening

- DrEureka helps us justify why "simulation-first" can still be credible: reward design and physical randomization are explicitly aimed at real-world transfer.
- Long-horizon robot and AINav support our three-layer architecture: LLM decision, behavior tree/safety supervisor, RL skill controller.
- SayTap gives us a simple fallback bridge between language and locomotion if full VLA is too heavy.
- MoRE should be framed as forward-looking inspiration unless we have enough data, compute, and time to test a minimal VLA pipeline.
