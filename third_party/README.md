# Third Party

This is the single reference hub for materials that are not native project code:

- official contest files
- local reference papers
- team-collected learning notes
- external repositories, papers, project pages, and videos

Internal design docs stay in `docs/`. External references and local reference files stay here.

## Local Files

| Type | File | Recommended Readers | Purpose |
|---|---|---|---|
| Contest | `contest/DG-202609比赛方案.pdf` | All members | Official problem statement and deliverable requirements |
| Notes | `learning/参考.md` | All members | Original collected links and notes |
| Paper | `learning/2604.08508v2.pdf` | Simulation, LLM decision, RL | Sumo whole-body loco-manipulation; useful for layered planning/control ideas |
| Paper | `learning/面向复杂地形的四足机器人RL-MPC分层运动控制方法研究_王留东.pdf` | RL group | RL-MPC hierarchical locomotion control, reward design, domain randomization, evaluation metrics |

## Extracted External Resources

| Priority | Name | Link | Type | Recommended Owner | How We Use It |
|---|---|---|---|---|---|
| Primary | MathFoundationRL | https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning | RL textbook | RL-F | Build RL vocabulary: state, action, reward, policy, actor-critic |
| Primary | Go2 RL Gym | https://github.com/wty-yy/go2_rl_gym | Training code | RL-G | Main candidate for Go2 locomotion baseline reproduction |
| Primary | RoboGauge | https://github.com/wty-yy/RoboGauge | Evaluation code | RL-H, Simulation-C | Borrow evaluation metrics and benchmark style |
| Primary | RoboGauge project page | https://robogauge.github.io/complete/ | Project page | RL-H | Understand demo tasks, metrics, and presentation style |
| Primary | MoE robust quadruped paper | https://arxiv.org/abs/2602.00678 | Paper | RL-F | Reference for robust Go2 locomotion and Sim2Real evaluation |
| Primary | Unitree C++ Deploy | https://github.com/wty-yy/unitree_cpp_deploy | Deployment code | RL-H, Deployment | Reference for ONNX/C++ real-robot deployment |
| Primary | RL-MPC paper DOI | https://doi.org/10.19886/j.cnki.dhdz.2025.0438 | Paper source | RL-F | Reference for safe hierarchical RL-MPC framing |
| Primary | Sumo project page | https://sumo.rai-inst.com/ | Project page | LLM-D, Simulation-A, RL-F | Reference for high-level planning steering low-level learned policy |
| Primary | DrEureka | https://arxiv.org/abs/2406.01967 | Paper | RL-F, Simulation-A | LLM-guided reward generation and domain randomization for Sim2Real |
| Primary | DrEureka project page | https://eureka-research.github.io/dr-eureka/ | Project page/code | RL-F, Simulation-A | Study open-source reward/DR examples and sim-to-real workflow |
| Primary | Long-horizon Locomotion and Manipulation with LLMs | https://arxiv.org/abs/2404.05291 | Paper | LLM-D, RL-G | Reference for LLM multi-agent planning over RL skill APIs |
| Primary | Long-horizon robot project page | https://sites.google.com/view/long-horizon-robot | Project page | LLM-D | Study skill API design, prompt decomposition, and executable robot code style |
| Primary | AINav: LLM-Based Adaptive Interactive Navigation | https://arxiv.org/abs/2503.22942 | Paper | LLM-D, Simulation-B | Reference for disaster/cluttered-scene navigation, primitive trees, and replanning |
| Candidate | MoRE quadruped VLA | https://arxiv.org/abs/2503.08007 | Paper | LLM-D, RL-F | Later reference for VLA/MoE architecture and RL fine-tuning |
| Candidate | SayTap | https://arxiv.org/abs/2306.07580 | Paper | LLM-D, RL-G | Lightweight language-to-gait idea using foot contact patterns as the interface |
| Candidate | SayTap project page | https://saytap.github.io/ | Project page | LLM-D, RL-G | Reference for contact-pattern prompt design and gait-command interface |
| Candidate | SWAP project page | https://swap-parkour.github.io/ | Project page | RL-F | Later-stage reference for agile locomotion/world-model ideas |
| Candidate | SWAP arXiv | https://arxiv.org/abs/2606.19928 | Paper | RL-F | Later-stage reading, not first-week implementation |
| Candidate | SWAP IEEE paper | https://ieeexplore.ieee.org/abstract/document/11495396 | Paper | RL-F | Optional academic reference |
| Candidate | SWAP YouTube | https://www.youtube.com/watch?v=LZwVcr40aUg | Video | All groups | Demo inspiration only |
| Candidate | AMP Go2 | https://github.com/ak1raljl/amp_go2 | Code | RL-G | Backup lightweight Go2 RL demo if main route is blocked |
| Candidate | My Unitree Go2 Gym | https://github.com/yusongmin1/My_unitree_go2_gym | Code | RL-G | Backup reference for trot/gait control |
| Later | LeggedManip Lab | https://github.com/zzzJie-Robot/LeggedManip_Lab | Training framework | RL-G, Simulation-A | Later reference for quadruped-arm whole-body control |
| Later | Go2Arm Lab | https://github.com/zzzJie-Robot/Go2Arm_Lab | Code | Simulation-B, RL-G | Later reference if we add arm manipulation |
| Later | Go2Arm Sim2Sim | https://github.com/zzzJie-Robot/Go2Arm_sim2sim | Code | Simulation-B, RL-G | Later reference for quadruped-arm sim2sim |
| Later | zzzJie-Robot GitHub | https://github.com/zzzJie-Robot | GitHub org/user | Simulation-B | Extra quadruped-arm references |

## Rules

- Do not copy full third-party repositories into this folder.
- Put links and reading notes in this README.
- If a repo must be used, clone it separately or discuss adding it as a submodule.
- Do not commit large model weights, datasets, or videos.
- Keep the priority label updated: `Primary`, `Candidate`, or `Later`.
