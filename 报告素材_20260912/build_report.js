// Build the DG-202609 research report docx (A4, figure-rich, evidence-labeled).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak, VerticalAlign,
} = require("docx");

const M = "D:/上海大学__/26挑战杯擂台赛/报告素材_20260912";
const OUT = "D:/上海大学__/26挑战杯擂台赛/智巡四足_研究报告_20260912.docx";
const img = (p) => fs.readFileSync(path.join(M, p));

const BODY = "SimSun";
const HEI = "SimHei";
const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function T(text, opts = {}) {
  return new TextRun({ text, font: opts.font || BODY, size: opts.size || 24, bold: !!opts.bold, italics: !!opts.italics, color: opts.color });
}
function P(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align,
    spacing: { before: opts.before ?? 60, after: opts.after ?? 60, line: opts.line ?? 312 },
    indent: opts.indentFirst ? { firstLine: 480 } : undefined,
    children: [T(text, opts)],
  });
}
function H(level, text) {
  return new Paragraph({ heading: level, spacing: { before: 300, after: 160 }, children: [new TextRun({ text, font: HEI, bold: true, size: level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 28 : 24 })] });
}
function caption(kind, num, text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 200 }, children: [new TextRun({ text: `${kind} ${num}　${text}`, font: HEI, size: 21, bold: true })] });
}
function figure(num, file, w, h, cap) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 }, children: [new ImageRun({ type: "png", data: img(file), transformation: { width: w, height: h }, altText: { title: cap, description: cap, name: `fig${num}` } })] }),
    caption("图", num, cap),
  ];
}
function cell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: opts.head ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: [new TextRun({ text: String(text), font: BODY, size: opts.size || 21, bold: !!opts.head })] })],
  });
}
function table(num, cap, widths, rows, fontSize = 21) {
  const total = widths.reduce((a, b) => a + b, 0);
  return [
    caption("表", num, cap),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: rows.map((r, i) => new TableRow({ children: r.map((c, j) => cell(c, widths[j], { head: i === 0, size: fontSize, align: i === 0 ? AlignmentType.CENTER : undefined })) })),
    }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}
function imgCell(file, w, h, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 40, right: 40 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: img(file), transformation: { width: w, height: h }, altText: { title: file, description: file, name: file } })] })],
  });
}
function imageGrid(files, w, h, cellWidth) {
  const rows = [];
  for (let i = 0; i < files.length; i += 3) {
    const trio = files.slice(i, i + 3);
    rows.push(new TableRow({ children: trio.map((f) => imgCell(f, w, h, cellWidth)) }));
  }
  return new Table({ width: { size: cellWidth * 3, type: WidthType.DXA }, columnWidths: [cellWidth, cellWidth, cellWidth], rows });
}

const routeResults = JSON.parse(fs.readFileSync(path.join(M, "04_数据与证据/route_results.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(path.join(M, "04_数据与证据/manifest.json"), "utf8"));

const children = [];

// ---------- 封面 ----------
children.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480, after: 120 }, children: [new TextRun({ text: "第十五届“挑战杯”竞赛 2026 年度中国青年科技创新", font: HEI, size: 28, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "“揭榜挂帅”擂台赛（DG-202609 榜题）", font: HEI, size: 28, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 240 }, children: [new TextRun({ text: "研  究  报  告", font: HEI, size: 56, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 }, children: [new TextRun({ text: "智巡四足", font: HEI, size: 44, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 }, children: [new TextRun({ text: "——基于 AI 大模型与强化学习的四足机器人智能控制系统", font: HEI, size: 26 })] }),
);
const coverRows = [
  ["作品名称", "智巡四足：AI 大模型与强化学习驱动的四足机器人智能控制系统"],
  ["榜题编号", "DG-202609（合肥华驱动力科技有限公司：AI 大模型与强化学习驱动的四足机器人智能控制系统）"],
  ["参赛赛道", "“揭榜挂帅”擂台赛 · 学生赛道"],
  ["作品类型", "智能系统 / 仿真验证 + 可复现实验包"],
  ["参赛学生", "[待填写：姓名、专业、年级]"],
  ["学校", "[待填写]"],
  ["指导教师", "[待填写：姓名、职称、研究方向]"],
  ["完成时间", "2026 年 9 月"],
];
children.push(
  new Table({
    width: { size: 9026, type: WidthType.DXA }, columnWidths: [2200, 6826],
    rows: coverRows.map(([k, v]) => new TableRow({ children: [cell(k, 2200, { head: true, align: AlignmentType.CENTER }), cell(v, 6826)] })),
  }),
  P("证据与用途声明：本报告所呈现的双场景演示与物理仿真画面均为本地 MuJoCo 高保真仿真，由正式评测第一名策略驱动，不是真机实验；全部量化成绩来自上海大学自强5000计算平台 63 组正式评测快照；全部素材经 SHA-256 哈希校验并可复核（附录 A）。", { before: 360, size: 20, italics: true }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------- 摘要 / Abstract / 目录 ----------
children.push(
  H(HeadingLevel.HEADING_1, "摘  要"),
  P("针对园区安防巡检与灾害应急侦察场景下四足机器人“任务理解—安全执行—鲁棒运动”一体化的需求，本报告设计并实现了“智巡四足”四足机器人智能控制系统。系统采用“大模型约束式任务规划—行为决策与安全监督—强化学习运动控制”三层架构：大模型仅将自然语言任务解析为可校验的结构化任务 JSON，经 JSON Schema、白名单与参数边界校验后交由行为树执行；安全监督器独立于大模型，具备限速、禁行区、姿态约束与急停的最高优先级仲裁能力；运动控制层采用 PPO 强化学习策略，以 45 维本体感知观测映射 12 维关节目标，经 PD 控制器驱动 Go2 四足机器人。", { indentFirst: true }),
  P("研究过程遵循“云端训练、本地验证、证据可溯”的技术路线：在上海大学自强5000平台完成三个候选策略的训练与 63 组正式评测（每策略 7 类地形），冻结综合排名第一的 baseline5001 策略（平均成功率 96.9%、总摔倒 177 次、平均碰撞率 0.065，均为三策略最优）；策略权重经 SHA-256 哈希交接至本地，数值一致性核验误差小于 5×10⁻⁶；随后在本地 MuJoCo 中以真实策略闭环复跑双场景任务路线——园区安全巡检 19 航点全路线 18/18 检查点、71.7 s 完成；灾害侦察与物资投送 11 航点全路线 10/10 检查点、117.3 s 完成，全程最低质心高度 0.297 m 与 0.311 m，远高于 0.18 m 安全阈值。全部演示视频、物理仿真视频、图表与数据均固化哈希并可逐件复核。本报告所有仿真画面均明确标注为仿真证据，不作为真机实验结论。", { indentFirst: true }),
  P("关键词：四足机器人；强化学习；AI 大模型任务规划；安全监督；MuJoCo 仿真；证据可追溯", { bold: true, before: 200 }),
  H(HeadingLevel.HEADING_1, "Abstract"),
  P("To meet the integrated requirement of task understanding, safe execution and robust locomotion for quadruped robots in campus patrol and disaster response scenarios, this report presents ZhiXun-Quadruped, an intelligent quadruped robot control system driven by an AI large language model and reinforcement learning. The system adopts a three-layer architecture: constrained LLM task planning, behavior execution with safety supervision, and RL locomotion control. The LLM only translates natural-language missions into a validated structured task JSON, which is checked against JSON Schema, whitelists and parameter bounds before execution; an independent safety supervisor holds the highest arbitration priority over speed limits, forbidden zones, posture constraints and emergency stop. The locomotion layer is a PPO policy mapping 45-dimensional proprioceptive observations to 12 joint targets through a PD controller on a Unitree Go2.", { indentFirst: true }),
  P("Following a cloud-training plus local-verification pipeline with fully traceable evidence, three candidate policies were trained and formally evaluated in 63 runs on the Ziqiang-5000 platform. The top-ranked policy baseline5001 was frozen (mean success rate 96.9%, 177 falls, mean collision rate 0.065, best among the three). Its weights were handed over with SHA-256 verification and cross-checked numerically with an error below 5e-6. The verified policy then completed both mission routes in local MuJoCo with closed-loop dynamics: 18/18 checkpoints in 71.7 s for campus security patrol and 10/10 checkpoints in 117.3 s for disaster reconnaissance and delivery, with minimum base heights of 0.297 m and 0.311 m, well above the 0.18 m safety threshold. All videos, figures and datasets are hash-registered for review. Every simulation result is explicitly labeled as simulation evidence rather than a real-robot experiment.", { indentFirst: true }),
  P("Key words: quadruped robot; reinforcement learning; LLM task planning; safety supervision; MuJoCo simulation; evidence traceability", { bold: true, before: 200 }),
  new Paragraph({ children: [new PageBreak()] }),
  new TableOfContents("目  录", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------- 1 研究背景 ----------
children.push(
  H(HeadingLevel.HEADING_1, "1 研究背景与问题提出"),
  H(HeadingLevel.HEADING_2, "1.1 产业需求与研究意义"),
  P("园区、变电站、化工区等场景的安防巡检长期依赖人工值守，存在值守成本高、夜间与恶劣天气覆盖不足、进入危险区域存在人身风险等问题；地震、火灾等灾害现场则要求侦察力量在“黄金时间”内进入结构不稳定区域完成目标搜索与物资投送，人员直接进入的代价和风险都很高。四足机器人凭借对台阶、坡道、碎石等非结构化地形的天然适应能力，成为上述两类任务的理想载体，但“听得懂任务、走得稳地形、控得住安全”的一体化系统仍是产业落地的关键瓶颈。", { indentFirst: true }),
  P("2026 年以来，具身智能与强化学习运动控制的成熟使四足机器人从“遥控平台”走向“自主作业系统”。本项目以 DG-202609 榜题为牵引，研究 AI 大模型任务理解与强化学习运动控制的工程融合：让机器人理解自然语言任务、在受约束的安全框架内自主执行，并以可复核的仿真证据体系支撑技术筛选。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "1.2 榜题要求与研究任务"),
  P("DG-202609 榜题“AI 大模型与强化学习驱动的四足机器人智能控制系统”要求参赛作品融合大模型与强化学习两项技术，完成四足机器人在典型作业场景中的智能控制闭环。团队将榜题要求分解为表 1 所示五项研究任务。"),
  ...table(1, "榜题要求与本项目研究任务对应关系", [4200, 4826],
    [["榜题要求", "本项目研究任务"],
     ["AI 大模型驱动的任务理解", "约束式任务规划：自然语言→结构化任务 JSON→Schema 校验→行为树执行（第 5 章）"],
     ["强化学习运动控制", "PPO + 域随机化 + 课程学习训练 12 关节运控策略（第 4 章）"],
     ["典型作业场景闭环", "园区安全巡检与灾害侦察投送双场景任务级仿真闭环（第 6 章）"],
     ["系统安全性", "独立安全监督器：限速/禁行区/姿态约束/急停，优先级高于大模型（第 3、5 章）"],
     ["成果可验证性", "63 组正式评测 + 本地复跑 + 全素材 SHA-256 哈希可追溯（第 6、7 章与附录）"]]),
  H(HeadingLevel.HEADING_2, "1.3 研究目标、指标与边界"),
  P("围绕初筛阶段的证据要求，项目设定表 2 所示可验证指标。所有指标均以“可复核数据”为达成依据，不以演示观感替代量化证据。"),
  ...table(2, "研究目标与可验证指标（初筛阶段）", [3000, 2800, 3226],
    [["指标", "目标值", "当前达成与证据"],
     ["运动策略正式评测平均成功率", "≥ 90%", "96.9%（63 组评测快照，表 4）"],
     ["双场景任务路线完成度", "全检查点通过", "园区 18/18、灾害 10/10（表 6）"],
     ["任务过程安全性", "无摔倒，最低质心 ≥ 0.18 m", "0.297 m / 0.311 m（表 6）"],
     ["策略部署一致性", "权重哈希一致，数值误差可忽略", "最大误差 4.8×10⁻⁶（4.5 节）"],
     ["证据可追溯性", "全部素材可哈希复核", "manifest 逐项 SHA-256（附录 A）"],
     ["大模型任务规划链路", "可解释、可校验", "任务 JSON + 规则校验流程（图 7）"]]),
  P("研究边界：本阶段成果为仿真级验证，不包含真机实验结论；报告中全部仿真画面均标注“本地任务级仿真/物理仿真”，量化成绩以自强5000平台正式评测快照为准。", { indentFirst: true }),
);

// ---------- 2 查新 ----------
children.push(
  H(HeadingLevel.HEADING_1, "2 国内外研究现状与技术查新"),
  H(HeadingLevel.HEADING_2, "2.1 四足机器人强化学习运动控制"),
  P("四足机器人运动控制经历了从模型预测控制到学习型控制的演进。MIT Cheetah 系列以凸优化 MPC 实现了高动态运动；苏黎世联邦理工 ANYmal 率先将深度强化学习部署到真实四足平台，证明了“仿真训练—实机部署”的可行性；以 PPO 为代表的策略梯度方法配合域随机化与课程学习，已成为宇树 Go2 等消费级四足平台鲁棒运控的主流路线。RSL-RL 生态进一步将该路线工程化，支持大规模并行训练与多地形课程。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "2.2 大模型驱动的机器人任务规划"),
  P("大语言模型用于机器人高层规划的代表性工作包括 SayCan（价值函数约束的可行性 grounding）、Code as Policies（策略即代码）以及大模型驱动行为树生成等。这些工作共同提示：大模型擅长语义解析与任务重组，但不应直接输出底层控制量；工程系统需要在大模型与执行层之间加入显式的校验与仲裁机制，否则无法保证行为安全性与可解释性。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "2.3 仿真评测体系与 Sim2Sim 验证"),
  P("Isaac Gym/Lab 提供大规模并行策略训练能力，MuJoCo 则以准确的接触动力学广泛用于交叉验证。RoboGauge 等评测框架主张以多地形、多扰动、多种子的批量指标替代单次演示，避免“挑片段”的证据偏差。Sim2Sim（跨仿真器复核）是进入真机前成本最低的可信度检验手段。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "2.4 技术查新结论"),
  ...table(3, "查新对比与本项目位置", [2200, 3413, 3413],
    [["对比维度", "现有代表性方案", "本项目位置"],
     ["运控算法", "MPC 或单一 RL 策略", "PPO + 域随机化 + 课程学习，三策略对比选型（第 4 章）"],
     ["大模型用法", "直接生成动作或代码", "只做受约束任务解析，输出可校验 JSON，不触碰底层控制（第 5 章）"],
     ["安全机制", "依赖人工监控或单一急停", "独立安全监督器 + 状态机优先级仲裁（3.3 节）"],
     ["验证方式", "单次演示或单一仿真器", "63 组正式评测 + 跨机复跑 + 全素材哈希可追溯（第 6、7 章）"]]),
);

// ---------- 3 总体方案 ----------
children.push(
  H(HeadingLevel.HEADING_1, "3 总体方案与技术路线"),
  H(HeadingLevel.HEADING_2, "3.1 需求分析与设计原则"),
  P("系统面向“非专业人员可下达任务、机器人可安全自主执行”的使用情境，确立四条设计原则：①安全优先——任何智能模块不得绕过安全监督；②分层解耦——任务理解、行为决策、运动控制各司其职，接口显式；③证据可复现——所有成绩可回溯到数据与命令；④低算力部署——运动策略仅以本体感知为输入，CPU 即可 50 Hz 推理，降低实机迁移门槛。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "3.2 系统分层架构"),
  P("系统总体架构如图 1 所示。自然语言任务经大模型解析为结构化任务 JSON；安全监督器对任务进行限速、禁行区与姿态约束校验；路线控制器将航点序列转换为机体速度指令；RL 运动策略以 50 Hz 输出 12 维关节目标，经 PD 控制器驱动 Go2；云端训练权重经哈希交接后在本地 MuJoCo 复跑形成报告证据。"),
  ...figure(1, "03_报告图片/图表/07_system_architecture.png", 600, 359, "智巡四足系统总体架构图（大模型—安全监督—路线控制—RL 运控四层）"),
  H(HeadingLevel.HEADING_2, "3.3 任务执行状态机与安全优先级"),
  P("双场景任务执行统一由状态机驱动（图 2）：任务解析→安全检查→巡逻/侦察执行→（异常告警→告警处置→局部重规划）→安全返航→任务完成。安全优先级从高到低为：人工接管/急停＞姿态异常保护＞安全监督约束＞任务执行指令；任何状态下急停或姿态越限立即进入安全停车，故障排除后方可复位。场景一的“发现入侵→告警上报→继续巡检”与场景二的“通道受阻→局部重规划→绕行返航”均为该状态机的实例化。"),
  ...figure(2, "03_报告图片/图表/d3_safety_state_machine.png", 600, 357, "双场景任务执行状态机与安全优先级"),
);

// ---------- 4 RL ----------
children.push(
  H(HeadingLevel.HEADING_1, "4 强化学习运动控制系统"),
  H(HeadingLevel.HEADING_2, "4.1 策略网络结构与控制链路"),
  P("运动控制层采用“策略网络 + PD 低层”的经典四足控制结构（图 3）。策略输入为 45 维本体感知观测：本体角速度（3 维）、投影重力（3 维）、速度指令（3 维）、关节位置偏移（12 维）、关节速度（12 维）与上一步动作（12 维）；网络为四层全连接 MLP（45→512→256→128→12，ELU 激活）；输出为 12 维关节位置目标偏移，经 0.25 缩放叠加默认站姿后，由 PD 控制器（Kp=20，Kd=0.5）输出关节力矩。策略不依赖视觉，满足赛题对低算力部署的要求。"),
  ...figure(3, "03_报告图片/图表/d1_rl_policy_network.png", 600, 357, "强化学习运动策略网络结构与控制链路（45 维观测→MLP→12 维关节目标→PD）"),
  H(HeadingLevel.HEADING_2, "4.2 训练方案：PPO、域随机化与课程学习"),
  P("训练基于 RSL-RL 框架在 Isaac Gym 大规模并行环境中进行，算法为 PPO。训练过程加入摩擦系数、质量、质心、关节参数、传感器噪声与外部推力等域随机化，并采用课程学习按“平地→坡道→台阶/障碍”分阶段提升地形难度。候选策略共三个：baseline5001（第三阶段追加 1500 次迭代的种子 1）与两个课程学习变体 curriculum_seed1_5601、curriculum_seed2_5601，形成可控对比。"),
  H(HeadingLevel.HEADING_2, "4.3 自强5000平台训练过程"),
  P("训练与评测全程在上海大学自强5000计算平台完成。图 4 为 baseline5001 第三阶段（第 3500–5001 次迭代）的训练曲线：平均回合奖励在阶段切换后快速回升并稳定在 22–23 区间，平均回合长度稳定于约 1250 步的满格水平，表明策略在该阶段已充分收敛；继续训练的边际收益低，且存在对早期地形课程遗忘的风险，故冻结第 5001 次迭代权重作为交付策略。"),
  ...figure(4, "03_报告图片/图表/06_training_curves.png", 600, 338, "baseline5001 第三阶段训练曲线（自强5000平台 TensorBoard 日志）"),
  P("图 5 为自强5000平台正式评测与结果归档记录，对应 63 组评测任务的执行与结果落盘过程。"),
  ...figure(5, "03_报告图片/平台截图/上海大学自强5000平台训练截图.PNG", 600, 298, "自强5000平台正式评测与结果归档记录"),
);

// ---------- 4.4 评测 ----------
children.push(
  H(HeadingLevel.HEADING_2, "4.4 三策略 63 组正式评测"),
  P("三个候选策略在 7 类地形（平地、斜坡、碎石坡、上台阶、下台阶、离散障碍、波浪地形）上各完成 21 组正式评测，合计 63 组。总体指标见表 4，baseline5001 分地形结果见表 5，六指标对比如图 6。"),
  ...table(4, "三策略 63 组正式评测总体指标（自强5000平台，2026-09-11 快照）",
    [700, 1550, 1150, 1000, 1150, 1350, 1100, 1026],
    [["排名", "策略", "平均成功率", "总摔倒", "平均碰撞率", "跟踪RMSE(m/s)", "平均功率(W)", "平均倾角(rad)"],
     ["1", "baseline5001", "96.9%", "177", "0.065", "0.326", "24.8", "0.196"],
     ["2", "curriculum_seed1_5601", "95.9%", "238", "0.075", "0.232", "31.2", "0.169"],
     ["3", "curriculum_seed2_5601", "93.6%", "381", "0.070", "0.218", "34.5", "0.182"]], 20),
  ...table(5, "baseline5001 分地形正式评测结果", [1600, 1300, 1100, 1250, 1400, 1200, 1176],
    [["地形", "成功率", "摔倒次数", "碰撞率", "RMSE(m/s)", "功率(W)", "倾角(rad)"],
     ["平地 flat", "100.0%", "0", "0.000", "0.113", "23.4", "0.032"],
     ["斜坡 slope", "100.0%", "0", "0.000", "0.240", "26.7", "0.182"],
     ["碎石坡 rough_slope", "100.0%", "0", "0.000", "0.268", "27.3", "0.180"],
     ["上台阶 stairs_up", "95.8%", "32", "0.238", "0.459", "26.6", "0.182"],
     ["下台阶 stairs_down", "92.1%", "67", "0.103", "0.445", "20.1", "0.349"],
     ["离散障碍 obstacles", "91.9%", "67", "0.109", "0.373", "22.3", "0.158"],
     ["波浪地形 wave", "98.6%", "11", "0.005", "0.385", "27.6", "0.287"]], 20),
  ...figure(6, "03_报告图片/图表/05_formal_evaluation_comparison.png", 600, 338, "三策略 63 组正式评测六指标对比"),
  P("选型论证：baseline5001 在平均成功率（96.9%）、总摔倒次数（177）、平均碰撞率（0.065）与平均机械功率（24.8 W）四项安全与效率指标上均为三策略最优，故冻结为演示与交付策略；curriculum_seed1_5601 的跟踪 RMSE 最低（0.232 m/s），但摔倒次数高出 34%，仅作为跟踪精度对照，不进入安全演示；curriculum_seed2_5601 综合表现最弱，予以淘汰。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "4.5 策略导出与本地部署链路"),
  P("冻结策略经 play.py 导出为 ONNX 与 JIT 两种推理格式，连同 checkpoint、Go2 MuJoCo 资产与训练日志打包，以 SHA-256 清单哈希交接至本地。本地以 NumPy 重建策略前向计算，与 ONNX Runtime 在 200 组随机观测上逐点比对，最大绝对误差 4.8×10⁻⁶，确认“云端权重”与“本地复跑策略”数值一致后，方可用于报告证据渲染（链路见图 12）。"),
);

// ---------- 5 LLM ----------
children.push(
  H(HeadingLevel.HEADING_1, "5 大模型任务规划与安全监督"),
  H(HeadingLevel.HEADING_2, "5.1 约束式任务规划流程"),
  P("大模型在本系统中只负责任务语义解析：将“巡检园区重点设施，发现异常立即上报”等自然语言转换为结构化任务 JSON（场景类型、航点序列、检查项、速度等级、安全规则、失败回退与终止条件）。任务 JSON 必须通过 JSON Schema 结构校验、任务白名单与参数边界检查后才进入行为树执行；校验不通过的任务被拒绝、转入人工确认或按规则回退。该设计保留了大模型的交互灵活性，同时保证机器人行为的可解释与可审计（图 7）。"),
  ...figure(7, "03_报告图片/图表/d2_llm_task_planning.png", 600, 357, "大模型约束式任务规划与安全仲裁流程"),
  H(HeadingLevel.HEADING_2, "5.2 路线控制与航点跟踪"),
  P("路线控制器将任务 JSON 中的航点序列转换为机体速度指令：以当前航向与目标航点的夹角为误差做比例控制（增益 1.6，偏航角速度限幅 ±1.0 rad/s）；夹角大于 0.9 rad 时先原地转向、对齐后再前进，前进速度同时受任务限速与剩余距离约束。航点到达判定采用 0.36 m 安全容差，避免机器人在航点附近反复修正造成的振荡。该控制器与 RL 策略之间只传递有界速度指令（前进速度、偏航角速度），不改变策略内部结构。"),
  H(HeadingLevel.HEADING_2, "5.3 安全监督机制"),
  P("安全监督器独立于大模型与任务控制器运行，拥有最高控制优先级：①速度限幅——按场景冻结速度上限（巡检 0.50 m/s，灾害 0.35 m/s）；②禁行区与路线校验——任务下发前冻结路线并完成合法性检查；③姿态约束——任务全程监测质心高度与机身倾角，低于安全阈值立即停车；④急停与人工接管接口——任意状态可中断任务。第 6 章双场景运行中机器人最低质心高度分别为 0.297 m 与 0.311 m，全程未触发姿态保护。", { indentFirst: true }),
);

// ---------- 6 双场景仿真验证 ----------
const campusShots = ["01_起点出发","02_主路巡逻","03_发现异常点","04_设施巡检","05_动力机房检查","06_安全返航"].map(s => `03_报告图片/物理仿真截图/campus_security_${s}.png`);
const disasterShots = ["01_进入灾区","02_碎石通道","03_坡道通行","04_目标侦察确认","05_物资投送","06_绕行返航"].map(s => `03_报告图片/物理仿真截图/disaster_response_${s}.png`);
children.push(
  H(HeadingLevel.HEADING_1, "6 双场景仿真验证"),
  H(HeadingLevel.HEADING_2, "6.1 场景一：园区安全巡检"),
  P("任务设定：机器人从园区主入口出发，沿 19 航点冻结路线完成周界与重点设施巡检，覆盖办公楼门禁、西侧围墙盲区、北侧电子围栏、仓库防火门、停车区与动力机房 6 类检查点；行至西侧围墙盲区时触发“发现异常→告警上报”事件，随后继续巡检并安全返航。全程限速 0.50 m/s。任务级演示关键帧如图 8 所示，画面同步呈现二维任务视图、事件时间线、系统链路状态与物理仿真内嵌视频。"),
  ...figure(8, "03_报告图片/关键帧/03_hero_campus_security.png", 600, 338, "场景一园区安全巡检任务级演示关键帧（本地任务级仿真）"),
  P("图 9 给出同一次运行中 MuJoCo 物理仿真视图的六个任务时刻：出发、主路巡逻、发现异常点、设施巡检、动力机房检查与安全返航。画面中机器人由 baseline5001 真实策略闭环驱动，全部根运动来自 MuJoCo 动力学积分。"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, children: [] }),
  imageGrid(campusShots, 200, 113, 3008),
  caption("图", 9, "场景一物理仿真过程截图（本地 MuJoCo，真实策略闭环）"),
  P("运行结果：18/18 检查点全部通过，任务用时 71.7 s，全程最低质心高度 0.297 m（安全阈值 0.18 m），无摔倒、无姿态保护触发（表 6）。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "6.2 场景二：灾害侦察与物资投送"),
  P("任务设定：机器人从安全区出发，经任务入口进入受灾区域，依次通过碎石通道、坡道与烟雾边界，抵达目标侦察点确认待救援目标，在物资投送点完成急救包投送；返航通道被倒塌构件阻断后，系统触发局部重规划，沿安全绕行路线经碎石区出口返回安全区。全程限速 0.35 m/s。任务级演示关键帧如图 10 所示。"),
  ...figure(10, "03_报告图片/关键帧/04_hero_disaster_response.png", 600, 338, "场景二灾害侦察与物资投送任务级演示关键帧（本地任务级仿真）"),
  P("图 11 给出该运行物理仿真视图的六个任务时刻：进入灾区、碎石通道、坡道通行、目标侦察确认、物资投送与绕行返航。场景内碎石、坡道与倒塌构件均为可碰撞刚体，机器人通行过程完全由动力学仿真决定。"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, children: [] }),
  imageGrid(disasterShots, 200, 113, 3008),
  caption("图", 11, "场景二物理仿真过程截图（本地 MuJoCo，真实策略闭环）"),
  P("运行结果：10/10 检查点全部通过，任务用时 117.3 s，全程最低质心高度 0.311 m，无摔倒、无姿态保护触发（表 6）。", { indentFirst: true }),
  ...table(6, "双场景任务路线运行结果（本地 MuJoCo，baseline5001 策略闭环）", [2100, 1700, 1400, 2100, 1726],
    [["场景", "检查点通过", "任务用时", "最低质心高度", "运行状态"],
     ["园区安全巡检", "18 / 18", "71.7 s", "0.297 m（阈值 0.18 m）", "PASS"],
     ["灾害侦察与物资投送", "10 / 10", "117.3 s", "0.311 m（阈值 0.18 m）", "PASS"]]),
  H(HeadingLevel.HEADING_2, "6.3 证据链与可追溯体系"),
  P("报告全部证据遵循“云端训练、本地验证、哈希可溯”的链路（图 12）：训练日志、评测快照、策略权重、仿真素材全部经 SHA-256 校验登记，报告中每一张图均可回溯到生成命令与原始数据。图 13 为本次提交的证据一览拼图。"),
  ...figure(12, "03_报告图片/图表/d4_training_provenance.png", 600, 357, "训练评测链路与素材可追溯体系"),
  ...figure(13, "03_报告图片/图表/08_evidence_contact_sheet.png", 600, 351, "提交证据一览（本地仿真演示 + 云端正式评测）"),
);

// ---------- 7 结果与分析 ----------
children.push(
  H(HeadingLevel.HEADING_1, "7 结果与分析"),
  H(HeadingLevel.HEADING_2, "7.1 正式评测结果分析"),
  P("63 组正式评测表明（表 4、图 6）：baseline5001 以 96.9% 平均成功率居首，且总摔倒次数（177）较第二名少 25.6%、较第三名少 53.5%，平均机械功率最低（24.8 W），呈现“成功率更高、摔倒更少、能耗更省”的综合优势；两个课程学习变体虽然在跟踪 RMSE 上略优（0.232/0.218 vs 0.326 m/s），但以更多摔倒为代价换取跟踪精度，不符合巡检与灾害场景“安全第一”的选型原则。分地形看（表 5），baseline5001 在平地、斜坡、碎石坡三类地形达成 100% 成功率且零摔倒；下台阶（92.1%）与离散障碍（91.9%）是相对薄弱地形，是后续训练的重点方向。"),
  H(HeadingLevel.HEADING_2, "7.2 训练收敛性分析"),
  P("第三阶段训练曲线（图 4）显示，追加 1500 次迭代后奖励平台期稳定、回合长度满格，策略已收敛；同时平台期未再显著抬升，说明该阶段网络容量与课程设置下的性能趋于饱和。这一观察支撑了“冻结 checkpoint 5001、不再盲目续训”的决策，也规避了续训对早期地形课程的遗忘风险。"),
  H(HeadingLevel.HEADING_2, "7.3 任务完成度与安全性分析"),
  P("双场景任务级运行（表 6）实现了两个“全量”：全检查点通过（18/18、10/10）与全程无摔倒（最低质心 0.297/0.311 m，远高于 0.18 m 阈值）。场景一平均速度约 0.44 m/s（含转向与检查点驻留），场景二约 0.30 m/s，均处于安全限速内；场景二在坡道、碎石与通道受阻重规划等扰动下未出现失稳，验证了“路线控制器 + RL 策略”接口的有界速度指令设计在复杂地形任务中的有效性。"),
);

// ---------- 8 经济性 ----------
children.push(
  H(HeadingLevel.HEADING_1, "8 经济性与工程可行性分析"),
  P("算力成本：训练与正式评测依托上海大学自强5000教育计算平台完成，边际算力成本为零；本地复跑与素材渲染在普通工作站（CPU 推理 50 Hz）即可完成，无需持续占用付费 GPU。软件成本：MuJoCo、PyTorch、ONNX Runtime、OpenCV 等全栈开源，授权成本为零。迁移成本：运动策略以 ONNX/JIT 标准格式交付，仅依赖 45 维本体感知与 12 维关节接口，对接企业原型机时只需核对关节顺序、零位、控制频率与通信协议即可完成最小适配（详见参赛方案大纲的原型机适配清单）。", { indentFirst: true }),
);

// ---------- 9 创新 ----------
children.push(
  H(HeadingLevel.HEADING_1, "9 社会价值、创新性与责任边界"),
  H(HeadingLevel.HEADING_2, "9.1 主要创新点"),
  ...table(7, "项目创新点及其证据", [2600, 3626, 2800],
    [["创新点", "说明", "证据"],
     ["约束式大模型任务规划", "大模型只输出可校验任务 JSON，不触碰底层控制，兼顾灵活性与安全性", "图 7；第 5 章"],
     ["独立安全监督架构", "限速/禁行区/姿态/急停四级约束，优先级高于大模型与任务指令", "图 1、图 2；5.3 节"],
     ["可复核证据链", "训练—评测—部署—素材全链路 SHA-256 哈希登记，逐件可复查", "图 12；附录 A"],
     ["三策略对比选型方法", "以 63 组正式评测六指标选型而非单次演示，结论可辩护", "表 4、表 5、图 6"]]),
  H(HeadingLevel.HEADING_2, "9.2 证据真实性与负责任科研"),
  P("本报告严格区分三类证据：云端正式评测数据（量化成绩的唯一依据）、本地 MuJoCo 仿真演示（任务能力的可视化证据，非真机）、平台运行记录截图（过程佐证）。全部仿真画面均明确标注“本地任务级仿真/物理仿真”，不表述为真机实验；全部数据文件哈希登记。团队承诺：不以仿真结果冒充真机结论，不选择性展示失败案例之外的片段。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "9.3 与榜题评审维度的对应"),
  ...table(8, "项目对评审维度的响应", [2600, 6426],
    [["评审维度", "本项目响应"],
     ["大模型与强化学习融合", "大模型任务规划 + RL 运动控制分层融合，接口显式（第 3–5 章）"],
     ["系统完整性与闭环", "任务理解→安全校验→路线控制→运动执行→结果审计全链路（第 3、6 章）"],
     ["量化成绩", "63 组正式评测，成功率 96.9%，双场景全检查点通过（表 4–6）"],
     ["创新性", "约束式规划、安全监督架构、可复核证据链（表 7）"],
     ["工程可行性", "低算力 CPU 部署、标准 ONNX/JIT 接口、原型机适配清单（第 8 章）"]]),
);

// ---------- 10 结论 ----------
children.push(
  H(HeadingLevel.HEADING_1, "10 结论与展望"),
  H(HeadingLevel.HEADING_2, "10.1 研究结论"),
  P("本项目完成了“智巡四足”四足机器人智能控制系统的方案设计、训练评测与双场景仿真验证：①建成“大模型约束式任务规划—安全监督—RL 运动控制”三层架构并通过状态机驱动双场景任务闭环；②在自强5000平台完成三策略 63 组正式评测，冻结的 baseline5001 策略以 96.9% 平均成功率、177 次总摔倒、0.065 平均碰撞率综合领先；③策略权重经哈希交接与数值一致性核验后，在本地 MuJoCo 中以真实策略闭环完成园区巡检（18/18，71.7 s）与灾害响应（10/10，117.3 s）全路线，全程姿态安全；④全部素材经哈希登记可复核。榜题要求的“AI 大模型与强化学习驱动”在本系统中体现为分层融合、显式接口与安全仲裁的工程化落地。", { indentFirst: true }),
  H(HeadingLevel.HEADING_2, "10.2 局限性与后续工作"),
  P("本阶段成果为仿真级验证，主要局限与后续工作包括：①尚未开展真机实验，Sim2Real 差距需在企业原型机上按“悬空→站立→低速平地→复杂地形”分级验证；②下台阶与离散障碍地形成功率仍有约 8 个百分点提升空间，拟通过针对性课程与更长的第三阶段训练改进；③大模型当前为离线任务解析，后续将接入在线多模态感知（视觉/语音）形成“感知—规划—执行”在线闭环；④单机器人作业，未来可扩展多机协同巡检。", { indentFirst: true }),
);

// ---------- 参考文献 ----------
children.push(
  H(HeadingLevel.HEADING_1, "参考文献"),
  ...[
    "[1] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[J]. arXiv:1707.06347, 2017.",
    "[2] Hwangbo J, Lee J, Dosovitskiy A, et al. Learning agile and dynamic motor skills for legged robots[J]. Science Robotics, 2019, 4(26): eaau5872.",
    "[3] Rudin N, Hoeller D, Reist P, et al. Learning to walk in minutes using massively parallel deep reinforcement learning[C]//CoRL, 2022.",
    "[4] Miki T, Lee J, Hwangbo J, et al. Learning robust perceptive locomotion for quadrupedal robots in the wild[J]. Science Robotics, 2022, 7(62): eabk2822.",
    "[5] Ahn M, Brohan A, Brown N, et al. Do as I can, not as I say: Grounding language in robotic affordances (SayCan)[C]//CoRL, 2022.",
    "[6] Liang J, Huang W, Xia F, et al. Code as Policies: Language model programs for embodied control[C]//ICRA, 2023.",
    "[7] Makoviychuk V, Wawrzyniak L, Guo Y, et al. Isaac Gym: High performance GPU-based physics simulation for robot learning[C]//NeurIPS Datasets and Benchmarks, 2021.",
    "[8] Todorov E, Erez T, Tassa Y. MuJoCo: A physics engine for model-based control[C]//IROS, 2012.",
    "[9] Tobin J, Fong R, Ray A, et al. Domain randomization for transferring deep neural networks from simulation to the real world[C]//IROS, 2017.",
    "[10] Unitree Robotics. Unitree Go2 四足机器人产品文档与开发资料[EB/OL]. 2024.",
    "[11] 合肥华驱动力科技有限公司. DG-202609 AI 大模型与强化学习驱动的四足机器人智能控制系统比赛方案[Z]. 2026.",
    "[12] 第十五届“挑战杯”竞赛组委会. 2026 年度中国青年科技创新“揭榜挂帅”擂台赛榜单通知[Z]. 2026.",
  ].map((t) => P(t, { size: 21, line: 276 })),
);

// ---------- 附录 ----------
const manifestRows = manifest.outputs.map((o) => [
  o.file, o.label, o.file.endsWith(".mp4") ? "本地仿真视频" : (o.file.includes("05") || o.file.includes("06") ? "云端评测/训练证据" : "本地生成图表"),
  o.sha256.slice(0, 12) + "…",
]);
children.push(
  new Paragraph({ children: [new PageBreak()] }),
  H(HeadingLevel.HEADING_1, "附录 A 素材包清单与哈希校验"),
  P("表 A-1 列出任务演示素材包全部文件及其 SHA-256 哈希（截断显示前 12 位）。完整哈希见素材包 manifest.json；复核命令：python -m simulation.demo_media.generate_submission_media --verify-only --output-dir outputs/submission_media_20260912。"),
  ...table("A-1", "提交素材清单与哈希登记（manifest.json）", [3000, 3300, 1600, 1126],
    [["文件", "说明", "证据属性", "SHA-256（前 12 位）"], ...manifestRows], 18),
  H(HeadingLevel.HEADING_1, "附录 B 软硬件版本与参数记录"),
  ...table("B-1", "关键软硬件与参数版本", [3600, 5426],
    [["项目", "版本 / 参数"],
     ["机器人模型", "Unitree Go2（MuJoCo MJCF，12 关节）"],
     ["策略 checkpoint", "model_5001.pt（stage3_control_add1500_seed1，第 5001 次迭代）"],
     ["策略网络", "MLP 45→512→256→128→12，ELU"],
     ["控制参数", "PD Kp=20 / Kd=0.5；action scale 0.25；50 Hz；dt=0.002 s，decimation=10"],
     ["观测缩放", "角速度 ×0.25；关节速度 ×0.05；指令 ×[2, 2, 0.25]"],
     ["训练平台", "上海大学自强5000计算平台（Isaac Gym + RSL-RL / PPO）"],
     ["本地仿真", "MuJoCo 3.13；Python 3.14；PyTorch 2.12；ONNX Runtime 1.30"],
     ["正式评测", "63 组（3 策略 × 7 地形 × 21 组），2026-09-11 数据快照"],
     ["本地路线复跑", "园区 18/18（71.7 s）；灾害 10/10（117.3 s），2026-09-12"],
     ["素材哈希校验", "SHA-256 逐项登记（manifest.json），--verify-only 复核通过"]], 20),
  P("报告结束。本文件全部图表与数据可由 LMM-RL-QRC 仓库代码与素材包复现。", { before: 300, italics: true, size: 20 }),
);

// ---------- 组装 ----------
const doc = new Document({
  creator: "智巡四足团队",
  title: "智巡四足研究报告（DG-202609）",
  styles: {
    default: { document: { run: { font: BODY, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: HEI }, paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: HEI }, paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: HEI }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "智巡四足 · DG-202609 揭榜挂帅擂台赛研究报告", font: BODY, size: 18, color: "666666" })] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "- ", font: BODY, size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: BODY, size: 18 }), new TextRun({ text: " -", font: BODY, size: 18 })] })] }),
    },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("REPORT_WRITTEN", OUT, buf.length); });
