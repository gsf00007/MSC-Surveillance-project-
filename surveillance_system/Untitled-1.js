const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, TableOfContents, ImageRun, Header, Footer, PageNumber,
  LevelFormat, convertInchesToTwip, VerticalAlign, TabStopType, TabStopPosition
} = require("docx");

const FONT = "Calibri";
const NAVY = "1f3864";

function P(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: 200, line: 276 },
    children: Array.isArray(text) ? text : [new TextRun({ text, font: FONT, size: 22, italics: opts.italics, bold: opts.bold })],
  });
}

function TR(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: opts.size || 22, bold: opts.bold, italics: opts.italics });
}

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 240 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 32, color: NAVY })],
  });
}
function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 160 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 26, color: NAVY })],
  });
}
function H3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 23 })],
  });
}

function bullet(text) {
  return new Paragraph({
    spacing: { after: 100 },
    numbering: { reference: "bullet-list", level: 0 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}
function numbered(text) {
  return new Paragraph({
    spacing: { after: 100 },
    numbering: { reference: "num-list", level: 0 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 300 },
    children: [new TextRun({ text, font: FONT, size: 20, italics: true })],
  });
}

function imageCentered(path, widthPx, heightPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 80 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(path),
        transformation: { width: widthPx, height: heightPx },
      }),
    ],
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.shade ? { type: ShadingType.CLEAR, fill: opts.shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), font: FONT, size: 20, bold: opts.bold })],
    })],
  });
}

function dataTable(headers, rows, widths) {
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, { width: widths[i], shade: "1F3864", bold: true, align: AlignmentType.CENTER })),
      }),
      ...rows.map((r, ridx) => new TableRow({
        children: r.map((c, i) => cell(c, { width: widths[i], align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, shade: ridx % 2 === 1 ? "F2F2F2" : undefined })),
      })),
    ],
  });
}

function tableTitle(text) {
  return new Paragraph({
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: FONT, size: 20, bold: true })],
  });
}

const codeFont = "Consolas";
function codeBlock(lines) {
  return new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    spacing: { before: 120, after: 120 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
    },
    children: lines.flatMap((l, i) => {
      const run = new TextRun({ text: l, font: codeFont, size: 18, break: i === 0 ? 0 : 1 });
      return [run];
    }),
  });
}

// ---------- TITLE PAGE ----------
const titlePage = [
  new Paragraph({ spacing: { after: 1200 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new TextRun({ text: "UNIVERSITY OF BIRMINGHAM DUBAI", font: FONT, bold: true, size: 30, color: NAVY })],
  }),
  new Paragraph({ spacing: { after: 800 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "Multi-Camera Person Localization, AI Threat Prioritization and Automated Emergency Dispatch", font: FONT, bold: true, size: 32 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [new TextRun({ text: "A surveillance system that locates people, assesses situational severity and initiates emergency response under human oversight", font: FONT, italics: true, size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [new TextRun({ text: "MSc Dissertation", font: FONT, bold: true, size: 24 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new TextRun({ text: "Name - Gundlur Shaik Fahad", font: FONT, bold: true, size: 24 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [new TextRun({ text: "Supervisor - Anis Zarrad", font: FONT, bold: true, size: 24 })],
  }),
  new Paragraph({ spacing: { after: 1000 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new TextRun({ text: "MSc Artificial Intelligence and Machine Learning", font: FONT, size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new TextRun({ text: "University of Birmingham Dubai", font: FONT, size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new TextRun({ text: "2025-26", font: FONT, size: 22 })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- ABSTRACT ----------
const abstract = [
  H1("Abstract"),
  P("Buildings are saturated with cameras, yet almost all of that footage is watched by nobody. A guard monitoring a dozen screens misses things, and the events worth catching are precisely the rare ones that slip past a tired observer. This project asks whether a camera network can work out where a person is, judge how serious their situation looks, and contact the appropriate emergency service without waiting for a human to notice."),
  P("The system built here chains seven stages. YOLOv8 and ByteTrack locate and follow people within each camera, while an appearance model recognises the same individual across separate views. MediaPipe extracts body pose, reduced to a thirteen-joint representation shared between training and deployment. A recurrent network summarises one second of movement and a support vector machine assigns one of five classes. A threat engine then applies duration, confidence and history filters to produce a red, amber or green severity. Only after a ten-second window in which an operator may cancel does the dispatch layer place a telephone call."),
  P("Evaluation used three public datasets. Movement classification was trained on a subset of NTU RGB+D 120 and tested on held-out participants, reaching a macro F1 of 0.677 with collapse detection at F1 0.91 and perfect precision. No serious event was ever labelled as normal. Re-identification was fine-tuned on Market-1501, where it reached 87.1 per cent matching accuracy, and evaluated on WildTrack, which was never used for training, where it reached 24.8 per cent rank-1. Applying the threat engine reduced false alarms from 10.2 to 4.1 per cent while dispatching 76 per cent of genuine collapses."),
  P("Three findings emerged. Features drawn from intermediate layers of an ImageNet network outperform its final layer for re-identification, because that final layer encodes object category and every input is already the same category. Re-identification transfers poorly between datasets, a gap quantified here rather than cited. And the dispatch threshold is a policy decision rather than an optimisation, since its correct value depends on whether an operator is watching."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- ACKNOWLEDGEMENTS ----------
const ackn = [
  H1("Acknowledgements"),
  P("I am sincerely grateful to my supervisor, Anis Zarrad, for his guidance, patience and constructive feedback throughout the course of this project. His willingness to question design decisions and push for clearer justification shaped the direction of this work at every stage, and his support was instrumental in bringing it to completion."),
  P("I would also like to thank the University of Birmingham Dubai and the School of Computer Science for providing the academic environment, resources and computing facilities that made this research possible."),
  P("Finally, I extend my gratitude to my family, friends and colleagues for their patience and encouragement throughout the course of this degree. Their support made this an intellectually rewarding and personally meaningful journey."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- HONOUR CODE ----------
const honour = [
  H1("Honour Code"),
  P("I certify that this project is my own work. The research questions, system design, experimental decisions, implementation, evaluation, interpretation of results and conclusions presented in this report are my own. Generative AI tools were used to support the development and documentation of this project."),
  P("Generative AI tools were used as coding and technical assistants throughout the implementation. I used these tools to discuss implementation approaches, clarify technical concepts, generate and refine code, troubleshoot errors and explore alternative approaches. I reviewed, modified, executed, debugged and validated the resulting code and experiments myself. Generative AI was also used to assist with structuring, drafting, proofreading and improving the clarity of sections of this report."),
  P("The pretrained and fine-tuned models used within the system, including YOLOv8, ByteTrack, MediaPipe, the ResNet-based re-identification descriptors, the LSTM/SVM movement classifier and SHAP, were used as components of the system being developed and evaluated rather than as independent sources of experimental results. Model selection, configuration, experimentation, evaluation and interpretation were conducted as part of my own project work."),
  P("All experimental results, figures, tables and quantitative findings reported in this dissertation are based on experiments conducted by me. AI-generated suggestions and outputs were reviewed and verified against my implementation, experimental outputs and project records before being incorporated into the report. I remain responsible for the accuracy, integrity and originality of the work presented here."),
  new Paragraph({ spacing: { before: 400 }, children: [TR("Name: Gundlur Shaik Fahad", { bold: true })] }),
  new Paragraph({ children: [TR("Programme: MSc Artificial Intelligence and Machine Learning", { bold: true })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- TOC / LISTS ----------
const tocSection = [
  H1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),
];

function listOfFigures() {
  const figs = [
    ["Figure 1", "System architecture. Each stage produces an inspectable intermediate result.", "3"],
    ["Figure 2", "Re-identification training on Market-1501 over 20 epochs.", "3"],
    ["Figure 3", "Per-class precision, recall and F1 on held-out participants.", "4"],
    ["Figure 4", "Severity assigned to each class of test clip by the complete system.", "4"],
    ["Figure 5", "Effect of the red-severity confidence threshold on dispatch rate, operator alerting and false alarms.", "4"],
    ["Figure 6", "Rank-1 accuracy by re-identification descriptor.", "4"],
  ];
  return [
    H1("List of Figures"),
    ...figs.map(f => new Paragraph({
      spacing: { after: 120 },
      tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX, leader: "dot" }],
      children: [new TextRun({ text: `${f[0]}  ${f[1]}`, font: FONT, size: 20 }), new TextRun({ text: `\t${f[2]}`, font: FONT, size: 20 })],
    })),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function listOfTables() {
  const tabs = [
    ["Table 1", "Principal components and their configuration.", "3"],
    ["Table 2", "Training data composition after windowing.", "3"],
    ["Table 3", "Per-class classification performance on held-out participants.", "4"],
    ["Table 4", "Classifier performance compared with complete system behaviour.", "4"],
    ["Table 5", "Threshold sweep results.", "4"],
    ["Table 6", "Re-identification performance on WildTrack.", "4"],
  ];
  return [
    H1("List of Tables"),
    ...tabs.map(f => new Paragraph({
      spacing: { after: 120 },
      tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX, leader: "dot" }],
      children: [new TextRun({ text: `${f[0]}  ${f[1]}`, font: FONT, size: 20 }), new TextRun({ text: `\t${f[2]}`, font: FONT, size: 20 })],
    })),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ===================== CHAPTER 1 =====================
const ch1 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("CHAPTER 1", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Introduction", { bold: true, size: 36, })] }),

  H2("1.1 Motivation"),
  P("Closed-circuit television is now close to ubiquitous in commercial and public buildings, but its effectiveness rests on an assumption that rarely holds: that somebody is watching. Control rooms are typically staffed by one or two operators responsible for dozens of feeds, and sustained visual monitoring is a task humans perform poorly. Vigilance research has consistently found that detection performance for infrequent events degrades sharply within the first half hour of a monitoring shift. The consequence is a system that records everything and notices very little."),
  P("A second limitation is less often discussed. Conventional tracking operates within a single camera. When somebody walks out of one field of view and into another, most installations treat the second appearance as an unrelated person. The footage exists to reconstruct a route afterwards, but nothing in the live system knows where anyone currently is. For a security team responding to an incident, that distinction matters a great deal."),
  P("There is a third gap at the point of response. Even where an incident is noticed, the operator must interpret what they are seeing, decide it warrants escalation, telephone the appropriate service, and describe a location accurately while under stress. Each of those steps costs time, and each is a point at which the response can fail."),

  H2("1.2 Problem Statement"),
  P("This project addresses a single question: can a multi-camera system establish where somebody is, judge how serious their circumstances appear, and initiate an appropriate emergency response without waiting for instruction, while remaining accountable to a human operator?"),
  P("The question decomposes into four technical problems. People must be detected and followed reliably. The same individual must be recognised across camera boundaries. Body movement must be classified into categories that distinguish an emergency from ordinary activity. And the resulting classification must be converted into a decision about whether to act, in a way that can be explained afterwards."),

  H2("1.3 Aim and Objectives"),
  P("The aim is to design, build and evaluate a surveillance system that follows people across multiple cameras, judges the severity of observed events, and contacts emergency services when warranted, while presenting the reasoning behind each decision to a human operator who retains the ability to intervene."),
  P("Six objectives follow from this aim:"),
  numbered("Detect and track people within each camera feed using established real-time methods."),
  numbered("Recognise the same person across separate camera views and associate them with a location in the building."),
  numbered("Convert body pose into features suitable for classification, and train a classifier to distinguish emergency movement from ordinary activity."),
  numbered("Apply temporal and historical filtering to convert instantaneous classifications into a severity judgement."),
  numbered("Route confirmed emergencies to the appropriate service through a telephony interface, with a mandatory window for human cancellation."),
  numbered("Generate an explanation for every alert, and evaluate the system against metrics appropriate to a safety-critical application."),

  H2("1.4 Research Questions"),
  P("Three questions guided the experimental work:"),
  bullet("RQ1. Does a movement classifier trained on laboratory motion-capture data transfer to genuine surveillance footage, given the substantial differences in camera placement, resolution and subject behaviour?"),
  bullet("RQ2. Which representation of visual appearance performs best for cross-camera person re-identification, and does a purpose-trained model outperform features borrowed from a general-purpose image classifier?"),
  bullet("RQ3. How should the confidence threshold governing automatic dispatch be selected, given that false alarms and missed emergencies carry asymmetric costs?"),

  H2("1.5 Contributions"),
  P("The work makes four contributions. First, a complete implementation spanning detection through to emergency dispatch, with a human-in-the-loop safeguard built into the architecture rather than added afterwards. Second, an empirical comparison of six appearance representations for re-identification, including an explanation for why intermediate network layers outperform the final layer on this task. Third, a quantified measurement of the cross-dataset generalisation gap in re-identification, obtained on the author's own data rather than reported from the literature. Fourth, a threshold analysis that reframes the dispatch decision as a policy choice conditioned on operational context."),

  H2("1.6 Report Structure"),
  P("Chapter 2 reviews existing work in detection, tracking, re-identification, action recognition and explainability, and identifies the gap this project occupies. Chapter 3 sets out the research approach, dataset selection, system design and evaluation methodology. Chapter 4 presents the experimental results. Chapter 5 discusses their interpretation, acknowledges limitations, considers ethical implications, and proposes further work. Chapter 6 concludes."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===================== CHAPTER 2 =====================
const ch2 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("CHAPTER 2", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Background and Related Work", { bold: true, size: 36 })] }),

  H2("2.1 Person Detection and Tracking"),
  P("Single-stage object detectors have displaced earlier region-proposal architectures in applications where latency matters. The YOLO family predicts bounding boxes and class scores in a single forward pass, and successive versions have narrowed the accuracy gap with slower two-stage detectors while retaining real-time throughput [1]. For this project the detector is treated as a solved component: person detection is well served by publicly available weights trained on COCO, and retraining it would consume effort without producing a research contribution."),
  P("Tracking is a distinct problem. A detector has no memory, so linking detections across frames requires an association step. ByteTrack [2] departs from earlier trackers in retaining low-confidence detections during association rather than discarding them. The reasoning is that a detection whose confidence has dropped is often a partially occluded object rather than a false positive, and occlusion is exactly the circumstance in which continuity matters most. In crowded indoor scenes this materially reduces identity switches."),
  P("Both approaches share a limitation: they operate within a single camera. Neither carries information across a field-of-view boundary."),

  H2("2.2 Person Re-Identification"),
  P("Re-identification addresses that boundary. The task is to determine whether two images captured by different cameras depict the same individual, using appearance alone. It is difficult because viewpoint, illumination and pose vary between cameras while the underlying identity does not."),
  P("Early approaches relied on hand-crafted descriptors, typically colour histograms computed over horizontal bands of the body. These are computationally trivial and remain a reasonable baseline, but they degrade badly when clothing colours are similar or illumination differs between cameras. Learned representations largely superseded them, and the standard formulation trains a network to classify identities in a labelled gallery, then discards the classification head and uses the penultimate representation as a descriptor. Market-1501 [3] is among the most widely used benchmarks for this, comprising 1,501 identities photographed by six cameras."),
  P("A recurring difficulty in the literature is that models trained on one dataset transfer poorly to another. Published results frequently report high accuracy when training and test data originate from the same capture environment, and substantially lower accuracy when they do not. WildTrack [4] provides a useful counterpoint to Market-1501: seven synchronised, calibrated cameras observing a public square, with consistent identity labels across all views. Its overhead viewpoints and small pedestrian crops present conditions markedly different from Market-1501's approximately head-height capture."),

  H2("2.3 Skeleton-Based Action Recognition"),
  P("Recognising what a person is doing can proceed from raw video or from an intermediate skeletal representation. The latter has advantages for this application: it discards appearance, which is irrelevant to whether someone has fallen, and it reduces dimensionality substantially. NTU RGB+D 120 [5] is the standard benchmark, containing over 114,000 clips across 120 action classes recorded with a depth sensor that reports 25 three-dimensional joint positions per frame."),
  P("Architecturally, recurrent networks and graph convolutional networks are both established for this task. Graph convolutional approaches treat the skeleton as a graph whose edges encode anatomical connectivity, and generally achieve higher accuracy on standard benchmarks. Recurrent architectures are simpler to implement and train, and remain competitive at modest data scales."),
  P("A practical obstacle arises when training data and deployment data use different skeletal conventions. NTU's depth sensor reports 25 joints; MediaPipe [7], a widely used pose estimator for ordinary RGB video, reports 33 landmarks in a different order and with different anatomical definitions. This mismatch is easy to overlook because it produces no error at runtime."),

  H2("2.4 Explainability in Safety-Critical Systems"),
  P("A system capable of summoning emergency services occupies a different regulatory and ethical position from one that merely logs events. Where an automated decision has consequences for third parties, the ability to reconstruct the basis of that decision becomes a requirement rather than a convenience."),
  P("SHAP [6] provides a model-agnostic attribution method grounded in cooperative game theory, assigning each input feature a contribution to the difference between a prediction and a baseline. It is computationally tractable for classical models such as support vector machines and considerably more expensive for deep networks, which has implications for architectural choice in systems where explanation is required."),

  H2("2.5 The Research Gap"),
  P("The individual components above are mature. Detection, tracking, re-identification, action recognition and attribution each have established methods and benchmarks. What is less well covered is their integration into a system that acts on its conclusions."),
  P("Most published work on surveillance analytics terminates at classification. A model outputs a label and the pipeline ends. The question of how a probabilistic classification should be converted into a decision to summon an ambulance, what temporal evidence should be required first, and how accountability should be distributed between an automated system and a human operator, receives comparatively little attention. This project occupies that space, and treats the conversion from classification to action as a design problem in its own right."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===================== CHAPTER 3 =====================
const ch3 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("CHAPTER 3", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Methodology, System Design and Evaluation Strategy", { bold: true, size: 34 })] }),

  H2("3.1 Research Approach"),
  P("This work follows a design science methodology. The output is an artefact, and knowledge is generated through building it and evaluating its behaviour rather than through hypothesis testing on a pre-existing dataset. This is appropriate because the central question concerns whether a particular system architecture can achieve a practical outcome, which cannot be established analytically."),
  P("Within that frame, individual components are evaluated experimentally against public benchmarks. The two modes are complementary: the design science element addresses whether the integrated system works, while the experimental element establishes how well each part performs and permits comparison against published results."),

  H2("3.2 Why a Modular Pipeline"),
  P("An alternative design would train a single network to map video directly to a severity decision. This was rejected for three reasons."),
  P("Such a model would require training data of a form that does not exist: video of genuine emergencies in the target environment, labelled with the correct dispatch decision. Assembling it would be impractical and, for realistic quantities, unethical."),
  P("An end-to-end model also offers no intermediate quantities to inspect. When the system escalates, an operator needs to know what it observed, and a single opaque score does not provide this. The modular design exposes a detection, a track, an identity, a pose sequence, a class label and a confidence at each stage."),
  P("Finally, modularity permits component substitution. The appearance descriptor was replaced six times during evaluation without altering anything downstream, which would be impossible in a monolithic model."),

  H2("3.3 Dataset Selection"),
  P("Three public datasets were used, each for a specific purpose."),
  P("NTU RGB+D 120 supplies movement classification training data. Its principal advantage is that labels are encoded in filenames, eliminating manual annotation, and that skeleton data is distributed separately from video, reducing the download from terabytes to a few gigabytes. Its principal disadvantage is that actions are performed on instruction by volunteers in a laboratory, which differs from spontaneous behaviour in a public building."),
  P("Market-1501 supplies re-identification training data, chosen because it is large, standard, and its bounding boxes were produced by an automatic detector rather than drawn by hand, which better reflects deployment conditions."),
  P("WildTrack supplies re-identification evaluation data and was deliberately excluded from all training. Restricting it to evaluation means every reported figure derives from a model that has never encountered a WildTrack image, eliminating any possibility of information leakage and permitting the transfer gap to be measured directly."),

  H2("3.4 Evaluation Strategy"),
  P("Two decisions shape the evaluation and are worth stating explicitly."),
  P("Data is partitioned by participant rather than by clip. If the same individual appeared in both training and test partitions, the classifier could achieve high accuracy by learning that person's idiosyncratic movement rather than the movement class itself. Partitioning by participant guarantees that every test subject is unfamiliar, which lowers reported accuracy but measures the quantity of interest."),
  P("The system is evaluated at two levels. Classifier-level metrics describe the model in isolation. System-level metrics describe behaviour after the threat engine's filters are applied. These differ substantially, and reporting only the first would misrepresent what the deployed system does."),

  H2("3.5 Architecture Overview"),
  P("The system comprises seven stages arranged as a pipeline. Each frame passes through detection, tracking, re-identification, pose extraction, classification, threat assessment and, when warranted, dispatch. Table 1 summarises the principal components; the pipeline is illustrated conceptually below."),
  tableTitle("Pipeline stages"),
  dataTable(
    ["Stage", "Function"],
    [
      ["1. Camera input", "Multiple synchronised camera feeds"],
      ["2. Detection & tracking", "YOLOv8 person boxes per frame; ByteTrack links one ID per person; 30-frame crop buffer"],
      ["3. Re-identification", "Appearance vector compared by cosine similarity (threshold 0.72); zone mapping via homography to floor plan"],
      ["4. Pose & features", "MediaPipe 33 keypoints reduced to 13 shared points; hip-centred, torso-scaled normalisation"],
      ["5. Classification", "LSTM reads a 30-frame sequence into a 128-number summary; SVM makes a calibrated five-class decision; SHAP explains which features decided it"],
      ["6. Threat prioritisation", "RED (confidence \u2265 0.75, held 10 s), YELLOW (confidence 0.65\u20130.74), GREEN (below 0.65, logged only)"],
      ["7. Dispatch & console", "Twilio voice API places the call after a 10-second operator override window; operator dashboard shows every decision"],
    ],
    [2800, 6900]
  ),
  caption("Figure 1: System architecture. Each stage produces an inspectable intermediate result."),
  P("Two branches operate on the same cropped person region but extract different information. The re-identification branch answers who and where; the pose branch answers what they are doing. They converge at the threat engine."),

  H2("3.6 Detection and Tracking"),
  P("YOLOv8 in its nano configuration performs detection, filtered to the person class. Detections below 0.45 confidence are discarded, which removes most spurious boxes on bags and furniture while retaining partially occluded people."),
  P("ByteTrack maintains identity within each camera, using a Kalman filter to predict positions and associating detections in two passes, first for high-confidence detections and then for the remainder. Every third frame is processed rather than every frame; human posture changes negligibly across a thirtieth of a second, and the reduction yields roughly threefold throughput improvement."),

  H2("3.7 Cross-Camera Re-Identification"),
  P("Each cropped person is converted to a fixed-length descriptor and compared against a gallery of previously observed individuals by cosine similarity. Similarity above 0.72 assigns the existing global identity; otherwise a new identity is created. Stored descriptors are updated by exponential moving average rather than replacement, so that a single poor observation does not corrupt an entry."),
  P("Three descriptor families were implemented and compared. The colour signature divides the body into six horizontal bands and computes a hue-saturation histogram over each, with histogram equalisation applied first to reduce inter-camera brightness variation. The ImageNet descriptor extracts activations from a chosen ResNet50 [8] layer, pooled within four horizontal stripes to preserve the vertical arrangement of clothing. The trained descriptor is a ResNet18 fine-tuned on Market-1501, described in Section 3.11."),

  H2("3.8 Pose Extraction and the Joint Mapping Problem"),
  P("MediaPipe estimates 33 body landmarks per cropped person. A complication arises immediately: the training data uses a different skeletal convention."),
  P("NTU RGB+D reports 25 joints derived from a depth sensor. MediaPipe reports 33 landmarks including facial features absent from the NTU skeleton, in a different index order. A model trained on one convention and deployed on the other receives numerically valid input whose semantics are entirely different. Critically, this produces no runtime error; the system continues to emit confident predictions that carry no information."),
  P("The resolution adopted here reduces both conventions to the thirteen joints they share: head, both shoulders, elbows, wrists, hips, knees and ankles. Two index arrays perform the mapping, one applied when reading training files and one applied to live pose output. Training and inference therefore operate on an identical representation."),
  codeBlock([
    "# the thirteen joints both conventions provide, in one order",
    "NTU_TO_COMMON = [3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]",
    "MEDIAPIPE_TO_COMMON = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]",
    "",
    "frames.append(first_person[NTU_TO_COMMON])   # when reading NTU",
    "joints = full_landmarks[MEDIAPIPE_TO_COMMON]  # at inference",
  ]),
  caption("Listing 1: Joint mapping. Applying an index array reorders and subsets in one operation, so training and deployment receive identically arranged input."),
  P("Normalisation follows in two steps. Every joint is translated so the midpoint of the hips lies at the origin, removing the person's position within the frame. All coordinates are then divided by torso length, computed as the median distance between hip and shoulder midpoints across the sequence, removing the effect of distance from the camera. The median rather than mean is used because occasional frames with undetected shoulders would otherwise distort the scale factor."),
  codeBlock([
    "L_HIP, R_HIP = 7, 8",
    "L_SHO, R_SHO = 1, 2",
    "hips = (seq[:, L_HIP] + seq[:, R_HIP]) / 2.0",
    "shoulders = (seq[:, L_SHO] + seq[:, R_SHO]) / 2.0",
    "seq -= hips[:, None, :]                       # position removed",
    "torso = np.linalg.norm(shoulders - hips, axis=1)",
    "good = torso[torso > 1e-6]",
    "scale = np.median(good) if good.size else 1.0",
    "return (seq / (scale + 1e-8))                 # scale removed",
  ]),
  caption("Listing 2: Normalisation. The median is used rather than the mean so that frames with undetected shoulders cannot distort the scale factor."),
  P("This normalisation is what permits NTU data, measured in metres by a depth sensor, and MediaPipe output, expressed in normalised image coordinates, to be compared at all."),

  H2("3.9 Temporal Windowing"),
  P("Classification operates on thirty-frame windows, approximately one second, with a stride of fifteen frames producing fifty per cent overlap. A single frame is insufficient: a person bent forward may be falling or retrieving a dropped object, and only the trajectory distinguishes these. The overlap doubles the number of training samples obtainable from a given clip and ensures the model encounters events beginning at varying positions within the window."),

  H2("3.10 Movement Classification"),
  P("Classification is performed by two models in sequence. A two-layer LSTM with 256 hidden units reads the thirty-frame sequence and produces a 128-dimensional summary from its final timestep. A support vector machine with a radial basis function kernel then assigns one of five classes from that summary."),
  P("Placing a classical model at the decision point is a deliberate architectural choice with a cost. A softmax layer on the LSTM would be simpler and might perform marginally better. However, attribution methods are considerably more tractable for a support vector machine than for a recurrent network, and in a system that can summon an ambulance, the ability to explain a decision was judged more valuable than a small accuracy gain. This trade-off is revisited in Section 5.1."),
  P("The support vector machine is configured with probability estimation enabled, without which the entire threat engine would be inoperable, and with balanced class weighting to prevent under-represented classes being ignored."),

  H2("3.11 Re-Identification Model Training"),
  P("A ResNet18 initialised from ImageNet weights was fine-tuned on Market-1501. The ImageNet classification head was replaced with a layer sized to the 736 training identities remaining after filtering out detector artefacts and individuals with fewer than four images."),
  P("Two loss functions were combined. Cross-entropy trains the network to identify which of the 736 known individuals an image depicts. Batch-hard triplet loss [9] trains the property actually required at deployment: that two images of the same unfamiliar person should lie close together in the representation space, and two images of different people far apart. For each image in a batch the loss selects the most dissimilar same-identity image and the most similar different-identity image, and penalises any margin violation between them."),
  P("Triplet loss requires multiple images of the same identity within each batch, so batches were assembled as eight identities with four images each rather than by random sampling. Input resolution was set to 128 by 64 pixels, deliberately small: WildTrack pedestrian crops measure between 30 and 55 pixels wide, and training at higher resolution would create a mismatch with the evaluation conditions."),
  imageCentered("fig2_training.png", 470, 246),
  caption("Figure 2: Re-identification training on Market-1501 over 20 epochs. Matching accuracy on held-out identities is the quantity of interest; naming accuracy on training identities is scaffolding."),

  H2("3.12 Threat Prioritisation"),
  P("The classifier observes one second at a time, which is an insufficient basis for summoning an ambulance. Three additional tests are applied before an event is treated as an emergency."),
  P("The duration test requires that a red-class event persist for ten seconds. This is expressed in seconds rather than frames, a distinction that proved important during development. An earlier implementation counted processed frames, which meant that the same configured value corresponded to half a second on thirty-frame-per-second video and nearly four seconds on a webcam running at four frames per second. Timing is now derived from video timestamps or wall-clock time as appropriate, so the configured value has consistent meaning."),
  P("The confidence test requires 0.75 or above for red and 0.65 for amber. The selection of 0.75 is examined in Section 4.4."),
  P("The history test escalates to red when three separate warning episodes occur for one individual within sixty seconds, catching gradual deterioration that never produces a single high-confidence detection. Episodes are counted rather than individual frames; continuous activity produces one warning per processed frame, and counting those directly would trigger escalation within a fraction of a second."),
  codeBlock([
    "if action in RED_CLASSES:",
    "    if confidence >= RED_CONFIDENCE and held >= RED_HOLD_SECONDS:",
    "        return \"RED\"          # both tests satisfied",
    "    if episodes >= ESCALATE_AFTER:",
    "        return \"RED\"          # repeated warnings",
    "    if held < RED_HOLD_SECONDS:",
    "        return \"YELLOW\"       # not yet held long enough",
    "    return \"YELLOW\"           # confidence below the bar",
  ]),
  caption("Listing 3: Threat engine decision logic. Severity is decided by explicit rules operating on the classifier's output rather than by the model itself, so each condition can be tuned and explained independently."),

  H2("3.13 Dispatch and Human Oversight"),
  P("Red-severity events initiate a ten-second countdown displayed prominently on the operator console with a cancellation control. Absent cancellation, a call is placed through the Twilio [10] voice API, delivering a synthesised message stating the camera, zone, event type and confidence, and bridging the on-duty guard into the same call."),
  P("The countdown is not primarily a usability feature. Its purpose is that no automatic call can occur without a human having had the opportunity to prevent it, which distributes accountability appropriately between the system and its operators. Ten seconds is short enough to be immaterial in a genuine emergency and long enough for an attentive operator to intervene."),
  P("Every dispatch decision, including cancellations, is written to an audit database recording timestamp, identity, class, confidence, operator action and call reference. During all development the destination number was a test handset."),

  H2("3.14 Implementation Summary"),
  tableTitle("Table 1: Principal components and their configuration."),
  dataTable(
    ["Component", "Implementation", "Key parameters"],
    [
      ["Detection", "YOLOv8-nano, COCO weights", "confidence 0.45, person class only"],
      ["Tracking", "ByteTrack", "every 3rd frame processed"],
      ["Re-identification", "ResNet18 fine-tuned on Market-1501", "cosine threshold 0.72"],
      ["Pose", "MediaPipe Pose", "33 landmarks reduced to 13 joints"],
      ["Temporal model", "2-layer LSTM", "256 hidden, 128-dim output"],
      ["Classifier", "SVM, RBF kernel", "C = 10, balanced class weights"],
      ["Threat engine", "Rule-based filters", "red at 0.75 held 10 s"],
      ["Dispatch", "Twilio voice API", "10 s cancellation window"],
    ],
    [2400, 3800, 3500]
  ),

  H2("3.15 Evaluation Datasets"),
  P("Movement classification was evaluated on NTU RGB+D 120. The available subset contained 3,865 skeleton files spanning 60 action classes, of which five were relevant to this application. After parsing and windowing, 986 sequences were obtained from 36 distinct participants."),
  P("Five NTU action classes were mapped to project labels: fall down to collapse, staggering to stagger, punching to assault, cross hands in front to a call for help, and a routine standing action to normal. The last mapping is a known weakness and is discussed in Section 5.4."),
  tableTitle("Table 2: Training data composition after windowing."),
  dataTable(
    ["Class", "Source action", "Sequences", "Partition"],
    [
      ["collapse", "A043 fall down", "187", "train 114 / val 35 / test 38"],
      ["stagger", "A027 staggering", "184", "train 138 / val 29 / test 17"],
      ["assault", "A024 punching", "192", "train 143 / val 34 / test 15"],
      ["sos", "A040 cross hands", "116", "train 79 / val 21 / test 16"],
      ["normal", "A001 routine action", "307", "train 194 / val 64 / test 49"],
    ],
    [1800, 2600, 1800, 3500]
  ),
  P("Partitioning was by participant: 26 individuals for training, 5 for validation and 5 for testing. No participant appears in more than one partition."),
  P("Re-identification was evaluated on WildTrack. Sixty annotated frames yielded 75 labelled individuals observed by seven cameras, producing 1,831 usable person crops after 47 were rejected as too small to describe meaningfully. This dataset was held back entirely from every training procedure described above."),

  H2("3.16 Metrics"),
  P("Classification is assessed by per-class precision, recall and F1, with macro-averaged F1 as the summary statistic. Macro averaging is preferred to accuracy because class sizes are unequal and the minority classes are the consequential ones."),
  P("Two additional metrics reflect the safety context. The false alarm rate is the proportion of normal clips that would trigger an emergency response. The missed event rate is the proportion of genuine emergencies classified as normal. These are asymmetric in consequence: a false alarm wastes a vehicle, while a missed collapse may cost a life."),
  P("Re-identification is assessed by rank-1 and rank-5 accuracy and by mean average precision. Queries are restricted to cameras other than the one in which the probe image was captured; permitting same-camera matches would allow consecutive frames of one person to inflate the result."),

  H2("3.17 Experimental Procedure"),
  P("Six appearance descriptors were compared on identical crops, so that any difference is attributable to the descriptor rather than to data selection. The comparison covered a colour histogram baseline, ResNet50 features drawn from the final layer and from layers two, three and four, a concatenation of colour and layer three features, and the Market-1501 fine-tuned model."),
  P("The dispatch confidence threshold was swept from 0.60 to 0.95 in increments of 0.05. At each setting the complete pipeline was applied to the test set, with each clip simulated as a continuous fourteen-second event, and three quantities recorded: the proportion of genuine emergencies auto-dispatched, the proportion alerting an operator at any severity, and the false alarm rate."),
  P("A qualitative transfer test was also performed. A publicly available doorbell camera recording of a delivery driver collapsing from heat exhaustion was processed by the complete system. This footage differs from the training data in camera height, angle, resolution and the fact that the collapse was genuine rather than performed."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===================== CHAPTER 4 =====================
const ch4 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("CHAPTER 4", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Experimental Results", { bold: true, size: 36 })] }),

  H2("4.1 Movement Classification"),
  P("Table 3 reports per-class performance on the held-out test partition of 135 sequences from five unseen participants."),
  tableTitle("Table 3: Per-class classification performance on held-out participants."),
  dataTable(
    ["Class", "Precision", "Recall", "F1", "Support"],
    [
      ["collapse", "1.00", "0.84", "0.91", "38"],
      ["stagger", "0.45", "0.59", "0.51", "17"],
      ["assault", "0.36", "0.53", "0.43", "15"],
      ["sos", "0.65", "0.69", "0.67", "16"],
      ["normal", "0.93", "0.80", "0.86", "49"],
      ["macro average", "0.68", "0.69", "0.68", "135"],
    ],
    [2900, 1700, 1700, 1700, 1700]
  ),
  imageCentered("fig3_perclass.png", 470, 260),
  caption("Figure 3: Per-class precision, recall and F1. Collapse and normal, the two classes on which the system's usefulness depends, are the strongest."),
  P("Performance is markedly uneven. Collapse achieves F1 0.91 with precision 1.00, meaning no clip of any other class was ever labelled a collapse. Normal activity reaches F1 0.86. These are the two classes on which the system's practical value depends."),
  P("Stagger and assault perform poorly, at F1 0.51 and 0.43. Inspection of the confusion pattern shows they are largely confused with one another, which is explicable: both involve rapid, asymmetric upper-body motion, and in a thirteen-joint skeletal representation stripped of appearance they are genuinely similar. Each class had approximately 66 source clips available, which is a small quantity from which to learn a fine distinction."),
  P("The missed event rate was 0.0 per cent: no clip of a serious class was labelled normal. The classifier-level false alarm rate was 10.2 per cent, with five of 49 normal clips assigned to a red class. Mean inference latency was 0.9 milliseconds per sequence."),

  H2("4.2 Classifier Compared with the Complete System"),
  P("The classifier-level false alarm rate is not the rate at which the deployed system would raise alarms, because the threat engine's filters intervene. Table 4 reports both."),
  tableTitle("Table 4: Classifier performance compared with complete system behaviour."),
  dataTable(
    ["Metric", "Classifier alone", "Complete system"],
    [
      ["False alarms on normal clips", "10.2% (5 of 49)", "4.1% (2 of 49)"],
      ["Collapses auto-dispatched", "not applicable", "76% (29 of 38)"],
      ["Collapses alerting an operator", "not applicable", "89% (34 of 38)"],
      ["Collapses entirely unnoticed", "not applicable", "11% (4 of 38)"],
    ],
    [4000, 2600, 2600]
  ),
  imageCentered("fig4_severity.png", 470, 260),
  caption("Figure 4: Severity assigned to each class of test clip by the complete system. Twenty-nine of 38 genuine collapses reach red severity."),
  P("The confidence and duration requirements remove 60 per cent of the classifier's false alarms. Of 38 genuine collapses, 29 would have resulted in an automatic ambulance call, a further five would have alerted an operator at amber severity, and four would have passed unnoticed."),
  P("Amber alerts should not be counted as failures where an operator is present, since they appear on the console with an escalation control. On that reading the system responds to 89 per cent of collapses. Where no operator is watching, only the 76 per cent auto-dispatch figure is meaningful."),

  H2("4.3 Transfer to Genuine Surveillance Footage"),
  P("The doorbell camera recording provided the strongest single test of RQ1. The classifier had been trained exclusively on depth-sensor recordings of volunteers performing scripted actions at approximately head height in controlled lighting. The test footage was captured by a consumer camera mounted above a doorway, viewing a real collapse from heat exhaustion."),
  P("The system identified the collapse at 85 to 86 per cent confidence, escalated from amber to red as the duration requirement was satisfied, and routed to ambulance. Given the magnitude of the domain difference, this result was not anticipated and represents meaningful evidence that skeletal representation, after normalisation, transfers across capture conditions more robustly than appearance-based approaches."),

  H2("4.4 Selection of the Dispatch Threshold"),
  P("Figure 5 presents the threshold sweep addressing RQ3."),
  imageCentered("fig5_threshold.png", 470, 260),
  caption("Figure 5: Effect of the red-severity confidence threshold on dispatch rate, operator alerting and false alarms."),
  tableTitle("Table 5: Threshold sweep results."),
  dataTable(
    ["Threshold", "Auto-dispatched", "Operator alerted", "False alarms"],
    [
      ["0.60", "74%", "77%", "6%"],
      ["0.70", "66%", "77%", "6%"],
      ["0.75", "64%", "77%", "4%"],
      ["0.80", "58%", "77%", "4%"],
      ["0.85", "38%", "77%", "4%"],
      ["0.90", "17%", "77%", "2%"],
      ["0.95", "13%", "77%", "0%"],
    ],
    [2200, 2600, 2600, 2200]
  ),
  P("At the initially chosen threshold of 0.90, only 17 per cent of genuine emergencies produced an automatic call. Reducing the threshold to 0.75 raised this to 64 per cent while increasing false alarms from 2 to 4 per cent, corresponding to one additional false call across 49 normal clips."),
  P("The proportion of events alerting an operator is invariant at 77 per cent across the entire range. This is informative: the threshold governs only whether an event escalates from amber to red, not whether it is noticed at all. Where an operator is present, threshold selection is therefore less consequential than it first appears."),
  P("The duration requirement proved non-binding at these settings, producing identical outcomes from two to ten seconds. The confidence threshold is doing the filtering work."),

  H2("4.5 Re-Identification"),
  P("Table 6 reports the six-way descriptor comparison on WildTrack, addressing RQ2."),
  tableTitle("Table 6: Re-identification performance on WildTrack, which was never used for training."),
  dataTable(
    ["Descriptor", "Rank-1", "Rank-5", "mAP", "ms per crop"],
    [
      ["Colour histogram", "13.3%", "22.2%", "5.3%", "0.2"],
      ["ResNet50 final layer", "15.0%", "25.0%", "5.3%", "78.7"],
      ["ResNet50 layer 2", "17.6%", "27.5%", "6.2%", "42.3"],
      ["ResNet50 layer 4", "18.7%", "31.1%", "6.8%", "172.7"],
      ["ResNet50 layer 3", "23.1%", "35.8%", "8.1%", "147.7"],
      ["Fine-tuned on Market-1501", "24.8%", "40.5%", "9.5%", "11.9"],
    ],
    [3400, 1500, 1500, 1500, 1900]
  ),
  imageCentered("fig6_reid.png", 470, 260),
  caption("Figure 6: Rank-1 accuracy by descriptor. Performance rises as features are drawn from progressively earlier network layers, peaking at layer three."),
  P("Two patterns are evident. Accuracy increases as features are drawn from earlier layers, rising from 15.0 per cent at the final layer to 23.1 per cent at layer three before declining to 17.6 per cent at layer two. And the purpose-trained model, at 24.8 per cent, outperforms all ImageNet variants while requiring an order of magnitude less computation than the deeper ImageNet features, owing to its smaller backbone and reduced input resolution."),
  P("The same fine-tuned model achieved 87.1 per cent matching accuracy on held-out Market-1501 identities. The disparity between 87.1 per cent on Market-1501 and 24.8 per cent on WildTrack is examined in Section 5.2."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===================== CHAPTER 5 =====================
const ch5 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("CHAPTER 5", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Discussion and Future Work", { bold: true, size: 36 })] }),

  H2("5.1 Why Intermediate Layers Outperform the Final Layer"),
  P("The layer-depth pattern in Table 6 has a straightforward explanation that is worth stating precisely, because it is a property of the task rather than an artefact of this implementation."),
  P("ResNet was trained on ImageNet to assign images to object categories. Representations at successive depths become progressively more specialised toward that objective, and the final layer before classification encodes, in effect, the answer to the question of what kind of object is present. For the re-identification task this representation is degenerate: every input crop depicts a person, so the final layer produces approximately the same output for all of them. The property that makes the network effective at its original task renders it uninformative here."),
  P("Earlier layers retain texture, colour distribution and garment shape, which are precisely the attributes distinguishing individuals. Layer two performs worse than layer three because it is too early: its representations capture edges and colour patches without sufficient spatial organisation to describe a person's overall appearance."),
  P("The practical implication is that borrowing features from a general-purpose classifier requires attention to which layer is used, and that the conventional choice of the penultimate layer is inappropriate when all inputs share a category."),

  H2("5.2 Cross-Dataset Generalisation"),
  P("The fine-tuned model reaches 87.1 per cent on Market-1501 identities it has never seen, confirming that it learned re-identification rather than memorising individuals. On WildTrack it reaches 24.8 per cent."),
  P("The datasets differ in ways that plausibly account for this. Market-1501 was captured at approximately head height with pedestrian crops of reasonable resolution; WildTrack observes a crowded square from elevated cameras with crops between 30 and 55 pixels wide and considerable illumination variation between viewpoints. The model learned an appearance representation suited to the first environment, and that representation does not transfer."),
  P("This is consistent with the wider re-identification literature, where cross-dataset performance is routinely far below within-dataset performance. The contribution here is not the observation but its measurement: the figure was obtained on this system, with these components, under conditions where training contamination is impossible."),
  P("A preliminary experiment fine-tuning the Market model on WildTrack identities, with a partition by identity so that test individuals remained unseen, produced a rank-1 improvement from 70.2 to 80.0 per cent and mAP from 16.3 to 30.1 per cent under that evaluation protocol. This confirms the gap is attributable to domain shift rather than to a limitation of the approach. It is not reported as a headline figure because its evaluation protocol differs from that used in Table 6 and the two are not directly comparable."),

  H2("5.3 Were the Objectives Achieved?"),
  P("Objectives one, four, five and six were met. Detection and tracking function reliably; the threat engine converts classifications into severity judgements with measurable effect; dispatch operates with the intended human safeguard; and explanations are generated for every alert."),
  P("Objective three was partially met. Collapse detection, the class of primary importance, reached F1 0.91. Stagger and assault did not reach a standard suitable for deployment."),
  P("Objective two was met in implementation but only partially in performance. Cross-camera recognition functions and was measured, but 24.8 per cent rank-1 on unfamiliar footage is below what deployment would require."),

  H2("5.4 Limitations"),
  P("The normal class is too narrowly defined. The NTU action used to represent normal activity depicts a stationary person performing a routine gesture. Ordinary building activity encompasses walking, carrying objects, conversing and sitting. The classifier's conception of normal is consequently much narrower than reality, and this is the most probable source of false alarms in deployment. Widening this class is the single highest-value improvement available."),
  P("Two classes perform inadequately. Stagger and assault, at F1 0.51 and 0.43, are unreliable. The proximate cause is limited training data, approximately 66 clips each, combined with genuine similarity in skeletal representation."),
  P("Re-identification does not transfer well. Discussed in Section 5.2. The system would require fine-tuning on footage from any specific deployment site."),
  P("Training data depicts performed rather than genuine actions. NTU participants were instructed to fall. Real collapses differ in ways that are difficult to characterise. The doorbell camera test provides encouraging evidence but a single recording is not a basis for confidence."),
  P("Fire detection is not implemented. The dispatch layer supports routing to a fire service, but no detector supplies it. Smoke and flame recognition requires pixel-level colour and flicker analysis, an entirely different model from skeletal action recognition."),
  P("Live demonstration used a single camera. The pipeline supports multiple feeds and re-identification was evaluated across WildTrack's seven cameras, but end-to-end operation was demonstrated on one."),
  P("Explanations were not evaluated with users. Whether the generated explanations assist an operator in deciding quickly and correctly is unmeasured. Establishing this would require a study with practitioners."),

  H2("5.5 Ethical Considerations"),
  P("A system that continuously tracks and identifies individuals raises concerns that are not resolved by technical performance. Three merit explicit statement."),
  P("The capability constructed here is general. A system that recognises a person across cameras for safety purposes is architecturally identical to one that does so for behavioural monitoring. The safeguards adopted were to restrict the label set to safety-relevant classes and to retain only audit records rather than footage, but these are configuration choices that a subsequent operator could reverse."),
  P("Automated dispatch consumes a shared public resource. A false ambulance call may render a vehicle unavailable elsewhere. This asymmetry motivated both the confidence threshold and the human cancellation window, and it is the reason false alarm rate was treated as a primary metric rather than a secondary one."),
  P("Detection and re-identification models are known to perform unevenly across demographic groups. This system was not evaluated for such disparity, and no claim about its fairness is made. Any deployment should establish this before use, as differential false alarm rates would translate directly into differential contact with emergency services."),

  H2("5.6 Future Work"),
  P("Four directions follow from the limitations above, in order of expected value."),
  P("Broadening the normal class with recordings of genuine everyday movement would most directly address the principal source of false alarms, and requires data collection rather than architectural change."),
  P("Fine-tuning re-identification on footage from the target camera network, with partitioning by identity, would address the transfer gap. The preliminary experiment described in Section 5.2 indicates the improvement available."),
  P("Replacing the recurrent temporal model with a graph convolutional or transformer architecture would likely improve the weaker classes, at the cost of complicating explanation and increasing computational requirements."),
  P("Incorporating audio would provide a complementary signal. A scream or breaking glass is detectable when the camera view is obstructed, and agreement between independent modalities would raise confidence in a dispatch decision more than improving either alone."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===================== CHAPTER 6 =====================
const ch6 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("CHAPTER 6", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Conclusion", { bold: true, size: 36 })] }),

  P("The question that opened this project was whether a camera network could locate an individual, form a judgement about the seriousness of what it was observing, and summon assistance on its own initiative. A complete system was built and evaluated in order to answer it."),
  P("The system detects and tracks people, recognises them across camera boundaries, extracts and normalises body pose, classifies movement into five categories, applies temporal and historical filtering to produce a severity judgement, and places an emergency call after a mandatory interval during which an operator may intervene."),
  P("Collapse detection, the capability on which the system's value principally rests, achieved F1 0.91 with perfect precision on participants absent from training. No serious event was classified as normal. The threat engine reduced false alarms from 10.2 to 4.1 per cent while dispatching 76 per cent of genuine collapses and alerting an operator to 89 per cent. The classifier, trained on laboratory motion capture, correctly identified a genuine collapse in consumer doorbell footage at 85 per cent confidence."),
  P("Three findings extend beyond the artefact. Intermediate layers of an ImageNet network outperform its final layer for re-identification, because that final layer encodes a category shared by every input; the effect was measured across four depths and explained mechanistically. Re-identification transfers poorly between capture environments, with the same model scoring 87.1 per cent on its training dataset and 24.8 per cent on an unseen one. And the dispatch threshold is a policy decision rather than an optimisation, since operator presence determines whether amber alerts constitute a response."),
  P("The system's significant weaknesses are known and stated: a normal class too narrow to represent ordinary activity, two movement classes of inadequate accuracy, and re-identification that would require site-specific tuning. None invalidates the architecture, and each has an identified route to improvement."),
  P("The wider contribution is the treatment of the step from classification to action as a design problem deserving explicit attention. A classifier emitting a probability is not a system that decides. The interval between them, comprising temporal evidence, confidence thresholds, historical context and human oversight, is where a safety-critical system is actually determined, and it merits the same scrutiny customarily given to model architecture."),
  new Paragraph({ children: [new PageBreak()] }),
];

// ===================== APPENDIX =====================
const appendix = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("APPENDIX A", { bold: true, size: 22 })] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("Software", { bold: true, size: 32 })] }),
  P("The implementation consists of 23 Python files organised by function. Full source is submitted separately. This appendix lists the files so that the mapping between the report and the code is clear."),

  H2("A.1 Core Pipeline"),
  tableTitle("Table 7: Files implementing the main processing chain."),
  dataTable(
    ["File", "Purpose", "Report section"],
    [
      ["config.py", "All configurable parameters in one place", "3.14"],
      ["skeleton.py", "NTU parsing, joint mapping, normalisation", "3.8"],
      ["prepare_data.py", "Builds training sequences from NTU files", "3.15"],
      ["train.py", "Trains the LSTM and the SVM", "3.10"],
      ["run_live.py", "The complete pipeline running on a camera", "3.5"],
    ],
    [2800, 4700, 2200]
  ),

  H2("A.2 Decision and Response"),
  tableTitle("Table 8: Files converting a classification into an action."),
  dataTable(
    ["File", "Purpose", "Report section"],
    [
      ["threat.py", "Duration, confidence and history filters", "3.12"],
      ["dispatch.py", "Countdown, cancellation, call, audit log", "3.13"],
      ["dashboard.py", "Operator console served over HTTP", "3.13"],
      ["explain.py", "Plain-language and SHAP explanations", "2.4"],
      ["setup_calls.py", "Telephony credential check and test call", "3.13"],
    ],
    [2800, 4700, 2200]
  ),

  H2("A.3 Re-Identification and Localisation"),
  tableTitle("Table 9: Files implementing cross-camera recognition."),
  dataTable(
    ["File", "Purpose", "Report section"],
    [
      ["reid.py", "Appearance descriptors and person gallery", "3.7"],
      ["train_reid.py", "Fine-tunes the descriptor on Market-1501", "3.11"],
      ["calibrate.py", "Derives a homography from four floor points", "3.7"],
      ["demo_reid.py", "Demonstrates recognition on ordinary video", "\u2014"],
    ],
    [2800, 4700, 2200]
  ),

  H2("A.4 Evaluation"),
  tableTitle("Table 10: Files producing the results reported in Chapter 4."),
  dataTable(
    ["File", "Produces", "Report section"],
    [
      ["evaluate.py", "Per-class metrics, confusion matrix", "4.1"],
      ["eval_system.py", "Classifier versus complete system comparison", "4.2"],
      ["tune_thresholds.py", "Threshold sweep and operating point", "4.4"],
      ["compare_reid.py", "Six-way descriptor comparison", "4.5"],
      ["eval_wildtrack.py", "Rank-1, rank-5 and mAP on WildTrack", "4.5"],
      ["eval_location.py", "Inter-camera position agreement", "3.7"],
    ],
    [2800, 4700, 2200]
  ),

  H2("A.5 Reproducing the Results"),
  P("With the three datasets in place and their paths set in config.py, the reported figures are reproduced by running the following in order. A fixed random seed is set throughout, so repeated runs give identical partitions and results."),
  codeBlock([
    "python prepare_data.py        # NTU -> training sequences",
    "python train.py               # LSTM + SVM, split by participant",
    "python evaluate.py            # Table 3, Figure 3",
    "python eval_system.py         # Table 4, Figure 4",
    "python tune_thresholds.py     # Table 5, Figure 5",
    "python train_reid.py          # fine-tune on Market-1501, Figure 2",
    "python compare_reid.py        # Table 6, Figure 6",
  ]),
  caption("Listing 4: Commands reproducing every reported result, in order."),
];

// ===================== REFERENCES =====================
function ref(num, text) {
  return new Paragraph({
    spacing: { after: 160 },
    indent: { left: 400, hanging: 400 },
    children: [new TextRun({ text: `[${num}] `, font: FONT, size: 20 }), new TextRun({ text, font: FONT, size: 20 })],
  });
}
const references = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 0, after: 300 }, children: [TR("References", { bold: true, size: 32 })] }),
  ref(1, "Jocher, G., Chaurasia, A. and Qiu, J. (2023) Ultralytics YOLOv8. Available at: https://github.com/ultralytics/ultralytics"),
  ref(2, "Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W. and Wang, X. (2022) 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box', Proceedings of the European Conference on Computer Vision (ECCV), pp. 1\u201321."),
  ref(3, "Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J. and Tian, Q. (2015) 'Scalable Person Re-identification: A Benchmark', Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 1116\u20131124."),
  ref(4, "Chavdarova, T., Baqu\u00e9, P., Bouquet, S., Maksai, A., Jose, C., Bagautdinov, T., Lettry, L., Fua, P., Van Gool, L. and Fleuret, F. (2018) 'WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection', Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5030\u20135039."),
  ref(5, "Liu, J., Shahroudy, A., Perez, M., Wang, G., Duan, L.-Y. and Kot, A.C. (2019) 'NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding', IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(10), pp. 2684\u20132701."),
  ref(6, "Lundberg, S.M. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', Advances in Neural Information Processing Systems (NeurIPS), 30, pp. 4765\u20134774."),
  ref(7, "Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M.G., Lee, J., Chang, W.-T., Hua, W., Georg, M. and Grundmann, M. (2019) 'MediaPipe: A Framework for Building Perception Pipelines', arXiv preprint arXiv:1906.08172."),
  ref(8, "He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep Residual Learning for Image Recognition', Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770\u2013778."),
  ref(9, "Hermans, A., Beyer, L. and Leibe, B. (2017) 'In Defense of the Triplet Loss for Person Re-Identification', arXiv preprint arXiv:1703.07737."),
  ref(10, "Twilio Inc. (2024) Programmable Voice API Documentation. Available at: https://www.twilio.com/docs/voice"),
];

// ===================== DOCUMENT =====================
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
      },
      {
        reference: "num-list",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
    },
  },
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: titlePage,
    },
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [TR("Multi-Camera Surveillance and Emergency Dispatch", { size: 16 })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
      children: [
        ...abstract, ...ackn, ...honour, ...tocSection, ...listOfFigures(), ...listOfTables(),
        ...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...appendix, ...references,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/mnt/user-data/outputs/MSc_Final_Report_Reformatted.docx", buf);
  console.log("written");
});