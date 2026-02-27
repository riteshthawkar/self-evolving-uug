# Framework Architecture Figure Specification
## Paper: "Self-Evolving Unified Understanding and Generation Models through Unsupervised Self-Consistency"

---

## OVERVIEW

Create a single, clean, research-paper-quality figure showing the complete self-evolving training framework. The figure should be landscape orientation, suitable for a full-width (two-column) placement in an ECCV paper. Use a clean, professional color palette (no garish colors). Suggested palette: soft blue for understanding components, soft orange/amber for generation components, gray for the frozen backbone, green for reward/signal flow. Use rounded rectangles for modules, solid arrows for data flow, and dashed arrows for reward/gradient signal flow.

The figure has THREE main regions arranged left-to-right or as two panels side by side:
- LEFT PANEL (or TOP): The Understanding Loop
- RIGHT PANEL (or BOTTOM): The Generation Loop
- BOTTOM STRIP: Model-Agnostic Design (small, compact)

A central element (the Frozen Unified Backbone with LoRA adapters) connects both loops.

---

## CENTRAL ELEMENT: Frozen Backbone with Role Decomposition

Draw a large rounded rectangle in the center representing the "Pretrained Unified Model" with a lock icon or "FROZEN" label to indicate parameters are fixed. The label should read: "Frozen Unified Backbone (theta)".

From this backbone, three lightweight adapter branches emerge (drawn as small colored blocks attached to the backbone):

1. **Proposer adapter (phi_p)** — colored in light blue
   - Label: "Proposer LoRA (phi_p)"
   - Function annotation: "Generates visual questions"

2. **Solver adapter (phi_s)** — colored in light green or teal
   - Label: "Solver LoRA (phi_s)"
   - Function annotation: "Answers questions / Evaluates generation"
   - IMPORTANT: This adapter should visually connect to BOTH the understanding loop and the generation loop, emphasizing the solver's dual role. Draw it with a subtle visual indicator (e.g., a double border or a star/badge) labeled "Dual Role: Learner + Evaluator"

3. **Generator adapter (phi_g)** — colored in light orange/amber
   - Label: "Generator LoRA (phi_g)"
   - Function annotation: "Synthesizes images from text"

All three adapters sit on top of the same frozen backbone, sharing its representations. Make this visually clear by showing them as small colored tabs attached to the same gray block.

---

## LEFT PANEL: Understanding Loop (Self-Consistent Understanding)

This panel shows the cyclic flow of the understanding training step. Use a light blue background tint for this panel. Label the panel header: "Understanding Step".

### Flow (numbered steps in the figure):

**Step 1: Input**
- An unlabeled image icon (generic photo placeholder, labeled "Unlabeled Image I") enters the loop.

**Step 2: Proposer generates question**
- Arrow from Image to Proposer adapter.
- Proposer outputs a question "q". Show a small speech bubble or text box: "q: What is the spatial relationship between..."
- Label: "Proposer generates open-ended visual question"

**Step 3: Prompt-Perturbed Self-Consistency (PPS)**
- The question q is sent to the Solver, but THROUGH N different prompt framings.
- Show this as the question q entering a "Prompt Perturbation" module (small box with label "N Prompt Framings: rho_1, rho_2, ..., rho_N").
- From this module, N parallel arrows go to N copies of the Solver (or show a single Solver block with "x N" annotation).
- Each path produces an answer: a_1, a_2, ..., a_N.
- These N answers converge into a "Majority Vote + Entropy" computation block.
- Output from this block: two signals:
  - "H_sc: Sample Entropy" (the self-consistency entropy)
  - "a*: Majority Answer"
- Show the entropy formula nearby or as a small annotation: H_sc = -sum(p_c ln p_c)

**Step 4: Solver Token Entropy (STE)**
- Separately (parallel to PPS), show one single greedy forward pass from the Solver.
- Arrow from Solver (single pass) to a block labeled "Token-Level Entropy".
- Inside or beside this block, show a small depiction of per-token entropy values (e.g., a small bar chart of H_1, H_2, ..., H_K with one bar highlighted as the maximum).
- Output: "STE: d_ste" (normalized difficulty score in [0,1]).
- Small annotation: "max-token entropy, quantile normalized"

**Step 5: Reward Computation for Proposer**
- H_sc and d_ste flow into a "Proposer Reward" block.
- Show the combination: R_p = 0.70 * d_ste + 0.30 * r_sc
- Dashed arrow (reward signal) flows back to the Proposer adapter labeled "GRPO Update".

**Step 6: Reward Computation for Solver**
- The agreement pattern (which answers matched majority, which did not) flows into a "Solver Reward" block.
- Annotation: "+reward if agrees with majority, -penalty if disagrees"
- Dashed arrow (reward signal) flows back to the Solver adapter labeled "REINFORCE Update".

### Key visual element in Understanding Loop:
- Show a small curved arrow or cycle symbol indicating that as the proposer generates harder questions, the solver gets better, which in turn pushes the proposer further. Label this: "Progressive Difficulty Escalation".

---

## RIGHT PANEL: Generation Loop (Generation Assessment via Internal Evaluation)

This panel shows the cyclic flow of the generation training step. Use a light orange/amber background tint. Label the panel header: "Generation Step".

### Flow (numbered steps in the figure):

**Step 1: Input**
- Same unlabeled image icon "Real Image I" and an associated text prompt/caption "t" (e.g., extracted or generated caption of the real image).

**Step 2: Generator synthesizes image**
- Arrow from text prompt "t" to Generator adapter.
- Generator outputs a synthesized image (show a different-looking image icon, labeled "Generated Image I_hat").
- Label: "Generator synthesizes image from text"

**Step 3: QA Fidelity Scoring**
- Show the Solver being used in evaluator mode (use the same Solver block from the understanding loop, with a visual connection or "same module" indicator).
- Two parallel paths:
  - Path A: Solver answers M diagnostic questions on the Real Image I, producing reference answers {a_ref_1, ..., a_ref_M}.
  - Path B: Solver answers the SAME M questions on the Generated Image I_hat, producing answers {a_hat_1, ..., a_hat_M}.
- Both sets of answers feed into a "Compare" block.
- Output: "S_fid: QA Fidelity Score" (fraction of matching answers).
- Small annotation: S_fid = (1/M) * sum(1[a_hat_j = a_ref_j])

**Step 4: Cycle-Consistent Captioning**
- Solver generates a caption of the Generated Image I_hat, producing caption "t_hat".
- The original prompt "t" and reconstructed caption "t_hat" feed into a "Semantic Similarity" block.
- Also, the Generated Image I_hat and original prompt "t" feed into a "Visual-Textual Alignment" block.
- Both outputs combine into: "S_cyc: Cycle Consistency Score".
- Small annotation: S_cyc = 0.5 * sim(f(t), f(t_hat)) + 0.5 * sim_vl(I_hat, t)

**Step 5: Combined Generation Reward**
- S_fid and S_cyc (plus diversity S_div and contradiction penalty S_ctr shown as smaller inputs) flow into a "Generation Reward" block.
- Output: R_g
- Dashed arrow (reward signal) flows back to the Generator adapter labeled "REINFORCE Update".

### Key visual element in Generation Loop:
- Highlight the solver's dual role with a prominent visual connector showing the SAME solver module being used as evaluator here AND as learner in the understanding loop. Draw a curved bridge or connector between the two panels through the Solver, labeled: "Solver: Internal Evaluator (Understanding improves Generation quality)"

---

## CONNECTING THE TWO LOOPS

Between the two panels, show:

1. A cyclic arrow or timeline showing the training schedule:
   - "U steps (Understanding)" followed by "G steps (Generation)" repeating.
   - Label: "Training alternates: U understanding steps then G generation steps"

2. The critical coupling arrow from the Understanding Loop to the Generation Loop through the Solver:
   - "As solver improves in understanding loop, its evaluations in the generation loop become more discriminative"
   - This is a thick or highlighted arrow, because this is the core mechanism that couples the two modalities.

---

## BOTTOM STRIP: Model-Agnostic Design

Below the main figure, add a compact horizontal strip showing three small icons or blocks representing the three generation paradigms:

1. **Diffusion-based** (e.g., BLIP3o)
   - Small icon: iterative denoising process (show noise gradually becoming an image)
   - Label: "Diffusion (iterative denoising)"

2. **Flow-matching** (e.g., BAGEL)
   - Small icon: flow-based transformation (show a latent space arrow transforming to an image)
   - Label: "Flow-matching (VAE + flow decoder)"

3. **Autoregressive discrete-token** (e.g., VARGPT)
   - Small icon: sequential token prediction (show discrete tokens being laid out left to right forming an image)
   - Label: "Autoregressive (discrete codebook tokens)"

Above these three icons, place a bracket or header: "Model-Agnostic: Same algorithmic framework applied across three generation paradigms"

A key visual message: the entire framework (proposer, solver, PPS, STE, QA fidelity, cycle consistency) is IDENTICAL across all three. Only the generation interface changes. Show this by having the three icons connected to the same "Framework" block above with identical arrows.

---

## ANNOTATIONS AND LABELS TO INCLUDE

Place these as text annotations at appropriate locations in the figure:

1. Near the frozen backbone: "No human annotations. No external reward models. Unlabeled images only."
2. Near the STE block: "Continuous difficulty signal even when all solvers agree"
3. Near the PPS block: "Robustness of understanding, not stochastic noise"
4. Near the QA Fidelity block: "Understanding quality directly determines generation signal"
5. Near the cycle consistency block: "Internal similarity, no external scorer"

---

## STYLE GUIDELINES

- **Color coding**:
  - Understanding components: shades of blue (light blue background, darker blue for blocks)
  - Generation components: shades of orange/amber (light orange background, darker orange for blocks)
  - Frozen backbone: gray
  - Solver (dual role): teal or green, standing out from both loops
  - Reward/gradient signals: dashed green arrows
  - Data flow: solid dark gray or black arrows

- **Typography**: Use a clean sans-serif font. All labels should be legible at the figure size typically used in ECCV papers (full-width, approximately 17cm wide).

- **No architecture-specific details**: Do NOT show transformer blocks, attention heads, or any architecture internals. Keep everything at the functional/role level. The point is that the framework is architecture-agnostic.

- **No emojis or informal symbols**: This is a scientific figure.

- **Arrow legend**: Include a small legend in a corner:
  - Solid arrow = data flow
  - Dashed arrow = reward/gradient signal
  - Dotted arrow with bridge = solver dual-role connection

---

## FIGURE CAPTION (for the paper)

"Overview of the proposed self-evolving framework. Given only unlabeled images and a frozen unified backbone, three lightweight LoRA adapters instantiate a proposer, solver, and generator. During understanding steps (left), the proposer generates visual questions, the solver answers them under diverse prompt perturbations, and the resulting self-consistency entropy together with solver token entropy provide the training signal for both the proposer (via GRPO) and the solver (via REINFORCE). During generation steps (right), the generator synthesizes images from text, and the solver evaluates them through question-answering fidelity scoring and cycle-consistent captioning, directly coupling understanding quality to the generation training signal. The framework requires no human annotations or external reward models. The same algorithmic components are applied identically across three architecturally distinct generation paradigms (bottom), demonstrating model-agnostic design."

---

## ALTERNATIVE LAYOUT SUGGESTION

If a left-right layout is too wide, consider a top-bottom layout:
- TOP: Understanding Loop (horizontal flow, left to right)
- MIDDLE: Frozen Backbone with three LoRA adapters (horizontal bar)
- BOTTOM-LEFT: Generation Loop (horizontal flow, left to right)
- BOTTOM-RIGHT: Model-Agnostic strip (vertical stack of three paradigms)

The solver should be positioned at the interface between the understanding and generation sections, visually straddling both, to emphasize its dual role.

---

## CHECKLIST: What the figure MUST communicate at a glance

A reader looking at this figure for 10 seconds should understand:
- [ ] There is a frozen backbone with three lightweight adapters (proposer, solver, generator)
- [ ] Understanding loop: proposer asks questions, solver answers under perturbations, disagreement = signal
- [ ] STE provides a continuous signal even when all answers agree
- [ ] Generation loop: generator makes images, solver evaluates them (QA fidelity + cycle consistency)
- [ ] The solver connects both loops (dual role: learner + evaluator)
- [ ] No external supervision anywhere (everything is self-derived)
- [ ] Works across multiple generation architectures (model-agnostic)
