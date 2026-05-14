# ECCV Rebuttal Working Notes

Last updated: 2026-05-03

This is a working document for consolidating reviewer feedback, tracking which points are valid, and drafting a unified rebuttal strategy. The goal is not to defend every sentence of the paper; the goal is to identify the real risks, answer misunderstandings cleanly, and prioritize the highest-value additions to the rebuttal.

## Current Review Snapshot

- Reviewer 1: `3 / Borderline Reject`
  - Contribution type: `Algorithms/General`
  - Confidence: `5 / Expert`
  - Main themes: novelty, coupling justification, STE formulation, missing implementation detail
- Reviewer 2: `4 / Borderline Accept`
  - Contribution type: `Algorithms/General`
  - Confidence: `3 / Moderate`
  - Main themes: generation evaluation breadth, STE heuristic motivation, staged-training stability, data-source overlap
- Reviewer 3: `4 / Borderline Accept`
  - Contribution type: `Algorithms/General`
  - Confidence: `3 / Moderate`
  - Main themes: scalability ceiling, compute cost, generation evaluation breadth, full fine-tuning instability

## Cross-Review Themes

These are the recurring concerns that will likely matter most in the final rebuttal:

1. **Coupling claim is oversold**
   - Reviewer 1 challenges the claimed bidirectional loop directly.
   - Reviewer 2 asks for staged-vs-joint training.
   - Best rebuttal posture: concede that the coupling is asymmetric in parameter flow, and defend the framework as a **shared-Solver coupled alternating training recipe**, not fully symmetric co-training.

2. **STE is useful but looks heuristic**
   - Reviewers 1 and 2 both question the `max-over-tokens` design and want more motivation or ablation.
   - Best posture: frame `max-over-early-tokens` as a deliberate frontier-seeking heuristic that preserves decisive uncertainty spikes; if possible, support with an extra ablation against mean aggregation.

3. **Implementation details need to be made explicit**
   - Reviewer 1 specifically asks about cycle consistency models, reward-term definitions, prompt framing, reference answers, and token-length handling.
   - Many of these are already answerable from the code and supplement; the rebuttal should make that explicit and avoid sounding evasive.

4. **Generation evaluation breadth**
   - Reviewers 2 and 3 both want stronger generation evaluation beyond GenEval.
   - If no new benchmark numbers can be added, the rebuttal needs to justify why GenEval was the primary matched benchmark and acknowledge this as a limitation rather than pretending it is comprehensive.

5. **Scalability / efficiency / cost**
   - Reviewer 3 is the strongest on this.
   - The rebuttal should give concrete compute numbers and position the method as a **post-training recipe for extracting latent capability from a frozen unified model**, not as a claim of unlimited scaling.

## Reviewer 1 Working Notes

### Overall assessment

This is the most important review. It is technically strong, internally coherent, and attacks the paper at the level of algorithmic framing rather than only presentation. The biggest risk is not that every criticism is correct; the biggest risk is that this reviewer has identified where the manuscript currently overclaims.

### Point-by-point triage

#### W1. Limited novelty

- **Assessment:** partly valid
- **Risk level:** high
- **Best response:**
  - Do not argue primitive novelty too aggressively.
  - Reframe the contribution as:
    1. a self-evolving recipe that requires no external reward model,
    2. joint improvement of understanding and generation in one framework,
    3. unchanged transfer across diffusion, flow-matching, and autoregressive unified models.
- **Avoid:** claiming STE or QA+cycle are individually revolutionary.

#### W2. Weak coupling justification / ungrounded bidirectionality

- **Assessment:** largely valid in the current paper wording
- **Risk level:** critical
- **Best response:**
  - Explicitly concede that the interaction is **asymmetric**, not fully symmetric.
  - State that the framework couples the loops through a shared Solver, with the strongest direction being understanding `->` generation.
  - Argue that the contribution is not “full bidirectional co-training,” but a coupled alternating framework in which the Solver simultaneously acts as learner and internal evaluator.
- **Best possible evidence to add:** joint vs two-stage baseline
- **If no experiment is available:** do not overstate. Soften to “shared-evaluator coupling” and “asymmetric positive transfer.”

#### W3. Max-over-tokens heuristic

- **Assessment:** reasonable question, but rebuttable
- **Risk level:** medium
- **Existing technical support:**
  - The implementation computes entropy over the **first few generated tokens**, not the whole answer.
  - This is intended to catch early uncertainty spikes on decisive content tokens before trailing low-information tokens dilute the signal.
- **Best response:**
  - Explain why average entropy is less suitable for short-answer VQA because it is diluted by answer-closing tokens.
  - If possible, provide `max` vs `mean` ablation.

#### W4. Rolling-window normalization

- **Assessment:** rebuttable
- **Risk level:** medium
- **Existing evidence:** supplementary one-factor sweep over STE window sizes
- **Best response:**
  - Point directly to the supplementary ablation over `64 / 128 / 256`.
  - Emphasize that the adopted value is stable rather than uniquely optimal.

#### W5. Reference answer quality in generation evaluation

- **Assessment:** reviewer likely misunderstood the implementation
- **Risk level:** medium
- **Best response:**
  - Clarify that the Proposer generates **questions**, but reference answers are produced by the Solver on the **real/source image**.
  - State that this design grounds QA fidelity in the source image rather than in arbitrary Proposer outputs.

#### W6. Prompt-dependence of token distributions

- **Assessment:** mostly a misunderstanding
- **Risk level:** medium
- **Best response:**
  - Clarify that STE is not a cross-framing token-distribution consistency constraint.
  - Self-consistency is the framing-level signal; STE is an independent scalar uncertainty signal computed from solver decoding.

#### W7. Cycle consistency implementation unspecified

- **Assessment:** valid clarification gap
- **Risk level:** medium
- **Best response:**
  - State the exact frozen models used per backbone family:
    - BLIP3o: model-internal text embedding similarity with token-overlap fallback
    - BAGEL: frozen `openai/clip-vit-base-patch32`
    - VARGPT: frozen `openai/clip-vit-base-patch32`
  - This is a reproducibility omission, not a conceptual defect.

#### W8. S_div and S_ctr undefined

- **Assessment:** valid
- **Risk level:** medium
- **Best response:**
  - Define `S_div` as per-candidate diversity reward within the generation batch.
  - Define `S_ctr` as a contradiction penalty driven by yes/no polarity disagreement.
  - Present this as missing manuscript detail, not a missing method component.

#### W9. Free-form Proposer questions vs fixed framing templates

- **Assessment:** easy to answer
- **Risk level:** low
- **Best response:**
  - Clarify that the free-form question string is inserted verbatim into each of the fixed Solver preambles.
  - The “fixed framing” refers to the instruction wrapper, not to templated question content.

#### W10. Why unlabeled images instead of self-generated images

- **Assessment:** reviewer concern is fair, but already answerable
- **Risk level:** low-to-medium
- **Best response:**
  - Point to the self-generated-images-only ablation.
  - Main message: self-generated images are not excluded in principle; rather, real-image diversity remains important, and generated-only training underperformed.

#### W11 / W12. Token-length handling and token-length distribution

- **Assessment:** answerable; minor extra evidence would help
- **Risk level:** medium
- **Best response:**
  - Explain that STE uses only the first few generated tokens and that answers are explicitly constrained to short, concrete outputs.
  - If possible, add a short answer-length distribution summary.

#### W13. Citation for adaptive Gaussian target

- **Assessment:** trivial fix
- **Risk level:** low
- **Best response:** add citation or rephrase as a simple adaptive heuristic if citation fit is weak.

### Reviewer 1 likely winning rebuttal posture

- Concede asymmetry.
- Defend shared-Solver coupling.
- Clarify misunderstood implementation details precisely.
- Avoid inflated novelty claims.

## Reviewer 2 Working Notes

### Overall assessment

This is a favorable review. The reviewer already sees the work as promising and is open to raising the score with clarification and limited extra evidence. This review should be treated as “convertible.”

### Point-by-point triage

#### Major: generation evaluation relies heavily on GenEval

- **Assessment:** valid
- **Risk level:** medium
- **Best response:**
  - If additional results on `DPG-Bench` or `WISE` exist, include them in the rebuttal.
  - If not, state that GenEval was used because it directly tests compositional instruction following under a matched setup across the three backbones, but acknowledge that broader perceptual evaluation is a limitation.

#### Major: STE max-token entropy is heuristic

- **Assessment:** same as Reviewer 1 W3
- **Risk level:** medium
- **Best response:**
  - Reuse the same answer as for Reviewer 1.
  - If a mean-vs-max ablation can be added, mention it here too.

#### Minor: limited qualitative failure analysis

- **Assessment:** fair but low priority
- **Risk level:** low
- **Best response:** acknowledge and point to supplementary qualitative cases if strong enough.

#### Minor: schedule effect / staged training

- **Assessment:** important despite being listed as minor
- **Risk level:** medium-to-high
- **Best response:**
  - This overlaps strongly with Reviewer 1 W2.
  - If you run only one extra experiment, make it this one.

#### Minor: training pool overlaps GQA/TextVQA, but evaluation reports only TextVQA and not GQA

- **Assessment:** valid and potentially sensitive
- **Risk level:** medium
- **Best response:**
  - Be explicit about split usage and absence/presence of image-level deduplication.
  - If any GQA evaluation already exists internally, report it.
  - If not, explain that the unlabeled pool is not used with labels and that evaluation is still on standard held-out splits, but avoid sounding dismissive.

### Reviewer 2 likely winning rebuttal posture

- Use this review to reinforce that the work is already close to accept.
- Address schedule and GenEval breadth directly.
- Keep the tone constructive and evidence-driven.

## Reviewer 3 Working Notes

### Overall assessment

This review is favorable on the core idea but worried about ceiling, efficiency, and stability. The good news is that these are easier to answer with careful framing than Reviewer 1’s novelty/coupling critique.

### Point-by-point triage

#### Major: scalability ceiling with frozen backbone and small data

- **Assessment:** valid conceptual question, but not a fatal flaw
- **Risk level:** medium
- **Best response:**
  - Do not claim unlimited scaling.
  - Position the method as a **sample-efficient post-training framework for extracting and reorganizing latent capability under a frozen backbone**.
  - State that the expected scaling regime is improvement up to a saturation point determined by backbone capacity and evaluator quality.
  - This is aligned with the paper’s actual scope and avoids a false promise.

#### Major: compute cost / inference multiplier

- **Assessment:** valid
- **Risk level:** medium
- **Best response:**
  - Provide actual wall-clock or GPU-hour numbers in the rebuttal.
  - If possible, break them down by backbone or by understanding vs generation phases.
  - Also emphasize that the method is post-training only and uses lightweight LoRA updates on a frozen backbone.

#### Major: generation ablations rely mostly on GenEval, and margins are modest

- **Assessment:** same theme as Reviewer 2
- **Risk level:** medium
- **Best response:** same as above; add broader generation benchmarks if available.

#### Minor: catastrophic full fine-tuning failure needs deeper explanation

- **Assessment:** fair and important
- **Risk level:** medium
- **Best response:**
  - Explain that full fine-tuning exposes the entire backbone to self-generated, internally noisy rewards, which increases policy drift and destabilizes both understanding and generation interfaces simultaneously.
  - Emphasize that the frozen-backbone + LoRA design is not a workaround added after failure; it is a core stability principle of the framework.
  - Frame full FT failure as evidence that unsupervised internal rewards require constrained update capacity.

#### Minor: missing recent unified-model citations

- **Assessment:** easy fix
- **Risk level:** low
- **Best response:** add the citations in the camera-ready or response if allowed.

### Reviewer 3 likely winning rebuttal posture

- Be concrete on compute.
- Be honest about scaling ceiling.
- Explain full fine-tuning failure mechanistically rather than vaguely attributing it to “policy drift.”

## Existing Evidence Already Available

These points appear to be answerable with existing manuscript/supplement/code evidence and should be used directly in the rebuttal.

- **STE window ablation already exists**
  - Supplementary sweep includes `W=64, 128, 256`.

- **Question framing implementation is explicit**
  - The proposer generates a free-form question string, and the Solver wraps it with fixed PPS preambles.

- **Reference answers for generation are solver-generated on the real image**
  - This directly answers Reviewer 1 W5.

- **Self-generated-images-only variant already underperforms**
  - This directly answers Reviewer 1 W10.

- **STE uses early-token entropy, not whole-answer alignment**
  - This helps answer Reviewer 1 W3/W11 and Reviewer 2’s heuristic concern.

- **Cycle-consistency backends and reward terms are implemented**
  - These answer Reviewer 1 W7/W8.

## Highest-Value Additional Experiments / Evidence

If time is limited, prioritize these in order:

1. **Two-stage baseline**
   - Train understanding first, then freeze proposer/solver and train generation.
   - This is the single best answer to Reviewer 1 W2 and Reviewer 2’s schedule concern.

2. **STE aggregation ablation**
   - `max` vs `mean` (and optionally top-2 mean).
   - This directly addresses Reviewers 1 and 2.

3. **Compute report**
   - Total GPU hours / wall-clock time for the 10k-step runs.
   - Direct answer to Reviewer 3.

4. **Answer-length distribution summary**
   - Small table or histogram.
   - Direct answer to Reviewer 1 W11/W12.

5. **Additional generation benchmark(s)**
   - `DPG-Bench`, `WISE`, or any already available matched benchmark.
   - Helps Reviewers 2 and 3.

## Rebuttal Messaging Rules

When writing the final rebuttal:

- Do **not** defend everything at maximum strength.
- Concede asymmetry where needed.
- Use precise implementation clarifications where the reviewer misunderstood the method.
- Keep novelty claims at the system level.
- Separate:
  - **clarification gaps**,
  - **wording overclaims**,
  - **actual algorithmic limitations**.

## Pending Items for Future Reviews

- Add remaining reviewer comments here under new sections.
- Track whether any later review reinforces:
  - coupling skepticism,
  - novelty skepticism,
  - evaluation breadth,
  - compute/scalability,
  - stability/full fine-tuning.

