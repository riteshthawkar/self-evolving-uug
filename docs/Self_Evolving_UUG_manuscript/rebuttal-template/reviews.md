Dear Authors of Submission #2837,

Thanks to the efforts of all reviewers, emergency reviewers, ACs, and SACs, preliminary reviews are now available. Below you will find the preliminary reviews for your ECCV 2026 submission, Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards (#2837).
Reviews

You can access the reviews for your submission by clicking on the corresponding submission in OpenReview: https://openreview.net/group?id=thecvf.com/ECCV/2026/Conference/Authors

A very small number of papers received two reviews. We are still chasing the remaining reviews and will keep the impacted authors posted.

A small number of papers received more than three reviews. If your paper is one of these, it is simply due to the Area Chair’s strenuous efforts in securing emergency reviewers and does not indicate anything special about your paper.

In a few exceptional cases, papers received five or more reviews. We'll reach out to those authors separately.

Rebuttal

If you have questions or concerns regarding the reviews, we encourage you to address them in your rebuttal, where they will be carefully considered by the reviewers and Area Chairs. To ensure a fair and consistent process, the rebuttal is the appropriate channel for such discussions. The Program Chairs will generally not respond to individual inquiries about specific reviews at this stage, except in cases involving conflicts of interest or ethical concerns.

If you wish to respond to the reviews, you may upload a one-page PDF document using the template (rebuttal.tex) that can be found in the ECCV Author Kit (https://www.overleaf.com/project/696680793124a7ce34e9739e). The entire rebuttal must fit on one page, including any references. We also removed the need for a title in the template to give you more space. To upload your rebuttal, click the “Rebuttal” button on the page of the paper, upload the PDF, and submit the form.

Rebuttals longer than one page will not be reviewed. The same goes for rebuttals, where the margins and formatting are deemed to have been altered from the template.

The rebuttal CANNOT include external links to videos, code repositories, etc (anonymous or otherwise).

The rebuttal must maintain anonymity.

The goal of the rebuttal is to refute any factual errors in reviews or to supply additional information or clarifications requested by the reviewers. Rebuttals may include minor additional experiments or analysis requested by reviewers. They may also include figures, graphs, or proofs to better illustrate your arguments. Rebuttals MUST NOT add new contributions (theorems, algorithms, experiments) that were absent in the original submission and were not specifically requested by the reviewers. We allow reviewers to request small experiments that could be reasonably run during the rebuttal phase with academic resources. If reviewers request experiments that require too many resources or are otherwise impossible to run, the authors should explain why they are not feasible. Authors should refrain from including significant new experimental results in the rebuttal not specifically requested by the reviewers.

Confidential Comment to AC

In addition to the PDF, the rebuttal submission form includes an optional box for authors to write a confidential comment to the Area Chair, to address any significant concerns related to policies, ethics, etc. Please do so only in exceptional circumstances. Note that overly lengthy or charged comments are unlikely to benefit the outcome of your paper.
Withdrawal

If you wish to withdraw your paper after reading the reviews, you may do so by pressing the “Withdraw” button on the paper’s page, confirming the withdrawal, and submitting the form.
Best,
Paolo Favaro, Zuzana Kukelova, Atsuto Maki, Anna Rohrbach, Konrad Schindler, Federico Tombari
ECCV 2026 Program Chairs

Reviewer BxGH

Title

Review of paper#2837

Choose The Contribution Type You Are Using For Your Review

Algorithms/General

Justify The Above Choice

I agree

Paper Summary

This paper proposes a fully self-supervised post-training framework for unified large multimodal models (LMMs). It aims to concurrently improve visual understanding and image generation without relying on external human annotations or reward models. The authors introduce a "Proposer-Solver-Generator" paradigm utilizing lightweight LoRA adapters on a frozen backbone. The framework consists of two coupled loops: an understanding loop driven by a novel Solver Token Entropy (STE) mechanism combined with sample-level self-consistency, and a generation loop where the Solver acts as an internal evaluator using QA fidelity and cycle-consistent captioning. The method demonstrates consistent improvements across three distinct generative architectures (diffusion, flow matching, and autoregressive) using a pool of 6,000 unlabeled images.

Paper Strengths

This paper introduces Solver Token Entropy (STE), which effectively addresses the issue of reward degeneracy by capturing token-level prediction uncertainty. This mechanism provides a reliable, continuous difficulty signal capable of distinguishing between genuine mastery and overconfident errors.
While many alignment methods are tailored to specific network topologies or generation mechanisms, this framework demonstrates broad applicability. Applying the exact same algorithmic recipe to three fundamentally different generative paradigms (Diffusion, Flow Matching, and Autoregressive) and achieving consistent performance gains is a solid empirical contribution.
The dual-role design of the Solver—acting as the learner in the understanding loop and the evaluator in the generation loop—is structurally elegant. The ablation studies provide convincing evidence that improvements in visual understanding directly translate into higher-quality supervision signals for generation, establishing an effective positive feedback loop.
Major Weaknesses

The authors emphasize that the method achieves notable gains using only 6,000 images over 10,000 training steps. While this highlights sample efficiency, it raises reasonable doubts about the framework's theoretical upper bound. As acknowledged in the conclusion, generation supervision is inherently bounded by the Solver's innate quality. Furthermore, because the backbone network is strictly frozen (with Table 6 indicating that full fine-tuning leads to performance degradation), the model is restricted from learning fundamentally new visual features. It remains questionable whether this method could sustain performance growth if scaled to a much larger dataset—for instance, if the unlabeled image pool were expanded to 1 million images, would the model's capabilities continue to improve, or would it simply hit a performance bottleneck? Ultimately, it is unclear if the framework can truly scale, or if it primarily acts to extract existing knowledge before quickly reaching a hard performance ceiling.
Although the dataset size is small, the inference multiplier per training sample is substantial. During each step, the Proposer generates multiple candidate questions, the Solver performs numerous forward passes under prompt perturbations to calculate consistency and STE, and the Generator synthesizes several images for subsequent internal evaluation. This high-frequency inference pipeline—especially the sampling required for diffusion and flow matching models—likely incurs a massive computational cost. The submission lacks specific details regarding GPU hours or wall-clock time, making it difficult to properly assess the practical efficiency of the proposed framework.
The generation ablation studies rely almost entirely on GenEval, and the absolute differences between variants are relatively marginal (often 1-2 points). Given that GenEval is based on external VQA models and can be subject to fluctuation, incorporating additional metrics (e.g., DPG-Bench and wise) would make the generation-side conclusions more robust.
Minor Weaknesses

Table 6 shows that full fine-tuning not only fails to improve upon the base model but actually degrades performance. While this is briefly attributed to policy drift, a deeper discussion on the mechanics of why catastrophic forgetting occurs so severely under this specific reward framework is needed. More importantly, the inability to stably perform full fine-tuning raises valid concerns about the intrinsic stability of this self-evolving paradigm. Considering that other post-training alignment methods routinely execute full fine-tuning without catastrophic collapse, this failure suggests that the internal, self-generated reward signals might be too brittle or noisy to safely guide a full-parameter update. Addressing whether this instability is a fundamental limitation of the proposed unsupervised reward formulation would significantly strengthen the paper's technical depth.
The manuscript lacks citations and brief discussions of several recent Unified Multimodal Models, such as JanusFlow, Show-o2, OmniMamba, X-Omni, LongCat-Next, and Liquid.
Preliminary Recommendation

4: Borderline Accept

Justification Of Preliminary Recommendation

In light of the Algorithms/General contribution type, I recommend a Borderline Accept. The paper presents a highly compelling core algorithmic contribution: the Solver Token Entropy (STE) mechanism elegantly resolves the issue of reward degeneracy. Furthermore, demonstrating its effectiveness across three diverse generative architectures is an impressive empirical achievement.

However, there are two primary concerns that currently temper my rating. First, the strict reliance on a frozen backbone and the small data scale (6,000 images) raise valid questions about the framework's scalability and theoretical capacity ceiling. It remains unclear if the method can genuinely scale to much larger datasets or if it quickly hits a performance bottleneck. Second, the catastrophic failure observed during full fine-tuning (Table 6) suggests potential instability in the self-generated reward paradigm, especially when compared to standard alignment methods that routinely support full-parameter updates.

Overall, the core idea is very promising. Providing a convincing discussion and addressing these two specific concerns regarding scalability and paradigm stability during the rebuttal would be highly valuable, and is critical for me to justify a stronger recommendation.

Suggestions For Rebuttal

To secure a stronger recommendation, please explicitly address the following concerns in your rebuttal:

Discuss the framework's scaling behavior if the unlabeled data pool is significantly expanded (e.g., to 1M images). Address the concern of whether the model quickly hits a hard performance bottleneck due to the frozen backbone.

Clarify the underlying mechanics behind the catastrophic failure during full fine-tuning (Table 6) to alleviate concerns about the self-generated reward signals being too noisy or brittle for deep feature learning.

Briefly report the total training cost (e.g., total GPU hours) required for the 10,000 steps.

Ethics Review Flag

No

Confidence Level

3: Moderate Confidence - The reviewer is reasonably knowledgeable about the topic. They understand the paper's methodology and results but may not be a leading expert in the specific subfield.

Reviewer 2eb3

Title

Review

Choose The Contribution Type You Are Using For Your Review

Algorithms/General

Justify The Above Choice

I agree.

Paper Summary

This paper proposes a self-evolving training framework for unified MLLMs that improves both visual understanding and image generation without any external supervision. It proposes to decompose the frozen backbone into three collaborative roles implemented via lightweight adapters, i.e., Proposer, Solver, and Generator. The model learns entirely from unlabeled images through internal self-consistency signals: the Proposer generates questions, the Solver answers and evaluates them under perturbations, and the Generator synthesizes images, forming a closed feedback loop that couples understanding and generation. To address reward degeneracy, the authors introduce Solver Token Entropy (STE) as a token-level uncertainty signal, and further design a multi-scale generation evaluation combining QA fidelity and cycle-consistent captioning. The framework is architecture-agnostic and demonstrates consistent improvements across diffusion, flow matching, and autoregressive models, suggesting a scalable and general alternative to supervision-heavy post-training.

Paper Strengths

The paper presents a meaningful step toward fully self-supervised post-training for unified multimodal models. Unlike prior work that either relies on external rewards or is restricted to specific architectures, this method offers a general, architecture-agnostic framework validated across three distinct generation paradigms. The introduction of a three-role decomposition (Proposer–Solver–Generator) is conceptually clean and aligns well with recent trends in self-play and role-based learning.

The paper introduces Solver Token Entropy (STE) to address reward degeneracy in self-consistency learning, enabling a stable difficulty signal and effective curriculum without supervision.

The paper achieves consistent improvements across multiple benchmarks and backbones, with thorough ablations validating the contribution of each component.

Major Weaknesses

The generation evaluation relies heavily on GenEval, which may not fully capture perceptual quality. Adding more text-to-image benchmarks would help strengthen the effectiveness of the proposed method.

The proposed STE formulation (max token entropy) is somewhat heuristic. Authors can provide more motivations about this design choice. Also, alternative formulations (e.g., averaged entropy or calibration-based uncertainty) are not explored.

Minor Weaknesses

The paper displays limited qualitative analysis of failure cases.
The generation loop critically relies on the Solver as an internal evaluator. However, the paper does not study whether training schedules affect performance, e.g., pretraining the understanding loop before enabling generation. Since the Solver is initially weak, early-stage generation rewards may be noisy or biased. It would be valuable to compare joint training with staged training (understanding-first) to better understand stability and optimization dynamics.
The training data includes images from GQA and TextVQA, but the evaluation only reports results on TextVQA and omits GQA. This raises questions about in-domain performance and potential distribution bias. It would be helpful to understand how the model performs on datasets that overlap with the training pool, and whether improvements are driven by generalization or by reinforcing dataset-specific patterns. Additionally, an ablation on different data sources (e.g., excluding GQA or TextVQA) would provide insight into how data composition affects the self-evolving process.
Preliminary Recommendation

4: Borderline Accept

Justification Of Preliminary Recommendation

The paper presents a promising fully self-supervised framework for unified multimodal post-training, with clear empirical gains across diverse architectures and a technically interesting design. That said, due to several remaining questions around specific design choices and evaluation aspects, I am currently leaning borderline accept.

Suggestions For Rebuttal

Please refer to the weakness part. If the authors can provide more convincing explanations and additional evidence in the rebuttal, I would be willing to raise my score.

Ethics Review Flag

No

Confidence Level

3: Moderate Confidence - The reviewer is reasonably knowledgeable about the topic. They understand the paper's methodology and results but may not be a leading expert in the specific subfield.

Reviewer ksp6

Title

Official review

Choose The Contribution Type You Are Using For Your Review

Algorithms/General

Justify The Above Choice

I agree

Paper Summary

This work proposes a self-evolving training framework for unified multimodal models that jointly improves visual understanding and image generation using only unlabeled images. The framework decomposes a frozen backbone into three LoRA-based roles: a Proposer that generates visual questions, a Solver that answers and evaluates them, and a Generator that synthesizes images. Training is driven entirely by internal consistency signals without external supervision: for understanding, the method introduces Solver Token Entropy (STE), a token-level uncertainty measure that provides continuous difficulty signals when sample-level self-consistency saturates; for generation, a multi-scale internal evaluation scheme combines QA fidelity scoring with cycle-consistent captioning, using the Solver as an internal evaluator to couple understanding quality with generation optimization. The framework is architecture-agnostic and is validated across three unified model paradigms, diffusion-based (BLIP3o), flow-matching (BAGEL), and autoregressive (VARGPT-v1.1), with consistent improvements reported on eight benchmarks for both understanding and generation tasks.

Paper Strengths

Solver Token Entropy (STE): The refinement from sample-level continuous rewards (as in EvoLMM) to token-level prediction uncertainty is a reasonable design choice. It provides a more granular training signal that remains informative even when sample-level agreement saturates.

Multi-scale generation evaluation: Combining local QA fidelity with global cycle-consistent captioning is a reasonable engineering choice to capture both attribute-level and scene-level generation quality.

Architecture-agnostic design: Validating a single training recipe across three fundamentally different generation paradigms (BLIP3o, BAGEL, VARGPT) is commendable and demonstrates practical portability.

Comprehensive empirical evaluation: The paper reports results on eight benchmarks with detailed ablations, loop coupling analysis, and training dynamics, providing substantial empirical support.

Major Weaknesses

W1. Limited novelty.

The proposed framework largely synthesizes and refines three established research directions: role-based self-play for VQA (e.g., EvoLMM), text-to-image generation (e.g., UniCorn), and bidirectional understanding–generation co-training (e.g., UniRL). While the engineering integration is coherent, the methodological novelty is incremental: (1) Solver Token Entropy (STE) builds directly on EvoLMM’s continuous self-consistency reward, while refining the signal granularity from sample-level probability to token-level probability. (2) The proposed multi-scale scheme, combining QA fidelity scoring with cycle-consistent captioning, appears to mainly merge two components already present in UniCorn: its QA-based evaluation (UniCycle) and its score-based (10-point scale) judging framework. While this combination may be reasonable, the paper does not explain the motivation behind this specific design.

W2. Weak coupling justification and ungrounded bidirectional assumption.

The claimed “positive feedback loop” is not supported by the training design. In practice, optimization is largely unidirectional: Proposer/Solver are trained only via understanding signals (e.g., self-consistency), while generation rewards update only the Generator. There is no explicit mechanism feeding generation quality back into understanding. Thus, the assumed bidirectional synergy is ungrounded, and the interaction is effectively asymmetric (understanding → generation, but not vice versa). Given this, a simpler two-stage pipeline (understanding stage as in EvoLMM + generation stage as in UniCorn) would likely achieve similar results with greater stability. This further weakens the claimed novelty, as the framework reduces to a loose combination of existing approaches.

W3. Max-over-tokens heuristic:

Taking the maximum entropy across decoding steps (Eq. 4) may overemphasize noisy or uninformative tokens (e.g., function words like "the"). A weighted average or attention-based aggregation might be more robust.

W4. Rolling window normalization:

The quantile-based normalization of STE assumes entropy values are comparable across diverse images and questions. This assumption is not validated, and the choice of window size (128 samples) lacks ablation support.

W5. Reference answer quality in Multi-scale Generation Evaluation:

QA fidelity scoring (Eq. 6) relies on reference answers from the Proposer, yet the Proposer is not optimized for this objective. The paper does not address how reference answer quality is maintained or how errors propagate.

W6. Prompt-dependence of token distributions:

Solver Token Entropy (STE) computes next-token entropy under different prompt framings. However, even semantically equivalent prompts can induce different prefix contexts, leading to slightly divergent token distributions. Requiring consistent next-token probabilities across framings may impose an overly strict constraint.

W7. Cycle consistency implementation:

The paper mentions using "frozen text embedding f(·)" and "frozen vision–language similarity sim_vl" but does not specify which models are used. This omission hinders reproducibility and makes it difficult to assess whether the signal is truly architecture-agnostic.

W8. Diversity/contradiction rewards:

S_div and S_ctr are mentioned in Eq. 8 but not defined. Their computation and weighting significantly impact generation optimization and should be clarified.

W9. Prompt Framing for Self-Consistency:

Line 186 describes querying the Solver under N fixed prompt framings. However, the Proposer generates free-form questions, not template-based queries. The paper does not explain how model-generated questions are mapped to fixed framing templates, raising concerns about the feasibility and consistency of this procedure.

W10. Motivational gap: use of unlabeled vs. generated images.

The paper emphasizes training on unlabeled images but does not justify excluding self-generated images from the Generator. Prior self-evolving frameworks already construct training signals purely from internal outputs and consistency without external supervision. If the Generator can produce diverse, high-quality samples, incorporating them could further enrich the training distribution. The rationale for restricting training to only external unlabeled images remains unclear.

W11. Token-length handling.

Semantically equivalent answers may vary in length. Since token-level entropy is computed at each generation step from next-token probability distributions aggregated across all answers, it is unclear how variability in answer length is properly aligned or normalized.

Minor Weaknesses

W12. Token length distribution:

Since STE is token-level, reporting the distribution of answer token lengths across benchmarks would help assess whether the token-level design is necessary.

W13. Citation needed:

The "Gaussian centered at a running EMA target µ" (line 208) should cite prior work using similar adaptive targeting mechanisms.

Preliminary Recommendation

3: Borderline Reject

Justification Of Preliminary Recommendation

The paper is positioned as an algorithmic contribution, but the overall framework is highly similar to a synthesis of existing approaches such as EvoLMM and UniCorn. Conceptually, it can be viewed as combining EvoLMM-style understanding training with UniCorn-style generation supervision in a two-stage pipeline, with the primary difference being that the two components are coupled into a loop. However, since the interaction between them is weak and largely unidirectional, this coupling does not introduce a fundamentally new capability beyond prior work, but rather serves as a structural integration of existing paradigms. It refine EvoLMM's sample-level entropy to token-level entropy and adapt UniCorn's multi-scale evaluation.

From a technical soundness perspective, several key design choices lack sufficient justification, particularly the STE formulation and the claimed bidirectional coupling, which appears effectively unidirectional in practice. From an empirical standpoint, although the evaluation is comprehensive, the reported gains from joint optimization are modest (~0.3–1.0 points), and do not clearly justify the added complexity over simpler staged alternatives.

Suggestions For Rebuttal

Explicitly compare the proposed framework with a simple combination of EvoLMM and UniCorn, both of which already provide self-evolving understanding and self-play-based generation, and clarify what additional capability is gained. The paper should justify why the proposed refinements (e.g., STE, multi-scale evaluation) are necessary rather than optional extensions.

Loop coupling ablation: Add an experiment comparing the full joint training against a two-stage pipeline (first evolve Proposer/Solver, then freeze them to train Generator).

STE design validation:
Ablate the max-over-tokens heuristic versus average or attention-weighted aggregation.

Report results with different rolling window sizes (e.g., 64, 256) to demonstrate robustness.

Implementation details: Specify the frozen embedding/similarity models used for cycle consistency, and define S_div/S_ctr computation. Clarify how free-form Proposer questions are mapped to fixed framing templates.

Token length analysis: Report the distribution of answer lengths and discuss how length variation is handled in STE computation.

Ethics Review Flag

No

Confidence Level

5: Expert - The reviewer is an authority in the specific subfield addressed by the paper. They are extremely confident in their evaluation and understanding of the work.

Please note that responding to this email will direct your reply to cvpr_2026_pcs@computer.org.