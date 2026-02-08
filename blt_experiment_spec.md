# Patch Granularity and Reasoning Dynamics in Byte Latent Transformers

## Research Question

How does patch-level tokenization granularity in Byte Latent Transformers affect the mechanistic structure of reasoning, and do patches serve analogous functional roles to token-level "thought anchors"?

---

## Hypotheses

**H1 (Non-Monotonicity):** Model performance exhibits a non-monotonic relationship with patch granularity, with distinct failure modes at extremes:

| Regime | Failure Mode | Observable Signature |
|--------|--------------|---------------------|
| Too fine (low threshold) | Goal drift—model loses track of problem structure | Diffuse attention; reduced causal influence of problem-defining patches (activation patching); errors skew toward "correct operations, wrong problem" |
| Too coarse (high threshold) | Detail loss—model loses critical information | Over-concentrated attention on patch centroids; high causal influence from aliased/incorrect patches; errors skew toward arithmetic/factual mistakes |

**H2 (Task-Dependent Optima):** Optimal granularity threshold varies by reasoning type—mathematical reasoning requires finer granularity than factual retrieval, yielding divergent accuracy×threshold curves.

**H3 (Anchor Conservation):** Patches function as computational waypoints analogous to token-level thought anchors (per Nanda et al.), but anchor efficacy degrades predictably with granularity deviation from task-optimal threshold.

---

## Experimental Design

### Independent Variable
- **Entropy threshold** for patch boundary decisions
- Sweep: [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0] (8 levels; refined based on pilot)
- Models: `facebook/blt-1b`, `facebook/blt-7b`

### Dependent Variables

| Variable | Operationalization |
|----------|-------------------|
| Task accuracy | Exact match (ARC), calibrated accuracy (MMLU-Pro) |
| Attention distribution | Entropy of attention weights; patch-to-patch attention matrices aggregated by reasoning type |
| Causal influence | Activation patching on problem-defining vs. reasoning-step patches; measured as logit difference on correct answer |
| Error taxonomy | Classified as: structural (wrong problem), detail (wrong execution), incoherence (no valid reasoning) |
| Cross-type attention | Attention flow between sentence types following Thought Anchors categories: Problem Setup, Reasoning Steps, Intermediate Conclusions, Final Answer |

### Attention Attribution Method
**Uniform byte attribution (acknowledged limitation).** For patch P spanning bytes B₁...Bₙ with attention weight A, assign A/n per byte. Aggregation to word/sentence level via summation over constituent bytes.

*Justification:* Enables cross-threshold comparison with consistent methodology. Acknowledged bias: assumes uniform intra-patch salience. Future work: entropy-weighted or gradient-based attribution.

---

## Datasets & Sample Sizes

| Dataset | Purpose | N per threshold | Trials |
|---------|---------|-----------------|--------|
| ARC (Challenge) | Reasoning-heavy evaluation | 500 | 3 |
| MMLU-Pro | Capability baseline + breadth | 500 | 3 |

**Total forward passes:** 500 × 8 thresholds × 2 models × 2 datasets × 3 trials = 48,000

*Power note:* 500 samples provides 80% power to detect 6% accuracy difference at α=0.05. Will evaluate error bars after initial trial; scale to full test sets if variance demands.

---

## Reasoning Type Labeling

Following Thought Anchors methodology:
- LLM judge (GPT-4 or Claude) with fixed rubric classifies each sentence into: Problem Restatement, Reasoning Step, Intermediate Conclusion, Final Answer, Filler/Transition
- Validation: 50 examples hand-labeled; require >85% human-LLM agreement before proceeding
- Inter-rater: Run judge twice (T=0.3); require >90% self-agreement

---

## Analysis Plan

### Primary Analyses
1. **Accuracy × Threshold curves** by model and dataset—test for non-monotonicity via quadratic fit comparison to linear
2. **Error type distribution shift** across threshold regimes—χ² test for distributional differences at extremes vs. midpoint
3. **Activation patching** on problem-setup patches—compare causal influence (logit diff) across thresholds

### Secondary/Exploratory
4. Cross-type attention flow matrices—qualitative comparison to Thought Anchors findings
5. Optimal threshold estimation per reasoning category (math vs. factual vs. scientific)
6. Scale effects: 1B vs 7B divergence patterns

---

## Primary Contribution Claim

We provide the first mechanistic analysis of how dynamic tokenization granularity in Byte Latent Transformers modulates reasoning structure, demonstrating (1) non-monotonic performance with distinct, diagnosable failure modes, and (2) task-dependent optimal granularity with implications for adaptive tokenization strategies.

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| BLT threshold doesn't produce meaningful variation | Pilot with extreme thresholds first; abort if <5% accuracy spread |
| Uniform byte attribution introduces systematic bias | Acknowledge; run sensitivity check with entropy-weighted attribution on subset |
| LLM judge unreliable | Validate before main experiment; fall back to manual labeling if needed |
| Effects don't replicate across model scales | Report as finding (scale-dependent effects are interesting) |
