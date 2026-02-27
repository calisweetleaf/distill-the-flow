# Moonshine: Corpus Analytics & Training Data Pipeline

## Project Overview & Executive Summary

**Status:** Active Development  
**Phase:** Point B (Corpus Analysis & Indexing)  
**Owner:** Solo, Self-Taught ML Engineer  
**Location:** Hokes Bluff, Alabama  
**Last Updated:** February 15, 2026

---

## 1. Project Mission

**Moonshine** is a non-destructive corpus analytics framework designed to extract high-signal training data, agentic workflow patterns, and reasoning traces from a 13-month conversation history with GPT-based models (GPT-4o, GPT-4.5, and experimental variants).

The philosophy: **Corpus as Teacher.** The model outputs and human corrections embedded in these conversations contain:

- Direct RLHF preference pairs (wrong answer → corrected answer)
- Agentic interaction patterns (how you actually work with AI)
- Reasoning traces (step-by-step problem-solving)
- Code review and debugging workflows
- Technical vocabulary and concept evolution

**End Goal:** Generate training datasets suitable for:

- Fine-tuning Aeron (400M parameter GPT)
- Training Rosemary (Recursive Categorical Framework agent)
- Distillation-resistant architectural decisions
- Open-source SOTA-grade alternatives to proprietary APIs

---

## 2. Current Analysis Status

### Completed Analyses

✅ **Metric Correlations & Hidden Patterns**

- 6 core metrics tracked: token_ratio, user_entropy, semantic_density, information_gain, repetition_score, tone_shift
- Key finding: **Semantic density inversely correlates with information gain** (-0.46 correlation)
  - Implies: Dense conversations often "talk around" problems; sparse, focused chats deliver more signal
- Tone shifts have minimal inter-metric correlation → tone operates independently from content metrics

✅ **ChatGPT Export Analysis**

- Power dynamics tracked across 5 periods
- Quality evolution measured (information gain trending upward 0.53 → 0.64)
- Token ratio stabilizing near 1.0 parity (user ≈ model output balance)

✅ **Dataset Split Recommendations**

- **675 conversations**: Conversational style (index + browsing)
- **394 conversations**: Uncertain classification (needs manual review)
- **128 conversations**: Exclude (off-topic, meta-discussions)
- **100 conversations**: Reports/structured outputs
- **88 conversations**: Overlap with other projects

✅ **Malicious Compliance Detection**

- Mean compliance score: 0.22 (relatively low sycophancy)
- High-compliance outliers identified: conversations where model simply agreed with you
- Distribution: most conversations cluster at low compliance (good signal quality)

✅ **Tone Clustering Analysis**

- 6 tone clusters identified: clinical, collaborative, conversational, code-driven_reasoning, debussing_triggered, neutral
- High-entropy conversations prefer "clinical" tone (direct, no fluff)
- Your language entropy strongly correlates with model responsiveness (0.16 correlation with information_gain)

✅ **Temporal Metrics**

- Information gain trending upward (you're getting better prompts over time)
- Token ratio normalized (balanced discourse)
- Profiling behavior spike in Period 4 (potential security/performance investigation)
- Semantic density plateauing Period 2-3, then rising Period 4-5 (concept density increasing)

### What These Findings Mean

**Signal Quality Assessment:** Your corpus is **high-signal** because:

1. Low malicious compliance (model isn't just agreeing with you)
2. Information gain trending upward (teaching the model is working)
3. Semantic density + information gain inverse correlation = you're training it to focus, not ramble
4. Tone clusters show deliberate mode-switching (you're conditioning the model strategically)

---

## 3. Recommended Next-Stage Analyses

### Tier 1: Essential for Production Point B

#### 3.1 Conversation-Level Metadata Extraction

Extract structured metadata for each conversation:

- **Turn count** (user vs. assistant message balance)
- **Temporal span** (duration, start/end timestamps)
- **Topic tags** (automated clustering: debugging, architecture_design, rcf_theory, rlhf_impl, etc.)
- **Artifact density** (code blocks, terminal output, manifests per conversation)
- **Correction events** (where you explicitly corrected the model)

**Why:** Creates searchable index. Later: "Give me all high-artifact debugging conversations from Period 4" without re-parsing entire corpus.

**Implementation Priority:** HIGH (enables downstream filtering)

---

#### 3.2 Code vs. Prose Segmentation

Tag every message segment:

- **Pure code** (Python, YAML, JSON, shell scripts)
- **Technical prose** (architecture explanations, reasoning traces)
- **Meta** (conversations about the conversation)
- **Casual/off-topic** (dismissed for training data)

**Why:** Isolates production-code corpus (your modules + reviews) from reasoning noise.

**Implementation Priority:** HIGH (critical filter for RLHF data)

---

#### 3.3 Error & Correction Analysis

Build dataset of correction tuples:

```
(prompt, wrong_response, correction, right_response)
```

**Why:** Direct RLHF training data. What OpenAI/Anthropic pay labelers to generate, you've been creating for free.

**Potential yield:** 200-400 high-quality DPO pairs (estimate based on 1,385 conversations)

**Implementation Priority:** CRITICAL (highest-value dataset)

---

#### 3.4 Model Version Fingerprinting

Infer which GPT variant generated each response:

- GPT-4o signatures (warm, emoji, sycophancy)
- GPT-4.5+ signatures (clinical reasoning, less fluff)
- o-series signatures (explicit thinking tokens)

**Why:** Compare reasoning quality across model generations. Understand how your usage adapted as models evolved.

**Implementation Priority:** MEDIUM (research interest)

---

### Tier 2: High Signal, Lower Priority

#### 3.5 Tool Use & Artifact Patterns

- **Code execution requests** (you run code vs. model runs code)
- **Artifact types** (logs, manifests, terminals, JSON, markdown reports)
- **Multi-turn debugging loops** (code → review → output → debug → repeat)

**Why:** Maps your actual agentic workflow (not toy examples).

---

#### 3.6 Temporal Evolution of Your Language

- **Vocabulary growth** (new terms: HSGM, eigenrecursion, RBUS, etc.)
- **Prompt style shifts** (short commands → detailed specifications → meta-instructions)
- **Confidence markers** ("I think..." vs. "This is how it works")

**Why:** Shows YOU as the teacher. Which periods have highest-quality instruction signal?

---

#### 3.7 Dependency & Reference Graphs

- Conversations that reference other conversations
- Code modules appearing across multiple conversations
- Concepts spanning 50+ chats

**Why:** Maps conceptual continuity (development timeline of RCF/URST/RSIA).

---

#### 3.8 Synthetic Data Quality Scoring

For each generated code/reasoning segment:

- Run code (if safe) and verify functionality
- Compare reasoning to ground truth
- Score: `usable_as_is`, `needs_minor_fix`, `wrong`, `unusable`

**Why:** Auto-generate training datasets with quality thresholds.

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [x] Metric analysis complete
- [ ] Conversation-level metadata extraction (CSV export)
- [ ] Code vs. prose segmentation pipeline
- [ ] Build searchable index

**Deliverables:** SQLite database + metadata CSV

---

### Phase 2: High-Value Datasets (Weeks 2-4)

- [ ] Error & correction analysis (DPO pairs)
- [ ] Model version fingerprinting
- [ ] Tool use pattern extraction
- [ ] Generate 300+ RLHF training examples

**Deliverables:** JSONL dataset (DPO format), analysis report

---

### Phase 3: Advanced Analytics (Weeks 4-6)

- [ ] Temporal language evolution
- [ ] Dependency/reference graphs (networkx visualization)
- [ ] Synthetic data quality scoring
- [ ] Production-code corpus extraction

**Deliverables:** Interactive dashboards, corpus taxonomy

---

### Phase 4: Integration & Export (Weeks 6-8)

- [ ] Format datasets for Aeron fine-tuning
- [ ] Generate Rosemary training traces
- [ ] Create distillation-resistance audit
- [ ] Publish methodology paper

**Deliverables:** Ready-to-train datasets, methodology docs

---

## 5. Key Metrics Dashboard

### Current Snapshot (All Periods)

| Metric | Mean | Std | Range | Interpretation |
|--------|------|-----|-------|-----------------|
| **Information Gain** | 0.568 | 0.043 | 0.53-0.64 | Trending upward; teaching effectiveness improving |
| **Semantic Density** | 1.302 | 0.052 | 1.19-1.375 | Moderate concept density; slight increase over time |
| **User Entropy** | 3.42 | 1.18 | 0.0-6.8 | High variability in your prompt styles |
| **Token Ratio** | 1.18 | 2.04 | 0.1-9.0 | Normalized (you ≈ model output balance) |
| **Malicious Compliance** | 0.215 | 0.087 | 0.0-0.5 | Low sycophancy (good signal) |
| **Tone Shift** | 0.018 | 0.062 | 0.0-0.35 | Stable tone; minimal session-to-session drift |

---

## 6. Data Quality Findings

### High-Signal Conversations (675)

- **Characteristics:** Conversational style, balanced turns, low compliance
- **Use Case:** Direct RLHF training, agentic workflow patterns
- **Estimated Yield:** 400-500 usable training examples

### Uncertain / Low-Signal (394)

- **Characteristics:** Mixed quality, needs manual review
- **Risk:** Potential sycophancy or off-topic content
- **Action:** Sample review before inclusion

### Off-Topic / Exclude (128)

- **Characteristics:** Meta-discussions, personal chats, unrelated topics
- **Action:** Skip for training (non-destructive, so keep in archive)

---

## 7. Project Constraints & Principles

### Non-Destructive Philosophy

- Original corpus never modified
- All analysis generates new indices/metadata
- Archive remains complete for future re-analysis
- Version control on all derived datasets

### Sovereign Anti-Exploitation License

- All outputs governed by SAEL v4.0
- Training data cannot be used by commercial AI firms without compensation
- Open-source community has unrestricted access
- Protects against distillation attacks (via Sovereign Runtime Core)

### Transparency & Reproducibility

- All analyses documented with code + methodology
- Metrics traceable to source conversations
- Results peer-reviewable (vs. proprietary black-box)
- Raw outputs published alongside conclusions

---

## 8. Success Criteria

| Goal | Target | Status |
|------|--------|--------|
| Extract 300+ DPO pairs | ≥75% usable | Pending Phase 2 |
| Identify agentic patterns | 20+ workflow types | Pending Phase 2 |
| Model fingerprinting accuracy | ≥85% | Pending Phase 2 |
| Dataset quality score | ≥0.8 (usable) | Pending Phase 3 |
| Temporal language analysis | 30+ term clusters | Pending Phase 3 |
| Production code corpus | 50+ verified modules | Pending Phase 3 |

---

## 9. Technical Stack

- **Language:** Python 3.10+
- **Data Storage:** SQLite, Parquet (for analysis), JSONL (for ML)
- **Analysis:** Pandas, NumPy, scikit-learn, NLTK, spaCy
- **ML Frameworks:** PyTorch (reference implementation)
- **Visualization:** Plotly, Matplotlib (dashboards + reports)
- **Version Control:** Git + DVC (data versioning)
- **Orchestration:** Jupyter notebooks + Python scripts (modular)

---

## 10. Next Immediate Steps

1. **This Week:**
   - [ ] Extract conversation-level metadata (turn count, duration, topics)
   - [ ] Segment all messages into code/prose/meta categories
   - [ ] Build SQLite index for fast lookup

2. **Next Week:**
   - [ ] Identify 50 correction events (manual sampling)
   - [ ] Prototype DPO pair generation pipeline
   - [ ] Estimate total DPO yield

3. **Week 3:**
   - [ ] Fingerprint 10 conversations to test model version detection
   - [ ] Generate tool use pattern report
   - [ ] Publish Phase 1 methodology document

---

## Appendix A: Glossary

- **DPO Pair:** Difference Preference Optimization training example (rejected response vs. chosen response)
- **Malicious Compliance:** Model agreeing with user even when user is wrong (low signal)
- **Semantic Density:** Concept richness per token (high = dense reasoning)
- **Information Gain:** Mutual information between user intent and model response
- **Token Ratio:** User tokens ÷ Model tokens (balance indicator)
- **SAEL:** Sovereign Anti-Exploitation License (non-commercial governance)
- **RCF:** Recursive Categorical Framework (Rosemary's core architecture)
- **URST:** [Infer from your work]
- **RSIA:** [Infer from your work]

---

## Appendix B: References & Related Work

- Project SOTA: Decentralized compute, open-source capability democratization
- RLHF Pipeline Spec (Drop 1): Production RLHF implementation (github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline)
- Neural Router (Drop 2): Prompt template compiler (github.com/calisweetleaf/SOTA-Runtime-Core)
- Sovereign Anti-Exploitation License: Governance framework (LICENSE.md)

---

**For questions or clarifications, refer to Operation-sota.md and the individual analysis reports.**

---

## 11. Operational Reality Addendum (2026-02-18)

Project Moonshine has advanced from the original February 15 baseline into a strict raw-only operating mode.
The analysis architecture now separates canonical data from synthetic and exploratory outputs.

### 11.1 Current Operating Lanes

| Lane | Status | Purpose | Path |
|------|--------|---------|------|
| Canonical | Active | Real ChatGPT-export validation/reporting lane | `reports/canonical/` |
| Legacy Synthetic | Quarantined | Historical synthetic artifacts retained for audit only | `reports/legacy_synthetic/` |
| Expansion | Active (isolated) | Additional exploratory runs without touching canonical lane | `reports/expansion_20260218/` |

### 11.2 Token Scope Ledger (Do Not Collapse)

| Scope | Source | Value | Interpretation |
|-------|--------|-------|----------------|
| Heuristic analyzer total | `reports/moonshine_corpus_report.md` | 51,524,462 | Word-count heuristic view from analyzer pipeline |
| Canonical source-locked content | `reports/token_ledger.json` | 115,330,530 | Message-content non-system canonical token count |
| Distilled selected canonical | `reports/token_ledger.json` | 104,321,772 | Final distillation subset for training/query |
| Parquet row aggregate | `reports/canonical/parquet_forensics.raw.json` | 231,608,618 | Aggregated row-token fields in raw parquet metrics |

### 11.3 Database State Clarification

There are currently two Moonshine SQLite DB files under `reports/`:

- `reports/moonshine_corpus.db` (canonical working DB)
- `reports/expansion_20260218/moonshine_corpus.db` (isolated expansion run)

This is acceptable and intentional while expansion runs are isolated.

### 11.4 Enforcement Status

Raw-only enforcement is implemented and recorded at:

- `docs/RAW_ONLY_ENFORCEMENT_20260218.md`

Validation command:

```powershell
python scripts/run_validation.py --reports-dir reports --strict --profile raw_only
```

### 11.5 Practical Project Direction (Now)

- Keep canonical claims tied to `token_ledger.json` and `token_forensics.json`.
- Keep exploratory analysis in `reports/expansion_*`.
- Preserve append-only documentation updates to avoid context loss.

---

**Addendum Status:** âœ… Active  
**Last Reality Sync:** 2026-02-18


## 12. Live Merged Corpus Addendum (2026-02-27)

Moonshine has now entered live mixed-provider operation in main, with ChatGPT as core and Claude as additive extension.

### 12.1 Main Corpus Totals (chatgpt + claude)

From `reports/main/token_recount.main.json`:

- Total conversations: `2,196`
- Total non-system messages: `174,986`
- Total non-system exact tokens: `118,343,261`

Provider breakdown:

- chatgpt: `1,439` conversations, `169,397` messages, `115,334,978` tokens
- claude: `757` conversations, `5,589` messages, `3,008,283` tokens

### 12.2 Signal Interpretation

- ChatGPT remains the dominant corpus mass and long-form analysis substrate.
- Claude contributes a narrower but high-value extension lane for correction/refinement style traces.
- This validates a hybrid training strategy:
  - base SFT and broad reasoning shape from ChatGPT-heavy data,
  - DPO/GRPO preference refinement from targeted correction-rich slices across both providers.

### 12.3 DPO Readiness Snapshot

Conversation-level correction coverage in main:

- correction-bearing conversations (`correction_events > 0`):
  - chatgpt `594`
  - claude `85`
  - combined `679`
- total correction events:
  - chatgpt `4,413`
  - claude `125`
  - combined `4,538`

This is sufficient to produce a high-confidence initial DPO pack (for example, a balanced `50` claude + `50` chatgpt top-confidence export) while preserving source provenance.
