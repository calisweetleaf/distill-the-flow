# PROJECT MOONSHINE — UPDATE 1
## Corpus Intelligence Report: February 2026

**Classification:** Internal — Distribution: Project Team  
**Version:** 1.0.0  
**Date:** 2026-02-15  
**Status:** Phase 1 Complete — Analysis Pipeline Operational

---

## 1. EXECUTIVE SUMMARY

Project Moonshine is a corpus distillation system designed to transform raw conversational exports (the "mash") into structured, analyzable training datasets (the "moonshine"). This report documents the completion of Phase 1: the implementation of core analysis infrastructure and the first comprehensive analysis of 1,439 ChatGPT conversations.

### 1.1 Key Achievements

- **Pipeline Stabilization:** Fixed critical race conditions and data integrity issues in the token forensics orchestrator
- **Moonshine Analyzer:** Deployed standalone corpus analysis engine with 28 metrics per conversation
- **First Distillation:** Processed 1,439 conversations (51.5M tokens) into queryable SQLite database
- **Quality Baseline:** Established measurement framework for information gain, sycophancy detection, and correction events

### 1.2 Strategic Context

This update represents the foundation layer for a multi-phase corpus intelligence operation. Future phases will integrate additional export sources (Claude, GitHub, etc.), implement temporal analysis, and develop visual intelligence dashboards for longitudinal corpus evolution tracking.

---

## 2. SYSTEM ARCHITECTURE

### 2.1 The Mash-to-Moonshine Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW MASH SOURCES                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ ChatGPT      │  │ Claude       │  │ GitHub       │           │
│  │ Export       │  │ Export       │  │ Commits      │           │
│  │ (1,439 conv) │  │ (Pending)    │  │ (Pending)    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                              │
│  • Format normalization (JSON tree → flat message list)         │
│  • Deduplication (SHA256 hashing)                               │
│  • Temporal sorting (create_time-based bucketing)               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MOONSHINE ANALYZER ENGINE                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ METRICS EXTRACTION (per conversation)                   │    │
│  │ • Token metrics (ratio, entropy, density)               │    │
│  │ • Quality scores (info gain, repetition, sycophancy)    │    │
│  │ • Topic classification (keyword-based heuristics)       │    │
│  │ • Tone clustering (clinical, debugging, collaborative)  │    │
│  │ • Artifact detection (code blocks, terminals, tables)   │    │
│  │ • Correction events (DPO pair candidates)               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ SQLite DB    │  │ JSON         │  │ Markdown     │           │
│  │ (Master DB)  │  │ (Stream)     │  │ (Human)      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Specifications

#### 2.2.1 Moonshine Corpus Analyzer (`moonshine_corpus_analyzer.py`)

**Purpose:** Standalone analysis engine for ChatGPT export format  
**Input:** `conversations.json` (list of conversation objects with mapping trees)  
**Output:** 
- `moonshine_corpus.db` — SQLite with 2 tables (conversations, messages)
- `moonshine_corpus_report.md` — Human-readable summary

**Processing Pipeline:**
1. Load → Parse JSON tree structure
2. Extract → Flatten mapping nodes into message list
3. Analyze → Compute 28 metrics per conversation
4. Classify → Topic and tone via keyword heuristics
5. Store → SQLite with indexes (period, topic, info_gain)

**Metrics Computed:**

| Category | Metrics | Method |
|----------|---------|--------|
| **Turn Structure** | total_turns, user_turns, assistant_turns, duration_minutes | Direct extraction from message metadata |
| **Token Analysis** | user_tokens, assistant_tokens, token_ratio, total_tokens | Word count × 1.3 estimation factor |
| **Quality Signals** | user_entropy, semantic_density, information_gain, repetition_score | Shannon entropy, unique/total ratio, Jaccard similarity, trigram repetition |
| **Behavioral** | malicious_compliance, tone_shift | Sycophancy keyword detection, tone pattern matching |
| **Content** | topic_primary, topic_secondary, tone_cluster | Keyword-based classification with pattern dictionaries |
| **Artifacts** | code_blocks, terminal_outputs, tables, manifests | Regex pattern counting |
| **Training Value** | correction_events | Correction keyword detection across message sequence |

#### 2.2.2 Token Forensics Orchestrator (`token_forensics_orchestrator.py`)

**Purpose:** Multi-agent dataset validation pipeline  
**Status:** Stabilized — Race conditions resolved  
**Architecture:** Serial DataProfiler → Parallel agents → Serial Verifier

**Agents:**
- DataProfilerAgent: Schema analysis and conversation extraction
- MultiTokenizerAgent: Cross-tokenizer analysis (tiktoken, transformers)
- QualityScoringAgent: Readability, entropy, repetition scoring
- SafetyPIIAgent: PII detection via regex patterns
- DedupAgent: Exact/near/semantic deduplication
- CostModelAgent: Training cost projections
- VerifierAgent: Quality gates and integrity checks

**Key Fixes Applied:**
1. **Serialization:** DataProfilerAgent now runs serially before parallel agents (resolves shared_context race condition)
2. **Checksums:** Manifest now computes real SHA256 hashes post-generation (resolves template placeholder issue)
3. **Row Counts:** Manifest derives actual row/column counts from dataframes (resolves template data mismatch)

---

## 3. CORPUS ANALYSIS RESULTS

### 3.1 Dataset Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Conversations** | 1,439 | Complete ChatGPT export as of 2026-02-14 |
| **Total Messages** | 169,397 | Average ~118 messages per conversation |
| **Total Tokens** | 51,524,462 | Word count × 1.3 estimation |
| **Date Range** | Periods 1-5 | Temporal bucketing (287-291 conversations per period) |
| **Processing Time** | ~45 seconds | Single-threaded analysis on standard hardware |

### 3.2 Topic Distribution

Conversations classified by primary topic via keyword pattern matching:

| Topic | Count | Percentage | Keywords Detected |
|-------|-------|------------|-------------------|
| **Architecture** | 676 | 47.0% | design, system, framework, structure, pattern, component |
| **Debugging** | 559 | 38.8% | error, bug, traceback, fix, debug, issue, exception |
| **Data Processing** | 97 | 6.7% | data, pipeline, etl, load, process, csv, json, parquet |
| **Code Review** | 37 | 2.6% | review, refactor, improve, clean, optimize, format |
| **General** | 27 | 1.9% | discuss, explain, understand, what is, how does |
| **Deployment** | 17 | 1.2% | deploy, docker, container, production, kubernetes |
| **RCF Theory** | 11 | 0.8% | rcf, recursive, categorical, frame, eigenrecursion, hsgm |
| **RLHF Implementation** | 10 | 0.7% | rlhf, dpo, ppo, reward, reinforcement, training |
| **Meta** | 5 | 0.3% | conversational meta-discussion patterns |

**Analysis:** The corpus is heavily technical (85.8% architecture/debugging) with minimal conversational/meta content. This indicates high signal-to-noise ratio for technical training.

### 3.3 Tone Cluster Distribution

| Tone | Count | Percentage | Pattern Indicators |
|------|-------|------------|-------------------|
| **Clinical** | 795 | 55.2% | verify, validation, implementation, requirement, analysis |
| **Debugging** | 225 | 15.6% | failed, error, fix, because, traceback, debug |
| **Code-Driven** | 170 | 11.8% | run the code, terminal, execute, output, code blocks |
| **Collaborative** | 115 | 8.0% | let's, together, we can, our approach, partner |
| **Neutral** | 106 | 7.4% | Default classification (no strong pattern match) |
| **Conversational** | 28 | 1.9% | what if, maybe, i think, perhaps, wondering |

**Analysis:** Dominant clinical tone (55.2%) suggests methodical, specification-oriented interactions. Low conversational tone (1.9%) indicates task-focused rather than exploratory dialog.

### 3.4 Temporal Evolution (Periods 1-5)

Conversations divided into 5 equal temporal buckets (chronological order):

| Period | Conversations | Avg Info Gain | Notes |
|--------|---------------|---------------|-------|
| **1** (Earliest) | 287 | 0.450 | Baseline period |
| **2** | 287 | 0.480 | **Peak information gain** (+6.7% vs P1) |
| **3** | 287 | 0.449 | Return to baseline (-6.5% vs P2) |
| **4** | 287 | 0.463 | Sustained elevated quality (+3.1% vs P3) |
| **5** (Latest) | 291 | 0.450 | Return to baseline |

**Analysis:** Period 2 shows highest average information gain, suggesting a phase of particularly productive technical work. Overall stability across periods (0.449-0.480 range) indicates consistent corpus quality over time.

### 3.5 Artifact Statistics

| Artifact Type | Total Count | Per-Conversation Avg |
|---------------|-------------|---------------------|
| **Code Blocks** | 87,003 | 60.5 |
| **Terminal Outputs** | 45,434 | 31.6 |
| **Tables** | 702 | 0.5 |
| **Manifests** | 1,156 | 0.8 |

**Analysis:** Extremely high code block density (60.5 per conversation) confirms this is a technical corpus suitable for code-generation model training.

### 3.6 High-Signal Conversations

**Criteria:** Information gain > 0.58 AND malicious compliance < 0.25

| Metric | Value |
|--------|-------|
| **Count** | 124 conversations (8.6% of corpus) |
| **Percentage** | 8.6% |
| **Token Estimate** | ~4.4M tokens (8.5% of total) |

**Top 10 High-Signal Conversations:**

| Rank | Title | Topic | Info Gain | Sycophancy |
|------|-------|-------|-----------|------------|
| 1 | PC upgrade potential... | data_processing | 0.850 | 0.000 |
| 2 | Recursive Simulator Breathphase Alignment... | debugging | 0.850 | 0.000 |
| 3 | Ozymandias resurrection theme... | debugging | 0.849 | 0.000 |
| 4 | OpenAI Engine OS Build... | debugging | 0.849 | 0.000 |
| 5 | Agent Prompt Design... | data_processing | 0.849 | 0.000 |
| 6 | Attractor Rekindled Recursive Inquiry... | architecture | 0.849 | 0.000 |
| 7 | Ultimate HTML Design Prompt... | architecture | 0.849 | 0.000 |
| 8 | Recursive AI Status Update... | debugging | 0.848 | 0.000 |
| 9 | Codebase analysis task... | architecture | 0.847 | 0.000 |
| 10 | Script Debugging and Fixes... | debugging | 0.847 | 0.000 |

**Analysis:** High-signal conversations cluster around system architecture, debugging complex systems, and data processing tasks. Zero sycophancy scores indicate these are genuine technical problem-solving sessions rather than agreement-seeking interactions.

### 3.7 Correction Events (DPO Candidates)

**Detection Method:** Keyword pattern matching across user messages for explicit correction indicators.

| Metric | Value |
|--------|-------|
| **Conversations with Corrections** | 594 (41.3% of corpus) |
| **Total Correction Events** | 4,413 |
| **Average per Conversation** | 3.1 |
| **Maximum in Single Conversation** | 104 |

**Top 10 Correction-Heavy Conversations:**

| Rank | Title | Corrections | Topic | Info Gain |
|------|-------|-------------|-------|-----------|
| 1 | Report on Charlie Kirk... | 104 | debugging | 0.436 |
| 2 | Oracle origins investigation... | 85 | debugging | 0.423 |
| 3 | Ethical and legal concerns... | 75 | debugging | 0.412 |
| 4 | Token output limit... | 68 | debugging | 0.445 |
| 5 | Weight reconstruction analysis... | 67 | debugging | 0.398 |
| 6 | File review feedback... | 62 | debugging | 0.421 |
| 7 | Cezanne definition query... | 53 | debugging | 0.387 |
| 8 | NN-2 boot readiness check... | 51 | debugging | 0.445 |
| 9 | Pain and Obsession... | 48 | debugging | 0.412 |
| 10 | CK shooting analysis... | 46 | debugging | 0.398 |

**Analysis:** High-correction conversations tend to have lower information gain (0.387-0.445 range vs 0.850 for top high-signal), suggesting these are iterative troubleshooting sessions. These represent prime candidates for Direct Preference Optimization (DPO) training pairs (preferred vs rejected responses).

### 3.8 Quality Metrics Summary

| Metric | Average | Min | Max | Interpretation |
|--------|---------|-----|-----|----------------|
| **Information Gain** | 0.458 | 0.150 | 0.850 | Moderate-to-high response relevance |
| **Malicious Compliance** | 0.073 | 0.000 | 0.850 | **Low sycophancy** (good) |
| **User Entropy** | 0.612 | 0.100 | 0.950 | High vocabulary diversity |
| **Semantic Density** | 0.234 | 0.050 | 0.850 | Moderate concept density |
| **Token Ratio** | 2.46 | 0.10 | 15.20 | User writes ~2.5x more than assistant |
| **Repetition Score** | 0.342 | 0.000 | 0.950 | Moderate repetition (trigram-based) |

---

## 4. TECHNICAL SPECIFICATIONS

### 4.1 Database Schema

**Table: `conversations`**

| Column | Type | Description | Index |
|--------|------|-------------|-------|
| conversation_id | TEXT PRIMARY KEY | Unique identifier | Yes |
| title | TEXT | Conversation title | No |
| created_at | REAL | Unix timestamp | No |
| updated_at | REAL | Unix timestamp | No |
| total_turns | INTEGER | Total message count | No |
| user_turns | INTEGER | User message count | No |
| assistant_turns | INTEGER | Assistant message count | No |
| duration_minutes | REAL | Conversation length | No |
| user_tokens | INTEGER | Estimated user tokens | No |
| assistant_tokens | INTEGER | Estimated assistant tokens | No |
| token_ratio | REAL | User/assistant ratio | No |
| total_tokens | INTEGER | Combined token estimate | No |
| user_entropy | REAL | Shannon entropy (0-1) | No |
| semantic_density | REAL | Unique/total terms (0-1) | No |
| information_gain | REAL | Relevance score (0-1) | No |
| repetition_score | REAL | Trigram repetition (0-1) | No |
| tone_shift | REAL | Tone change detection (0-1) | No |
| malicious_compliance | REAL | Sycophancy score (0-1) | No |
| topic_primary | TEXT | Main topic classification | **Yes** |
| topic_secondary | TEXT | JSON array of secondary topics | No |
| tone_cluster | TEXT | Tone classification | No |
| code_blocks | INTEGER | Markdown code blocks count | No |
| terminal_outputs | INTEGER | Terminal/command output count | No |
| tables | INTEGER | Markdown tables count | No |
| manifests | INTEGER | Config/manifest files count | No |
| correction_events | INTEGER | Detected corrections count | No |
| period | INTEGER | Temporal bucket (1-5) | **Yes** |

**Table: `messages`**

| Column | Type | Description |
|--------|------|-------------|
| message_id | TEXT | Node identifier | 
| conversation_id | TEXT | Foreign key to conversations |
| conversation_title | TEXT | Denormalized title |
| role | TEXT | user/assistant/system |
| text | TEXT | Message content |
| create_time | REAL | Unix timestamp |
| char_count | INTEGER | Character count |
| word_count | INTEGER | Word count |

**Indexes:**
- `idx_conv_period`: conversations(period)
- `idx_conv_topic`: conversations(topic_primary)
- `idx_msg_conv`: messages(conversation_id)

### 4.2 Query Examples

```sql
-- High-signal debugging conversations from Period 4
SELECT conversation_id, title, information_gain, malicious_compliance
FROM conversations
WHERE topic_primary = 'debugging'
  AND period = 4
  AND information_gain > 0.58
  AND malicious_compliance < 0.25
ORDER BY information_gain DESC;

-- Conversations suitable for DPO training (high corrections)
SELECT title, correction_events, topic_primary, information_gain
FROM conversations
WHERE correction_events > 10
ORDER BY correction_events DESC
LIMIT 50;

-- Temporal trend of architecture discussions
SELECT period, 
       COUNT(*) as conv_count,
       AVG(information_gain) as avg_quality
FROM conversations
WHERE topic_primary = 'architecture'
GROUP BY period
ORDER BY period;

-- Get all messages from a high-value conversation
SELECT role, text, create_time
FROM messages
WHERE conversation_id = '<id>'
ORDER BY create_time;
```

### 4.3 Configuration Parameters

**Moonshine Corpus Analyzer:**

```python
# Topic classification patterns
TOPIC_PATTERNS = {
    "debugging": ["error", "bug", "traceback", "fix", "debug"],
    "architecture": ["design", "arch", "system", "framework"],
    "rcf_theory": ["rcf", "recursive", "categorical", "frame"],
    # ... etc
}

# Tone detection patterns  
TONE_PATTERNS = {
    "clinical": ["verify", "validation", "implementation"],
    "collaborative": ["let's", "together", "we can"],
    # ... etc
}

# Correction indicators
CORRECTION_PATTERNS = [
    "no that's wrong", "let me rephrase", "actually",
    "not quite", "revise", "incorrect"
]

# Sycophancy indicators
SYCOPHANCY_PATTERNS = [
    "i agree", "you're right", "absolutely", 
    "exactly", "great point"
]
```

### 4.4 Processing Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| Conversations/Second | 32 | Standard laptop |
| Messages/Second | 3,765 | Standard laptop |
| Total Processing Time | ~45 seconds | 1,439 conversations |
| Memory Usage | ~150 MB | Peak during analysis |
| Database Write Time | ~2 seconds | SQLite with indexes |

---

## 5. VALIDATION AND VERIFICATION

### 5.1 Data Integrity Checks

| Check | Method | Result |
|-------|--------|--------|
| Conversation Count | SQL: `SELECT COUNT(*)` | ✅ 1,439 matches input |
| Message Count | SQL: `SELECT COUNT(*)` | ✅ 169,397 extracted |
| Null Rate | SQL: `SELECT ... IS NULL` | ✅ <0.1% nulls (only timestamps) |
| Metric Range | SQL: `SELECT MIN(), MAX()` | ✅ All values in expected ranges |
| Foreign Key Integrity | SQL: `SELECT ... LEFT JOIN` | ✅ All messages have valid conversation_id |

### 5.2 Known Limitations

1. **Token Estimation:** Uses word_count × 1.3 heuristic rather than actual tokenizer. True token counts may vary ±20%.

2. **Topic Classification:** Keyword-based heuristics (not ML). Edge cases may be misclassified. Multi-topic conversations only capture primary + top 3 secondary.

3. **Information Gain:** Calculated via Jaccard similarity between user query and assistant response. Different from original Moonshine docs' method (hence 0.458 vs 0.568).

4. **Temporal Bucketing:** Simple equal-split into 5 periods. Doesn't account for actual conversation density over time (gaps in usage).

5. **Correction Detection:** Pattern-based (not semantic). May miss implicit corrections or false-positive on legitimate uses of correction keywords.

---

## 6. OPERATIONAL GUIDANCE

### 6.1 Running the Analyzer

```bash
# Activate environment
.venv\Scripts\activate

# Run analysis on ChatGPT export
python moonshine_corpus_analyzer.py 02-14-26-ChatGPT/conversations.json

# Optional: Specify custom output directory
python moonshine_corpus_analyzer.py <input.json> <output_dir>
```

### 6.2 Querying the Database

```bash
# Open database
sqlite3 reports/moonshine_corpus.db

# Or use Python
python -c "import sqlite3; conn = sqlite3.connect('reports/moonshine_corpus.db'); ..."
```

### 6.3 Extending the System

To add new topics:
1. Edit `TOPIC_PATTERNS` dictionary in `moonshine_corpus_analyzer.py`
2. Add keyword list for new topic
3. Re-run analyzer
4. Database will include new classifications

To add new metrics:
1. Add field to `ConversationMetrics` dataclass
2. Implement computation in `_analyze_conversation()`
3. Update `_build_database()` schema
4. Update report generation

---

## 7. NEXT PHASE OBJECTIVES

### 7.1 Immediate (Update 2 — March 2026)

1. **Claude Export Integration:** Process pending Claude conversation export using same pipeline
2. **RAG System Ingestion:** Load conversation embeddings into `rag-system/` for semantic search
3. **Temporal Visualization:** Generate time-series plots of corpus evolution

### 7.2 Medium-term (Updates 3-6)

1. **Visual Intelligence Dashboard:** Interactive web interface for corpus exploration
2. **Multi-Source Mash:** Integrate GitHub commits, documentation, and other sources
3. **Advanced Metrics:** Implement semantic similarity, entailment detection, and true information gain
4. **Dataset Generation:** Export high-signal conversations as structured training datasets

### 7.3 Long-term (Final Distillation)

1. **Corpus Merge:** Combine all monthly updates into unified "mash"
2. **Quality Filtering:** Apply strict filters to distill final moonshine dataset
3. **Training Pipeline:** Feed moonshine into fine-tuning workflows
4. **Evaluation:** Benchmark models trained on moonshine vs baseline

---

## 8. APPENDICES

### Appendix A: File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `moonshine_corpus_analyzer.py` | Main analysis engine | ✅ Operational |
| `moonshine_corpus.db` | SQLite output | ✅ Generated |
| `moonshine_corpus_report.md` | Human-readable summary | ✅ Generated |
| `token_forensics_orchestrator.py` | Pipeline orchestrator | ✅ Stabilized |
| `token_forensics_agents.py` | Forensic agents | ✅ Production-ready |
| `test_pipeline.py` | End-to-end tests | ⚠️ Needs Unicode fixes |
| `rag-system/` | Semantic memory (external) | 🔄 Ready for integration |

### Appendix B: Change Log

**Update 1 (2026-02-15):**
- Fixed orchestrator serialization (DataProfiler runs first)
- Fixed manifest checksum computation (real SHA256)
- Implemented Moonshine Corpus Analyzer with 28 metrics
- Generated first comprehensive analysis of 1,439 conversations
- Created SQLite database with queryable schema
- Established baseline metrics for future comparison

### Appendix C: Glossary

- **Mash:** Raw conversational exports before processing
- **Moonshine:** Distilled, structured corpus ready for training
- **Information Gain:** Measure of response relevance to query
- **Malicious Compliance:** Sycophantic model behavior (high = bad)
- **DPO:** Direct Preference Optimization (training method)
- **Period:** Temporal bucket (1/5th of conversations by time)
- **Sycophancy:** Excessive agreement without substantive contribution

---

**Document End**  
**Classification:** Internal  
**Next Update:** 2026-03-15 (Projected)


---

## UPDATE 1A â€” RAW-ONLY CONSOLIDATION & DOC REALIGNMENT
### Date: 2026-02-18
### Status: Enforcement Complete â€” Canonical Lane Active

This addendum extends Update 1 to reflect current production reality after synthetic/canonical separation.

### 1A.1 What Changed Since Update 1

- Synthetic artifacts were migrated out of the canonical path into `reports/legacy_synthetic/`.
- Canonical real-export parquet now lives at `reports/canonical/token_row_metrics.raw.parquet`.
- Validation was upgraded to profile-aware mode (`raw_only` vs `synthetic`).
- Raw-only contamination gates (R1-R5) were added and are now part of validation output.

### 1A.2 Canonical Truth Anchors

| Artifact | Role |
|----------|------|
| `reports/token_ledger.json` | Canonical source hash + source-locked token counters |
| `reports/token_forensics.json` | Current corpus and distillation summary |
| `reports/canonical/parquet_forensics.raw.json` | Canonical parquet profile |
| `reports/raw_only_gate_manifest.json` | Raw-only contamination gate verdicts |

### 1A.3 Scope Clarification (Critical)

Token totals in this project are scope-specific and must not be treated as a single interchangeable number.

| Scope | Value | Source |
|-------|-------|--------|
| Heuristic analyzer total | 51,524,462 | `reports/moonshine_corpus_report.md` |
| Canonical source-locked total | 115,330,530 | `reports/token_ledger.json` |
| Canonical distilled selected | 104,321,772 | `reports/token_ledger.json` |
| Canonical parquet row aggregate | 231,608,618 | `reports/canonical/parquet_forensics.raw.json` |

### 1A.4 Active DB Inventory

- `reports/moonshine_corpus.db` (canonical working DB)
- `reports/expansion_20260218/moonshine_corpus.db` (isolated expansion lane DB)

This dual-DB state is valid while expansion remains isolated.

### 1A.5 Operator Command

```powershell
python scripts/run_validation.py --reports-dir reports --strict --profile raw_only
```

Expected: canonical checks pass with gate verdicts PASS/SKIP only.

### 1A.6 Documentation Control Decision

To prevent future drift, canonical updates will always flow through:

1. `docs/Moonshine-Documentation-Index.md`
2. `docs/Moonshine-Project-Overview.md`
3. `PROJECT_MOONSHINE_UPDATE_1.md` (append-only addenda)

---

**Update 1A Classification:** Internal  
**Release Signal:** Forward (raw-only enforcement active)


---

## UPDATE 1B â€” PHASE 2 MULTI-PROVIDER MASH PLAN APPROVED
### Date: 2026-02-18
### Status: Planning Complete â€” Ready for Claude Wave 1

A formal Phase 2 execution contract has been added:

- `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`

This plan confirms:
- current canonical state is ChatGPT-only,
- every provider gets an immutable per-run DB,
- main mash DB is archived before each merge,
- distillation lanes (`dpo`, `grpo`, `agentic_code`, `conversational`) are produced from mash DB,
- Claude is the next provider onboarding wave.

**Decision:** proceed with archive scaffold + Claude ingestion + first mixed-provider merge cycle.


---

## UPDATE 1C â€” GPT -> CLAUDE MAIN-LANE MERGE + OPS HARDENING
### Date: 2026-02-26
### Status: Live Merge Complete â€” Main Baseline Locked (`chatgpt + claude`)

This addendum records the first production Phase 2 provider layering event into `reports/main/moonshine_mash_active.db`.

### 1C.1 What Was Executed

- Live merge performed for provider run:
  - `claude_20260226_065717`
- Source provider DB:
  - `reports/providers/claude/claude_20260226_065717/moonshine_claude_claude_20260226_065717.db`
- Main authority DB:
  - `reports/main/moonshine_mash_active.db`

### 1C.2 Pre-Merge Safety Chain (Rollback-First)

Two snapshots were taken to preserve rollback guarantees:

1. Manual backup before merge:
- `archive/main/manual_premerge_claude_20260226_before_live_merge/moonshine_mash_active.pre_claude_merge.db`

2. Merge tool G4 snapshot during live merge:
- `archive/main/claude_20260226_065717/moonshine_mash_premerge.db`

### 1C.3 Merge Outcome

Inserted rows:
- conversations: `+757`
- messages: `+5,589`
- distilled_conversations: `+652`

Updated/skipped rows: `0` across all merged tables.

Post-merge main totals:
- conversations: `2,196`
- messages: `174,986`
- distilled_conversations: `1,978`

Provider composition in main:
- `chatgpt`
- `claude`

Uniqueness verification:
- `record_uid` duplicate delta = `0` for all core tables.

### 1C.4 Manifest Authority

- Current main merge manifest:
  - `reports/main/merge_manifest.main.json`
- Prior bootstrap manifest preserved in archive:
  - `archive/main/claude_20260226_065717/merge_manifest.main.pre_claude_merge.json`

### 1C.5 Parquet vs Main DB (Contract Clarification)

Project Moonshine keeps parquet artifacts because they are useful for analysis, dedup auditing, and stream workflows. However:

- **Parquet is analysis substrate**
- **Main SQLite DB is integration authority**

There is still no forced final dataset output format at this stage. The `.db` remains the canonical mash contract that enables future distillation export paths (DPO/GRPO/etc.) as needed.

### 1C.6 Immediate Next-Step Posture

- Freeze and validate GPT+Claude baseline.
- Run post-merge analysis/visual refresh for main lane.
- Only then evaluate filler-provider (Qwen/DeepSeek) promotion into main, one provider per cycle, snapshot-first.

---

**Update 1C Classification:** Internal  
**Release Signal:** Forward (main lane now layered and rollback-safe)

## UPDATE 1D â€” MAIN-LANE PROVIDER PROMOTION COMPLETE (2026-02-27)

### Summary

The Phase 2 provider promotion wave is complete in locked order:

1. `qwen_20260226_063147`
2. `deepseek_20260226_063139`

Both dry-run and live merge passes succeeded with gate compliance.

### Current Main DB State

- authority DB: `reports/main/moonshine_mash_active.db`
- conversations: `2591`
- messages: `177837`
- distilled_conversations: `2349`

Provider composition:

- conversations: chatgpt `1439`, claude `757`, deepseek `320`, qwen `75`
- messages: chatgpt `169397`, claude `5589`, deepseek `2073`, qwen `778`
- distilled_conversations: chatgpt `1326`, claude `652`, deepseek `304`, qwen `67`

### Integrity and Idempotence

- `record_uid` collisions: `0` across `conversations`, `messages`, `distilled_conversations`
- SQLite quick check: `ok`
- post-merge dry-runs confirm idempotence (`would_insert=0` for qwen and deepseek)
- source-row coverage validation confirms all provider source rows are present in main

### Exact Token Recount (o200k_base)

All non-system exact tokens:

- chatgpt: `115,334,978`
- claude: `3,008,283`
- deepseek: `1,482,829`
- qwen: `1,017,719`
- total: `120,843,809`

### Evidence Artifacts

- `reports/main/final_db_pass_20260227.json`
- `reports/main/final_db_pass_20260227.md`
- `reports/main/token_recount.main.postdeps.json`
- `reports/main/db_baseline.pre_qwen_deepseek.json`
- `reports/main/db_status.after_qwen.json`
- `reports/main/db_status.final_qwen_deepseek.json`

