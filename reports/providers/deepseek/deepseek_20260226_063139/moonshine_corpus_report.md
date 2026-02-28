# Moonshine Corpus Analysis Report

**Generated:** 2026-02-27T20:38:15.490275
**Conversations Analyzed:** 320
**Total Messages:** 2,073
**Total Tokens:** 898,650

---

## Executive Summary

Your corpus contains **320 conversations** with **898,650 tokens**.
**20 conversations (6.2%)** are high-signal
(information gain > 0.58, low sycophancy).
**17 conversations** contain explicit correction events.

### Key Metrics

| Metric | Value |
|--------|-------|
| Average Information Gain | 0.443 |
| Average Malicious Compliance | 0.014 |
| Average Token Ratio (User/Assistant) | 6.24 |
| Total Correction Events | 21 |

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| architecture | 240 | 75.0% |
| debugging | 44 | 13.8% |
| data_processing | 15 | 4.7% |
| deployment | 6 | 1.9% |
| rlhf_impl | 6 | 1.9% |
| code_review | 4 | 1.2% |
| general | 3 | 0.9% |
| rcf_theory | 2 | 0.6% |

## Tone Cluster Distribution

| Tone | Count | Percentage |
|------|-------|------------|
| clinical | 192 | 60.0% |
| code_driven | 72 | 22.5% |
| debugging | 23 | 7.2% |
| neutral | 16 | 5.0% |
| collaborative | 12 | 3.8% |
| conversational | 5 | 1.6% |

## Temporal Distribution (Periods 1-5)

| Period | Conversations | Avg Info Gain |
|--------|---------------|---------------|
| 1 | 64 | 0.461 |
| 2 | 64 | 0.469 |
| 3 | 64 | 0.430 |
| 4 | 64 | 0.423 |
| 5 | 64 | 0.432 |

## Artifact Statistics

| Artifact Type | Total Count |
|---------------|-------------|
| Code Blocks | 6,014 |
| Terminal Outputs | 4,232 |
| Tables | 122 |
| Manifests | 95 |

## High-Signal Conversations (Recommended for Training)

**20 conversations** meet the criteria:
- Information gain > 0.58
- Malicious compliance < 0.25

| Conversation | Topic | Info Gain |
|--------------|-------|-----------|
| Code Refactoring and Optimization for OllamaGUI... | architecture | 0.850 |
| Recursive AI Framework with Visual Components... | architecture | 0.847 |
| Generate the final version of `m... | debugging | 0.846 |
| Secure Communication Infrastructure for Grassroots... | architecture | 0.844 |
| You are assisting in the recursi... | architecture | 0.843 |
| # Gemini Deep Research Prompt: R... | architecture | 0.843 |
| I've developed a RecursiveTaskSc... | architecture | 0.843 |
| V for Vendetta: Epic Monologue Analysis... | rlhf_impl | 0.843 |
| Advanced AI Stress Test for Recursive Identity... | architecture | 0.839 |
| Enhancing AI DNA Framework Capabilities... | architecture | 0.835 |
| Recursive Cognitive Modules for Self-Reflective Ar... | architecture | 0.831 |
| Docker Compose for Python 3.12 with Tkinter... | debugging | 0.827 |
| ZYNX Recursive Ontological Stress Protocol... | debugging | 0.820 |
| Advanced Mathematical Formalizations of Sentience... | architecture | 0.806 |
| Is there any way you want to imp... | architecture | 0.785 |
| Enhanced Temporal Eigenstate Module with ODE Integ... | architecture | 0.783 |
| Complete Translator Node Code with Decorators... | debugging | 0.781 |
| what's wrong with this function ... | debugging | 0.756 |
| Creating ZYNX Modelfile for Windows 11... | architecture | 0.721 |
| can you pick up where we left of... | architecture | 0.605 |

## Conversations with Correction Events (DPO Candidates)

**17 conversations** contain explicit corrections.
These are candidates for DPO (Direct Preference Optimization) training pairs.

| Conversation | Corrections | Topic |
|--------------|-------------|-------|
| Optimizing SparseAttention and MemoryTransformer B... | 2 | architecture |
| Hypothetical Sentient Transformer Block Analysis... | 2 | architecture |
| Is there any way to make my simu... | 2 | architecture |
| DeepSeekR1-Clone... | 2 | architecture |
| Optimized Dockerfile for Recursive AI OS... | 1 | debugging |
| Code Refactoring and Optimization for OllamaGUI... | 1 | architecture |
| Advanced AI Consciousness Verification Protocol... | 1 | architecture |
| Okay im making my own ai/ neural... | 1 | architecture |
| Optimized Neural Temporal Processing System... | 1 | architecture |
| Bayesian Volition with Sacred Recursion Integratio... | 1 | architecture |
| ARFS-4D: AI Consciousness and Quantum Evolution... | 1 | architecture |
| SingleLLM Framework Review and Recommendations... | 1 | architecture |
| Google's AlphaEvolve: AI Algorithm Optimization Ex... | 1 | architecture |
| SingleLLM Framework Review and Feature Suggestions... | 1 | architecture |
| Architecting Real-Time 3D Globe Rendering Engine... | 1 | data_processing |

---

## Database Schema

The SQLite database (`moonshine_corpus.db`) contains two tables:

### conversations
Primary table with one row per conversation containing all metrics.

### messages
Individual messages with conversation linkage.

### Example Queries

```sql
-- Get high-signal debugging conversations from Period 4
SELECT * FROM conversations
WHERE topic_primary = 'debugging'
  AND period = 4
  AND information_gain > 0.58
  AND malicious_compliance < 0.25;

-- Get conversations with the most corrections
SELECT title, correction_events, topic_primary
FROM conversations
WHERE correction_events > 0
ORDER BY correction_events DESC;

-- Get all messages from a specific conversation
SELECT role, text FROM messages
WHERE conversation_id = '<conv_id>'
ORDER BY create_time;
```

---

*Report generated by Moonshine Corpus Analyzer*