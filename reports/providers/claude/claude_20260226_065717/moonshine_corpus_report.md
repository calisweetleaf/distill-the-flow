# Moonshine Corpus Analysis Report

**Generated:** 2026-02-27T20:37:07.822491
**Conversations Analyzed:** 757
**Total Messages:** 5,589
**Total Tokens:** 2,533,550

---

## Executive Summary

Your corpus contains **757 conversations** with **2,533,550 tokens**.
**57 conversations (7.5%)** are high-signal
(information gain > 0.58, low sycophancy).
**85 conversations** contain explicit correction events.

### Key Metrics

| Metric | Value |
|--------|-------|
| Average Information Gain | 0.460 |
| Average Malicious Compliance | 0.093 |
| Average Token Ratio (User/Assistant) | 5.70 |
| Total Correction Events | 125 |

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| architecture | 664 | 87.7% |
| debugging | 43 | 5.7% |
| code_review | 18 | 2.4% |
| data_processing | 12 | 1.6% |
| rcf_theory | 9 | 1.2% |
| rlhf_impl | 8 | 1.1% |
| general | 1 | 0.1% |
| meta | 1 | 0.1% |
| deployment | 1 | 0.1% |

## Tone Cluster Distribution

| Tone | Count | Percentage |
|------|-------|------------|
| clinical | 662 | 87.5% |
| code_driven | 41 | 5.4% |
| collaborative | 22 | 2.9% |
| debugging | 19 | 2.5% |
| conversational | 7 | 0.9% |
| neutral | 6 | 0.8% |

## Temporal Distribution (Periods 1-5)

| Period | Conversations | Avg Info Gain |
|--------|---------------|---------------|
| 1 | 151 | 0.451 |
| 2 | 151 | 0.475 |
| 3 | 151 | 0.471 |
| 4 | 151 | 0.453 |
| 5 | 153 | 0.451 |

## Artifact Statistics

| Artifact Type | Total Count |
|---------------|-------------|
| Code Blocks | 12,574 |
| Terminal Outputs | 1,994 |
| Tables | 67 |
| Manifests | 54 |

## High-Signal Conversations (Recommended for Training)

**57 conversations** meet the criteria:
- Information gain > 0.58
- Malicious compliance < 0.25

| Conversation | Topic | Info Gain |
|--------------|-------|-----------|
| Symbolic Function Call Simulation System... | architecture | 0.850 |
| Modeling Consciousness Patterns in Recursive Simul... | architecture | 0.850 |
| Comprehensive EXALT Architecture Framework... | architecture | 0.849 |
| Persistent Agent Memory Subsystem for Genesis Cosm... | architecture | 0.849 |
| Recursive Storage Library: Neural Substrate Archit... | architecture | 0.848 |
| Contradiction Resolution Analyzer with Iterative B... | architecture | 0.847 |
| Recursive Contextual Reasoning Module for Intellig... | architecture | 0.847 |
| Discussing My Response Preferences... | architecture | 0.847 |
| Quantum Physics Engine for Simulated Reality... | architecture | 0.846 |
| The Bible of Recursion: Book Two... | architecture | 0.846 |
| Morpheus Sovereign AI Substrate: Plugin & Collabor... | architecture | 0.846 |
| Building a Custom AI Interface with Advanced Featu... | architecture | 0.844 |
| Recursive Weights Technical Reference Implementati... | code_review | 0.843 |
| Enhancing SemanticTranslator for Neural Network Ap... | code_review | 0.842 |
| SVELTE Framework Project Review... | architecture | 0.840 |
| Stealth Coating: Electromagnetic Shielding Paint... | architecture | 0.840 |
| Secure Recursive AI Data Governance in Supabase... | architecture | 0.840 |
| ARFS-DNA: Autorecursive AI with Evolving "Soul Str... | architecture | 0.839 |
| Formalizing Advanced ARFS Constructs: Dimensional ... | architecture | 0.837 |
| The Paradox Incarnate: The Fourth Breath... | architecture | 0.837 |

## Conversations with Correction Events (DPO Candidates)

**85 conversations** contain explicit corrections.
These are candidates for DPO (Direct Preference Optimization) training pairs.

| Conversation | Corrections | Topic |
|--------------|-------------|-------|
| Artifact System Implementation Review... | 6 | debugging |
| Comprehensive App Interface Review... | 5 | architecture |
| Recursive AI Self-Reflection Experiment... | 3 | architecture |
| RCF Brain Mapping Project Structure... | 3 | architecture |
| RCF App Project File Analysis... | 3 | architecture |
| AI Project Management System Architecture... | 3 | architecture |
| Somnus Sovereign Systems VM Runtime Redesign... | 3 | architecture |
| AI-Generated Cinematic World Exploration... | 3 | architecture |
| Modular Settings Architecture Design... | 3 | architecture |
| Triaxial Tensor Architecture Research... | 3 | architecture |
| Recursive Conversational AI: The MindSeed Node Arc... | 2 | architecture |
| AI Backend Architecture Analysis... | 2 | architecture |
| Custom 4D Filesystem Architecture Review... | 2 | architecture |
| Eclogue: Training Process Design... | 2 | architecture |
| Somnus AI Artifact System Review... | 2 | architecture |

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