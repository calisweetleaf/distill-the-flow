# Moonshine: Execution Roadmap & Timeline

**Project:** Point B Corpus Analysis  
**Owner:** Solo Developer  
**Resources:** Self (1 FTE)  
**Environment:** Single machine + cloud storage (optional)  
**Last Updated:** February 15, 2026

---

## 1. Executive Summary

This document outlines a **12-week execution plan** to transform 1,385 GPT conversations into a production-ready training dataset pipeline.

### High-Level Timeline

```
Week 1-2  : Foundation (Metadata + Indexing)
Week 3-4  : DPO Pair Generation (Core RLHF data)
Week 5-6  : Advanced Analytics (Model fingerprinting, language evolution)
Week 7-8  : Validation & Refinement
Week 9-10 : Export & Format Optimization
Week 11-12: Documentation & Release
```

**Estimated Output:**

- 300-500 high-quality DPO pairs
- Searchable SQLite index (1,385 conversations)
- 20-30 agentic workflow patterns
- Model version fingerprints for all messages
- Temporal vocabulary evolution dataset

---

## 2. Detailed Weekly Breakdown

### **WEEK 1-2: Foundation Layer**

**Goal:** Build searchable index + metadata layer

#### Task 1.1: Environment Setup

- [ ] Create project structure (`moonshine/` repo)
- [ ] Initialize SQLite database schema
- [ ] Set up Python environment (dependencies)
- [ ] Create config files (paths, thresholds, etc.)

**Effort:** 2-3 hours  
**Deliverables:** `moonshine/config.yaml`, environment ready

---

#### Task 1.2: Conversation Ingestion

Implement `MetadataExtractor` class (from Technical Implementation guide)

```
Subtasks:
- [ ] JSON parser for ChatGPT export format
- [ ] Handle edge cases (malformed messages, missing fields)
- [ ] Validate conversation structure
- [ ] Parse timestamps correctly
```

**Effort:** 4-6 hours  
**Deliverables:** `ingest.py`, test suite, ingestion logs

**Key Metric to Track:** Conversations successfully parsed (target: 100%)

---

#### Task 1.3: Metadata Extraction

Implement all metadata calculations:

- Turn counts (user vs. assistant)
- Duration & token counts
- Topic inference (keyword-based)
- Artifact counting (code blocks, terminals, tables)
- Correction event detection (heuristic-based)

```
Subtasks:
- [ ] Turn count logic + validation
- [ ] Topic keyword dictionary (20-30 topics)
- [ ] Artifact counters (code, terminal, yaml, etc.)
- [ ] Correction detector (indicator phrases)
```

**Effort:** 6-8 hours  
**Deliverables:** `metadata_extractor.py`, 1,385 rows of metadata

**Key Metric:** Metadata completeness ≥ 95%

---

#### Task 1.4: SQLite Indexing

- [ ] Create database schema (3 tables: conversations, messages, segments)
- [ ] Bulk insert all metadata
- [ ] Create indexes on frequently-queried columns
- [ ] Validate data integrity (no NULLs in required fields)

**Effort:** 3-4 hours  
**Deliverables:** `moonshine.db` (SQLite), index statistics

---

#### Task 1.5: CSV Export + QA

- [ ] Export metadata to `metadata.csv`
- [ ] Spot-check 50 rows manually
- [ ] Generate summary statistics report
- [ ] Document any anomalies

**Effort:** 2-3 hours  
**Deliverables:**

- `metadata.csv`
- `reports/week1_summary.txt`

**Success Criteria:**

- ✅ All 1,385 conversations indexed
- ✅ No missing metadata fields (≥95% complete)
- ✅ Spot checks pass manual validation
- ✅ Database query time <100ms for common filters

---

### **WEEK 3-4: DPO Pair Generation (Core RLHF Data)**

**Goal:** Extract 300+ high-quality correction pairs

#### Task 3.1: Correction Pattern Detection

Implement `DPOPairGenerator.detect_correction_pattern()`

**Subtasks:**

- [ ] Define correction pattern (User → Assistant → User correction → Assistant revised)
- [ ] Implement correction indicator phrases (20+ indicators)
- [ ] Text similarity heuristic (avoid >95% similarity, <30% divergence)
- [ ] Handle edge cases (multiple corrections in sequence, multi-turn corrections)

**Effort:** 5-7 hours  
**Deliverables:** `dpo_generator.py`, pattern detection logic

**Expected Yield Estimate:** ~400-500 raw candidates (before filtering)

---

#### Task 3.2: Correction Type Classification

Implement `infer_correction_type()`:

- `syntax_error` - bracket/quote/indentation issues
- `logic_error` - wrong reasoning/algorithm
- `incomplete` - unfinished response
- `unclear` - confusing explanation
- `style` - cleaner/better formatting

**Subtasks:**

- [ ] Rule-based classifiers for each type
- [ ] Validate classification accuracy (spot-check 30 pairs)
- [ ] Measure distribution across types

**Effort:** 3-4 hours  
**Deliverables:** Classification logic, type distribution report

---

#### Task 3.3: Confidence Scoring

Implement `compute_confidence()`:

Scoring factors:

- Prompt length (>20 words = +0.1)
- Response similarity (0.3-0.8 = +0.2)
- Correction message clarity (>10 chars = +0.1)
- Correction type (high-signal types = +0.1)
- Base score: 0.5

**Effort:** 2-3 hours  
**Deliverables:** Confidence scoring logic, distribution histogram

**Expected Distribution:**

- High (≥0.8): ~150-200 pairs (ready for training)
- Medium (0.5-0.8): ~150-200 pairs (needs review)
- Low (<0.5): ~100-150 pairs (archive)

---

#### Task 3.4: Manual Validation

Sample and validate correction pairs:

- [ ] Review 10 high-confidence pairs (≥0.9)
- [ ] Review 10 medium-confidence pairs (0.7-0.8)
- [ ] Review 10 low-confidence pairs (<0.5)
- [ ] Document false positives/negatives
- [ ] Adjust heuristics if needed

**Effort:** 3-4 hours  
**Deliverables:** Validation report, confidence threshold recommendations

---

#### Task 3.5: Export DPO Dataset

- [ ] Export high-confidence pairs to `dpo_pairs_train.jsonl` (threshold: ≥0.8)
- [ ] Export medium-confidence to `dpo_pairs_review.jsonl` (for human review)
- [ ] Generate metadata: pair statistics, type distribution, quality metrics

**Effort:** 2-3 hours  
**Deliverables:**

- `dpo_pairs_train.jsonl` (ready for RLHF training)
- `dpo_pairs_review.jsonl` (validation set)
- `reports/dpo_summary.json`

---

**Week 3-4 Success Criteria:**

- ✅ ≥300 DPO pairs extracted
- ✅ ≥70% manual validation pass rate
- ✅ Confidence distribution documented
- ✅ No PII/sensitive data in pairs
- ✅ JSONL format valid (parseable by training pipelines)

---

### **WEEK 5-6: Advanced Analytics**

#### Task 5.1: Model Version Fingerprinting

Implement `ModelFingerprinter` class

**Subtasks:**

- [ ] Define GPT-4o vs. GPT-4.5 vs. GPT-5 signatures
- [ ] Collect indicator phrases for each variant (20-30 per model)
- [ ] Implement emoji/punctuation frequency detectors
- [ ] Test on known examples (confidence target: ≥70%)

**Effort:** 4-5 hours  
**Deliverables:**

- `fingerprinter.py`
- Model detection accuracy report

**Expected Result:** ~60-70% conversations tagged with model version

---

#### Task 5.2: Tool Use Pattern Extraction

Implement `ToolUseAnalyzer` class

**Subtasks:**

- [ ] Artifact usage metrics (code generations, debugging loops, etc.)
- [ ] Workflow classification (debugging, code_generation, review, mixed)
- [ ] Temporal patterns (which periods had highest artifact density)

**Effort:** 3-4 hours  
**Deliverables:**

- `tool_analyzer.py`
- Workflow type distribution report

**Expected Patterns:**

- Interactive debugging: 30-40% of high-artifact conversations
- Code generation focused: 25-35%
- Code review focused: 15-25%
- Mixed: 10-15%

---

#### Task 5.3: Temporal Language Evolution

Implement `LanguageEvolutionAnalyzer` class

**Subtasks:**

- [ ] Extract technical terms (CamelCase, ALL_CAPS, underscored)
- [ ] Track term emergence by period
- [ ] Measure vocabulary growth
- [ ] Identify domain-specific terms (HSGM, RCF, eigenrecursion, etc.)

**Effort:** 3-4 hours  
**Deliverables:**

- `language_analyzer.py`
- Vocabulary evolution timeline
- New term emergence chart

**Expected Output:** 50-100 tracked technical terms, clear emergence pattern

---

#### Task 5.4: Dependency & Reference Graphs

Implement conversation linking:

**Subtasks:**

- [ ] Detect cross-conversation references ("as we discussed...")
- [ ] Build concept dependency graph (networkx)
- [ ] Identify foundational concepts (appearing in 30+ conversations)
- [ ] Visualize dependency structure

**Effort:** 4-5 hours  
**Deliverables:**

- Graph structure (JSON/pickle)
- Visualization (HTML graph or static image)
- Foundational concepts report

---

**Week 5-6 Success Criteria:**

- ✅ Model fingerprints assigned to ≥1,000 messages
- ✅ Workflow patterns identified and documented
- ✅ Vocabulary evolution timeline complete
- ✅ Dependency graph constructed
- ✅ Advanced analytics ready for dashboard

---

### **WEEK 7-8: Validation & Refinement**

#### Task 7.1: Data Quality Audit

- [ ] Duplicate detection (conversations, pairs, messages)
- [ ] Missing data analysis (% null values per field)
- [ ] Outlier detection (unusual token counts, turn counts, etc.)
- [ ] PII scanning (email addresses, API keys, phone numbers)

**Effort:** 3-4 hours  
**Deliverables:** Quality audit report, remediation recommendations

---

#### Task 7.2: Statistical Validation

- [ ] Recompute correlations (verify against original analysis)
- [ ] Validate temporal bucketing (periods 1-5 alignment)
- [ ] Test metric calculations (information gain, entropy, etc.)
- [ ] Cross-check against visualization data

**Effort:** 2-3 hours  
**Deliverables:** Statistical validation report

---

#### Task 7.3: Sampling & Manual Spot Checks

- [ ] Random sample 50 conversations
- [ ] Manual review of metadata completeness
- [ ] Verify topic classifications
- [ ] Check correction event detection (false positive rate)
- [ ] Validate model fingerprints on known examples

**Effort:** 4-5 hours  
**Deliverables:** Spot-check report, error rate measurements

**Acceptance Criteria:**

- Topic classification accuracy: ≥75%
- Correction detection recall: ≥80%
- Model fingerprint accuracy: ≥70%

---

#### Task 7.4: Heuristic Refinement

Based on validation findings:

- [ ] Adjust correction thresholds if needed
- [ ] Improve topic keywords
- [ ] Refine model signatures
- [ ] Re-run analysis on full dataset if major changes

**Effort:** 2-3 hours (if minor); 4-6 hours (if major)  
**Deliverables:** Updated heuristics, re-analysis results (if applicable)

---

**Week 7-8 Success Criteria:**

- ✅ All audit findings documented
- ✅ Data quality score ≥0.85
- ✅ No PII detected
- ✅ Spot-check accuracy ≥75% average
- ✅ Ready for production export

---

### **WEEK 9-10: Export & Format Optimization**

#### Task 9.1: Multi-Format Export

- [ ] JSONL export (DPO pairs, ready for training)
- [ ] CSV export (metadata summary)
- [ ] Parquet export (full messages with segmentation)
- [ ] HuggingFace dataset format (for easy integration)

**Effort:** 3-4 hours  
**Deliverables:**

- `dpo_pairs_v1.0.jsonl`
- `metadata_v1.0.csv`
- `messages_v1.0.parquet`
- `dataset_v1.0.hf/` (HF-compatible)

---

#### Task 9.2: Training Dataset Preparation

Format for Aeron (400M param) fine-tuning:

**Subtasks:**

- [ ] Filter to high-quality pairs (confidence ≥0.75)
- [ ] Stratify by correction type (even distribution)
- [ ] Train/val split (80/20)
- [ ] Tokenize and format for PyTorch

**Effort:** 3-4 hours  
**Deliverables:**

- `aeron_train.jsonl` (~250 pairs)
- `aeron_val.jsonl` (~50 pairs)
- Format specification document

---

#### Task 9.3: Rosemary Training Traces

Format for RCF (Recursive Categorical Framework) agent:

**Subtasks:**

- [ ] Extract reasoning traces (multi-turn conversations with step-by-step logic)
- [ ] Format as reasoning chains
- [ ] Tag with RCF concepts (recursion depth, categorical reasoning, etc.)

**Effort:** 2-3 hours  
**Deliverables:**

- `rosemary_traces.jsonl`
- RCF format specification

---

#### Task 9.4: Documentation & Release Notes

- [ ] Write dataset card (what's included, how to use)
- [ ] Create usage examples (Python code)
- [ ] Document version (v1.0, release date, changes)
- [ ] License assertion (SAEL v4.0)

**Effort:** 2-3 hours  
**Deliverables:**

- `DATASET_CARD.md`
- `USAGE_EXAMPLES.py`
- `CHANGELOG.md`

---

**Week 9-10 Success Criteria:**

- ✅ All datasets exported and validated
- ✅ Format compatibility verified (can load in PyTorch, HF)
- ✅ Documentation complete
- ✅ Version 1.0 ready for release

---

### **WEEK 11-12: Documentation & Release**

#### Task 11.1: Methodology Paper

Write comprehensive methodology document:

**Sections:**

- Introduction (corpus overview, motivation)
- Methods (extraction pipeline, heuristics, validation)
- Results (statistics, quality metrics, case studies)
- Limitations (known biases, false positive rates)
- Future work (improvements, extensions)

**Effort:** 4-5 hours  
**Deliverables:** `METHODOLOGY.md` (2,000-3,000 words)

---

#### Task 11.2: Internal Knowledge Base

Create internal wiki/documentation:

- [ ] Architecture diagrams (data flow)
- [ ] Heuristic explanations (why each threshold)
- [ ] Troubleshooting guide (common errors, fixes)
- [ ] Maintenance guide (how to re-run pipeline, update heuristics)

**Effort:** 3-4 hours  
**Deliverables:** Notion doc or README structure

---

#### Task 11.3: GitHub Release

- [ ] Push all code to `moonshine/` repo
- [ ] Tag v1.0 release
- [ ] Publish all datasets (SAEL licensed)
- [ ] Create GitHub release notes

**Effort:** 2-3 hours  
**Deliverables:** Public GitHub repo with v1.0 tag

---

#### Task 11.4: Project Retrospective

- [ ] Quantify results (pairs extracted, index size, etc.)
- [ ] Document lessons learned
- [ ] Plan Phase 2 improvements
- [ ] Estimate effort for future re-runs

**Effort:** 1-2 hours  
**Deliverables:** Retrospective report

---

**Week 11-12 Success Criteria:**

- ✅ Methodology paper published
- ✅ Documentation complete
- ✅ v1.0 released on GitHub
- ✅ All datasets accessible (SAEL compliant)
- ✅ Ready for Phase 3 work

---

## 3. Resource Allocation

### Personnel

- **Solo Developer:** 100% allocated (yourself)
- **External Reviewers:** Async validation (if needed)

### Infrastructure

- **Development:** Laptop (local SQLite)
- **Storage:** ~50GB total (raw corpus + intermediate files + exports)
- **Optional Cloud:** S3 backup of final datasets

### Time Budget

| Phase | Weeks | Hours | Notes |
|-------|-------|-------|-------|
| Foundation | 1-2 | 20-25 | Setup, indexing |
| DPO Generation | 3-4 | 15-20 | Core RLHF data |
| Advanced Analytics | 5-6 | 15-18 | Model FP, language evolution |
| Validation | 7-8 | 12-16 | QA, spot checks |
| Export | 9-10 | 12-15 | Multi-format, training prep |
| Documentation | 11-12 | 10-12 | Paper, release |
| **TOTAL** | **12 weeks** | **84-106 hours** | ~10-20 hrs/week (part-time pace) |

---

## 4. Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| JSON parse errors | Medium | Medium | Write robust parser with error handling |
| Metadata extraction bugs | Medium | High | Extensive unit tests, manual spot-checks |
| Heuristic over-fitting | High | Medium | Validate on held-out sample, adjust if needed |
| Database scaling | Low | Medium | SQLite sufficient for 1.3K conversations |
| DPO pair quality | Medium | High | Manual validation of 30-50 pairs, confidence thresholds |

### Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Underestimated complexity | Medium | Medium | Build buffer weeks 7-8 for refinement |
| Heuristic refinement loop | Medium | Low | Set iteration limit (max 2 rounds) |
| Data quality issues | Low | High | Plan audit tasks in week 7-8 early |

---

## 5. Success Metrics

### Primary Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| DPO pairs extracted | ≥300 | Pending | Phase 3-4 |
| Metadata completeness | ≥95% | Pending | Phase 1 |
| Model fingerprints | 1,000+ messages | Pending | Phase 5 |
| Validation accuracy | ≥75% | Pending | Phase 7-8 |
| Datasets released | 4+ formats | Pending | Phase 9-10 |

### Secondary Metrics

- Correction type distribution (balanced across 5 types)
- Workflow patterns identified (20+)
- Vocabulary evolution terms tracked (50+)
- Time-to-query performance (<100ms)

---

## 6. Next Actions (Starting This Week)

### Immediate (Next 3 Days)

- [ ] Create `moonshine/` project directory
- [ ] Initialize Git repo + GitHub
- [ ] Write `config.yaml` (paths, thresholds)
- [ ] Prepare ChatGPT export JSON file

### This Week

- [ ] Implement JSON ingestion + basic parsing
- [ ] Create SQLite schema
- [ ] Start metadata extraction (focus on turn counts first)
- [ ] Set up logging + error handling

### Next Week

- [ ] Complete metadata extraction for all 1,385 conversations
- [ ] Build SQLite database
- [ ] Perform initial QA (no NULLs, valid ranges)
- [ ] Generate first metadata report

---

## Appendix A: Glossary

- **DPO**: Difference Preference Optimization (RLHF method)
- **RLHF**: Reinforcement Learning from Human Feedback
- **Fingerprint**: Model version detection based on writing style
- **Heuristic**: Rule-based (non-ML) algorithm
- **Confidence Score**: Probability that extracted data is correct
- **Spot Check**: Manual validation of small random sample

---

## Appendix B: Related Documents

- `Moonshine-Project-Overview.md` - High-level strategy
- `Moonshine-Technical-Implementation.md` - Detailed code architecture
- `Moonshine-Analysis-Findings.md` - Detailed metric interpretations

---

**Prepared by:** Solo Developer  
**Date:** February 15, 2026  
**Next Review:** Weekly (end of week 2)
