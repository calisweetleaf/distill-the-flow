# Moonshine Documentation Index

**Complete documentation suite for Project Moonshine (Point B: Corpus Analysis)**

Generated: February 15, 2026

---

## 📚 Start Here

### For Project Overview

👉 **Read First:** `Moonshine-Project-Overview.md`

- Executive summary of the entire project
- Current analysis status
- High-level roadmap
- Success criteria

### For Detailed Understanding

👉 **Read Second:** `Moonshine-Analysis-Findings.md`

- Deep dive into all findings
- Interpretation of metrics
- Specific recommendations for Aeron and Rosemary training
- Known limitations and caveats

### For Implementation

👉 **Read When Coding:** `Moonshine-Technical-Implementation.md`

- Complete architecture overview
- Pseudocode for all components
- Database schemas (SQLite, JSONL, Parquet)
- Quality assurance procedures
- Deployment guidelines

### For Project Management

👉 **Read for Execution:** `Moonshine-Phase-Roadmap.md`

- 12-week execution timeline
- Detailed weekly breakdown with tasks
- Resource allocation
- Risk mitigation strategies
- Success metrics and tracking

### For Quick Reference

👉 **Bookmark This:** `Moonshine-Quick-Reference.md`

- Key numbers at a glance
- Critical findings summary
- Immediate action items
- Aeron/Rosemary training configs
- Emergency troubleshooting

### This File

📌 **You Are Here:** `Moonshine-Documentation-Index.md`

- Navigation guide
- File descriptions
- How to use this suite

---

## 📋 Document Descriptions

### 1. Moonshine-Project-Overview.md

**What:** High-level project strategy and current status report  
**Length:** ~3,500 words  
**Key Sections:**

- Project mission and philosophy
- Completed analyses summary (6 analyses done)
- Tier 1-3 analysis recommendations
- Implementation roadmap (Phase 1-4)
- Current metrics dashboard
- Success criteria

**When to Read:** First, before any technical work  
**Owner:** Strategic planning  
**Update Frequency:** Monthly

---

### 2. Moonshine-Analysis-Findings.md

**What:** Detailed interpretation of all findings with actionable insights  
**Length:** ~4,500 words  
**Key Sections:**

- Metric correlations deep dive (the core -0.46 finding)
- ChatGPT export analysis (temporal trends)
- Dataset split recommendations with confidence levels
- Malicious compliance analysis (sycophancy scoring)
- Tone clustering (6 distinct communication modes)
- Temporal trends (Period 1-5 evolution)
- RLHF training recommendations (conversation filtering)
- Rosemary RCF training guidance
- Distillation-resistance strategy
- Known limitations
- Future analysis opportunities

**When to Read:** After overview, before implementation  
**Owner:** Data analysis & decision-making  
**Update Frequency:** As new analyses complete

---

### 3. Moonshine-Technical-Implementation.md

**What:** Complete technical architecture and implementation guide  
**Length:** ~5,000 words  
**Key Sections:**

- System architecture diagram
- Phase 1: Metadata extraction (detailed pseudocode)
- Phase 2: Error & correction analysis (DPO pair generation)
- Phase 3: Advanced analytics (fingerprinting, tool use, language evolution)
- Data formats & schemas (SQLite, JSONL, CSV, Parquet)
- Quality assurance checklist
- Deployment & scaling strategies
- Dependencies list

**When to Read:** When implementing code  
**Owner:** Engineering & implementation  
**Update Frequency:** As code evolves

---

### 4. Moonshine-Phase-Roadmap.md

**What:** 12-week execution plan with detailed task breakdown  
**Length:** ~4,000 words  
**Key Sections:**

- Executive summary with timeline
- Weekly breakdown (Weeks 1-12) with:
  - Goal for each week
  - Specific subtasks
  - Effort estimates
  - Deliverables
  - Success criteria
- Resource allocation
- Risk mitigation table
- Success metrics
- Next immediate actions

**When to Read:** When planning and executing project  
**Owner:** Project management  
**Update Frequency:** Weekly (track progress)

---

### 5. Moonshine-Quick-Reference.md

**What:** One-page reference guide with key numbers and decisions  
**Length:** ~2,500 words  
**Key Sections:**

- File navigation table
- Key numbers at a glance
- Critical findings (5 TL;DR points)
- Decision checklist for training data selection
- Next immediate actions (prioritized)
- Weekly metrics to track
- Aeron fine-tuning config
- Rosemary preparation guide
- Emergency troubleshooting
- Glossary

**When to Read:** Daily; bookmark this  
**Owner:** Everyone working on project  
**Update Frequency:** Weekly

---

### 6. Moonshine-Documentation-Index.md

**What:** This file; navigation guide for entire documentation suite  
**Length:** ~1,500 words  
**Key Sections:**

- Start here guide
- Document descriptions
- Reading order recommendation
- Question lookup table
- File maintenance info

**When to Read:** When lost or onboarding new person  
**Owner:** Documentation  
**Update Frequency:** As new documents added

---

## 🎯 Reading Guide by Role

### If You're the Solo Developer (You)

**Read in order:**

1. Moonshine-Project-Overview.md (30 min)
2. Moonshine-Quick-Reference.md (15 min) - bookmark
3. Moonshine-Analysis-Findings.md (45 min) - for decisions
4. Moonshine-Phase-Roadmap.md (weekly reference)
5. Moonshine-Technical-Implementation.md (as needed for coding)

**Time investment:** ~3 hours initial, then 1-2 hours/week ongoing

---

### If You're a Reviewer/Collaborator

**Read in order:**

1. Moonshine-Project-Overview.md (30 min)
2. Moonshine-Analysis-Findings.md (45 min)
3. Moonshine-Quick-Reference.md (15 min)
4. Optional: Moonshine-Phase-Roadmap.md (for tracking)

**Time investment:** ~2 hours

---

### If You're Integrating Aeron/Rosemary Later

**Read in order:**

1. Moonshine-Quick-Reference.md (15 min) - get context
2. Moonshine-Analysis-Findings.md sections:
   - "Recommendations for RLHF Training (Aeron)"
   - "Recommendations for Rosemary (RCF Training)"
3. Aeron config from Quick-Reference.md
4. Rosemary config from Analysis-Findings.md

**Time investment:** ~1.5 hours

---

## ❓ Quick Question Lookup

**Q: How many DPO pairs will we get?**
A: See Moonshine-Quick-Reference.md section "Key Numbers" → 300-500 pairs expected  
→ Expanded in Analysis-Findings.md section "Correction Events"

---

**Q: What conversations should I prioritize for training?**
A: See Moonshine-Quick-Reference.md section "Quick Decisions Checklist"  
→ Detailed in Moonshine-Analysis-Findings.md section "Recommendations for RLHF Training"

---

**Q: What's the timeline for completion?**
A: See Moonshine-Phase-Roadmap.md "High-Level Timeline"  
→ 12 weeks total, ~100 hours (part-time pace)

---

**Q: What's the biggest finding from the analysis?**
A: See Moonshine-Analysis-Findings.md section "1.1 Core Finding: Semantic Density vs. Information Gain"  
→ Summary in Quick-Reference.md "Finding 1: Sparse > Dense"

---

**Q: Is my data good enough to train on?**
A: Yes, see Moonshine-Analysis-Findings.md section "Quality Score Distributions"  
→ 48.7% high-signal, mean compliance 0.215 (low sycophancy = good)

---

**Q: What should I do this week?**
A: See Moonshine-Quick-Reference.md section "Next Immediate Actions"  
→ Detailed breakdown in Moonshine-Phase-Roadmap.md for current week

---

**Q: How do I actually code this?**
A: See Moonshine-Technical-Implementation.md section matching your current phase  
→ Pseudocode + schemas provided

---

**Q: What are the risks?**
A: See Moonshine-Phase-Roadmap.md "Risk Mitigation" table  
→ Mitigations provided for each risk

---

**Q: How will I know if I'm done?**
A: See Moonshine-Phase-Roadmap.md "Success Criteria" for each phase  
→ Summary in Moonshine-Quick-Reference.md "Weekly Progress Indicators"

---

## 📁 File Dependencies

```
Moonshine-Documentation-Index.md  [YOU ARE HERE]
    ↓
    ├─ Moonshine-Project-Overview.md
    │    ├─ References: Analysis-Findings.md
    │    └─ References: Phase-Roadmap.md
    │
    ├─ Moonshine-Analysis-Findings.md  [DETAILED FINDINGS]
    │    ├─ Informs: Technical-Implementation.md
    │    ├─ Informs: Quick-Reference.md
    │    └─ Informs: Phase-Roadmap.md
    │
    ├─ Moonshine-Technical-Implementation.md  [CODE REFERENCE]
    │    └─ Based on: Analysis-Findings.md recommendations
    │
    ├─ Moonshine-Phase-Roadmap.md  [PROJECT MANAGEMENT]
    │    └─ Based on: Technical-Implementation.md effort estimates
    │
    └─ Moonshine-Quick-Reference.md  [DAILY REFERENCE]
         └─ Summarizes: All above documents
```

---

## 🔄 Document Maintenance

### Version Control

- All docs stored in `moonshine/docs/` directory
- Versioned via git (tag: `docs_v1.0`, `docs_v1.1`, etc.)
- Weekly updates tracked in changelog

### Update Schedule

| Document | Frequency | Owner |
|----------|-----------|-------|
| Project-Overview.md | Monthly | Solo dev |
| Analysis-Findings.md | As new analyses complete | Solo dev |
| Technical-Implementation.md | As code evolves | Solo dev |
| Phase-Roadmap.md | Weekly (progress tracking) | Solo dev |
| Quick-Reference.md | Weekly | Solo dev |

### How to Update

1. Edit markdown file directly
2. Commit to git with message: `docs: update [filename] - [change summary]`
3. Tag if version bump: `git tag docs_v1.1`
4. Update this index if new sections added

---

## 📞 Support & Questions

**For strategic questions:**
→ Refer to Moonshine-Project-Overview.md or Moonshine-Analysis-Findings.md

**For implementation questions:**
→ Refer to Moonshine-Technical-Implementation.md

**For timeline/tracking questions:**
→ Refer to Moonshine-Phase-Roadmap.md

**For quick lookup:**
→ Refer to Moonshine-Quick-Reference.md

**If you can't find answer:**
→ Check "Quick Question Lookup" section above

---

## 📊 Documentation Statistics

| Aspect | Value |
|--------|-------|
| Total documents | 6 files |
| Total words | ~20,000 words |
| Total sections | 50+ |
| Code examples | 20+ |
| Diagrams/tables | 30+ |
| Time to read all | ~4 hours |
| Time to read essential | ~1.5 hours |
| Last updated | February 15, 2026 |
| Status | ✅ Production ready |

---

## 🚀 Next Steps After Reading

1. **Read:** Moonshine-Project-Overview.md (30 min)
2. **Read:** Moonshine-Quick-Reference.md (15 min)
3. **Decide:** Which phase to start with (Phase 1 = Week 1-2)
4. **Create:** `moonshine/` GitHub repo
5. **Follow:** Moonshine-Phase-Roadmap.md for detailed execution
6. **Reference:** Moonshine-Technical-Implementation.md while coding
7. **Update:** Moonshine-Quick-Reference.md weekly with progress

---

**Documentation Suite Status: ✅ COMPLETE**  
**Ready for: Phase 1 (Week 1-2) Foundation Layer Execution**  
**Last Review: February 15, 2026**

---

*This documentation is governed by the Sovereign Anti-Exploitation License v4.0. All recommendations and code artifacts are open-source, non-commercial friendly, and designed for democratizing AI capability development.*

---

## ðŸ”” LIVE STATE ADDENDUM (2026-02-18)

This suite remains valid, but the project has advanced beyond the original February 15 baseline.
To prevent drift, use the following **canonical-first reading contract** before execution decisions.

### Canonical-First Read Order (Current)

1. `TOKEN_FORENSICS_README.md` (operator contract + lane rules)
2. `docs/RAW_ONLY_ENFORCEMENT_20260218.md` (implementation record)
3. `reports/token_ledger.json` (canonical source-locked token truth)
4. `reports/token_forensics.json` (current corpus + distillation summary)
5. `reports/canonical/parquet_forensics.raw.md` (raw parquet profile)
6. `reports/moonshine_corpus_report.md` (heuristic analyzer report; useful but scope-limited)

### Scope Guardrail (Mandatory)

Do not compare token totals without scope tags:

| Scope Tag | Source | Current Value |
|-----------|--------|---------------|
| `heuristic_analyzer` | `reports/moonshine_corpus_report.md` | 51,524,462 |
| `canonical_source_locked` | `reports/token_ledger.json` | 115,330,530 |
| `distilled_selected` | `reports/token_ledger.json` | 104,321,772 |
| `parquet_row_aggregate` | `reports/canonical/parquet_forensics.raw.json` | 231,608,618 |

### Lane Contract (Current)

| Lane | Purpose | Path |
|------|---------|------|
| Canonical | Real ChatGPT export artifacts | `reports/canonical/` |
| Legacy Synthetic | Quarantined synthetic artifacts only | `reports/legacy_synthetic/` |
| Expansion | Isolated exploratory runs | `reports/expansion_20260218/` |

### Documentation Drift Control

- Treat this file as navigation, not metric truth.
- Metric truth lives in canonical reports under `reports/`.
- When canonical counters change, update:
  1. `TOKEN_FORENSICS_README.md`
  2. `docs/Moonshine-Project-Overview.md`
  3. `PROJECT_MOONSHINE_UPDATE_1.md` (append-only update block)

---

**Addendum Status:** âœ… Active  
**Effective Date:** 2026-02-18  
**Reason:** Project advanced beyond original documentation baseline; canonical-first control re-established.

### New Planning Anchor (Phase 2)

- `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`
  - Multi-provider onboarding contract
  - Provider-per-run DB policy
  - Rolling mash DB archive strategy
  - Claude Wave 1 execution checklist

---

## ðŸ“¦ Reports Authority Map (2026-02-19)

This section defines what in `reports/` is authoritative vs exploratory vs legacy.

### A. Primary Database Authority

| Artifact | Timestamp | Role | Authority |
|----------|-----------|------|-----------|
| `reports/moonshine_corpus.db` | 2026-02-17 | Legacy canonical snapshot from Stage B repair run | Historical baseline |
| `reports/expansion_20260218/moonshine_corpus.db` | 2026-02-18 | Latest full rerun DB (same source hash; newer run_id) | **Current working DB for active analysis** |

**Operator note:** until `reports/main/moonshine_mash_active.db` is introduced, treat `reports/expansion_20260218/moonshine_corpus.db` as the latest operational DB.

### B. Parquet Role Clarification

| Artifact | Size (approx) | Purpose | Not Comparable With |
|----------|---------------|---------|---------------------|
| `reports/dedup_clusters.parquet` | 36.7 MB | Row/cluster mapping for dedup signals | token-count parquet totals |
| `reports/canonical/token_row_metrics.raw.parquet` | 16.8 MB | Canonical token metrics by row | dedup cluster parquet |

Different parquet files encode different feature families; size difference is expected and non-diagnostic by itself.

### C. Reports Folder Cleanup Priority (Execution Order)

1. Establish `reports/main/` and create `moonshine_mash_active.db` (new authoritative merge target).
2. Move/copy old root DB and reports into archive run folders before any provider merge.
3. Keep `canonical/` strictly for raw-only token forensics artifacts.
4. Keep `legacy_synthetic/` strictly quarantined.
5. Keep `expansion_*/` as exploratory lanes until promoted via manifested merge.
6. Emit a `reports_authority_manifest.json` on each cycle describing active authority pointers.

### D. Required Read Set Before Any Merge

1. `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`
2. `reports/token_ledger.json`
3. `reports/expansion_20260218/token_ledger.json`
4. `reports/raw_only_gate_manifest.json`

---

**Authority Map Status:** âœ… Active  
**Last Audit Date:** 2026-02-19

## âœ… Operational Cleanup Applied (2026-02-19)

The authority-lane bootstrap has been executed.

### Active Now

- `reports/main/moonshine_mash_active.db` (authoritative mash DB)
- `reports/main/token_ledger.main.json`
- `reports/main/merge_manifest.main.json`
- `reports/main/reports_authority_manifest.json`

### Archived from Root Legacy Snapshot

Moved to `archive/chatgpt/chatgpt_20260217_r1/`:

- `moonshine_corpus.legacy_root_20260217.db`
- `moonshine_corpus_report.legacy_root_20260217.md`
- `moonshine_distillation_manifest.legacy_root_20260217.json`

### Operator Rule (Effective Immediately)

For DB queries and dataset build operations, use `reports/main/moonshine_mash_active.db` unless a task explicitly targets a historical lane.


---

## Live Documentation Map Addendum (2026-02-27)

This addendum re-centers the index on the active mixed-provider state and current docs directory reality.

### Current Docs Tree Authority

- Canonical docs tree snapshot: `docs/docs-tree.md`
- Includes handoff packet directory: `docs/handoffs/20260226_attack_plan/`
- Use docs-tree as file-presence authority when reconciling references.

### Active Read Order (Current State)

1. `TOKEN_FORENSICS_README.md`
2. `docs/MOONSHINE_PHASE_2_MULTI_PROVIDER_PLAN_20260218.md`
3. `docs/QUERY_CONTRACTS_GATE_B.md`
4. `docs/Moonshine-Project-Overview.md`
5. `docs/Moonshine-Technical-Implementation.md`
6. `reports/main/merge_manifest.main.json`
7. `reports/main/token_recount.main.json`
8. `reports/CURRENT_REPORTS_FILETREE.md`

### Known Legacy Drift Notes

- Older sections in this index still reference planning-era assumptions and files like `Moonshine-Quick-Reference.md` that are not the current operational anchor.
- Treat this addendum plus the active read order above as the operative index contract.

### Core Narrative Preservation

Moonshine has expanded in execution surface area, but original core complexity and design intent are preserved:

- non-destructive source discipline,
- provenance-locked analytics,
- staged distillation contracts,
- archive-first merge governance,
- query-first operational DB usage.


## Public Wiki Linkage Addendum (2026-02-27)

For public teaser operations, include root wiki surface in the index chain:

1. `README.md`
2. `WIKI.md`
3. `PROJECT_DATABASE_DOCUMENTATION.md`
4. `docs/Moonshine-Documentation-Index.md`

`WIKI.md` is intended as a concise external navigation layer while this index remains the full internal documentation map.

## Main-Lane Evidence Addendum (2026-02-27)

For the completed qwen/deepseek promotion wave, use this evidence chain:

1. `PROJECT_DATABASE_DOCUMENTATION.md`
2. `reports/main/final_db_pass_20260227.md`
3. `reports/main/final_db_pass_20260227.json`
4. `reports/main/token_recount.main.postdeps.json`
5. `reports/main/merge_manifest.main.json`

