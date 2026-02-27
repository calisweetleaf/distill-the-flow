# Somnus Seed Corpus Preparation

## Overview

We are taking the full ChatGPT export (01-07-2025 → 02-14-2026) and treating it as a **raw corpus**, not as a dataset and not as training material.

This export remains intact and unmodified.

It represents the complete conversational record, including:

- reasoning traces
- architectural discussions
- debugging workflows
- system evolution
- metadata
- corrections
- iterative refinement patterns

The export is considered **Point A**.

---

## Objective

Move from:

**Point A**  
Raw, monolithic JSON export containing all tokens, metadata, and conversational history, in a format that is:

- unstructured
- The idea is just like how OpenAI, Anthropic, and Google etc all have one mass of data upon which they distill DATSETS from not even models just yet.
- unorganized, unreadable. Must be transformed into another markdown extension document but still in json format, but with a different extension. The idea is to make it more readable and organized but still in json format. This is just a transformation of the original export, not a dataset construction or training formatting. It's just a way to make the original export more accessible and structured without losing any information or making any assumptions about how it will be used downstream.

to

**Point B**  
A structurally prepared, internally organized, entropy-analyzed corpus that is:

- cleaned of unusable noise
- categorized at a structural level
- preserved in canonical form
- reproducible
- non-destructive to original content

Point B is **not a dataset**.

Point B is **not formatted for training**.

Point B is simply a prepared canonical corpus state.

---

## Rules

1. The original export is never modified.
2. All transformations are derived copies.
3. No irreversible formatting decisions are made.
4. No training format assumptions are introduced.
5. No downstream pipeline assumptions are introduced.
6. The goal is structural readiness, not model readiness.

---

## Scope Boundary

This project ends at Point B.

Anything beyond Point B — including:

- dataset construction
- DPO formatting
- RLHF structuring
- model distillation
- adapter training
- synthetic generation

—is explicitly outside scope.

This document defines preparation only.

Nothing beyond preparation is assumed.
