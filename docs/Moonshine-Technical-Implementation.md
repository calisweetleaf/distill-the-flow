# Moonshine: Technical Implementation Guide

**Version:** 1.0  
**Phase:** Point B Analysis Pipeline  
**Last Updated:** February 15, 2026

---

## Table of Contents

1. [Architecture Overview](#architecture)
2. [Phase 1: Metadata Extraction](#phase1)
3. [Phase 2: Error & Correction Analysis](#phase2)
4. [Phase 3: Advanced Analytics](#phase3)
5. [Data Formats & Schemas](#schemas)
6. [Quality Assurance](#qa)
7. [Deployment & Scaling](#deployment)

---

## <a name="architecture"></a>Architecture Overview

### System Design

```
Raw Corpus (1,385 conversations - Just chatgpt first, pure estimate.)
    ↓
[Ingestion Layer] - Parse JSON export
    ↓
[Metadata Extraction] - Turn count, duration, topics
    ↓
[Segmentation] - Code/Prose/Meta tags
    ↓
[SQLite Index] - Searchable database
    ↓
[Analysis Pipelines]
    ├─ DPO Pair Generation (RLHF)
    ├─ Model Fingerprinting
    ├─ Tool Use Patterns
    ├─ Language Evolution
    └─ Dependency Graphs
    ↓
[Export Layer] - JSONL, CSV, Parquet
    ↓
[Training Datasets]
    ├─ Aeron fine-tuning (400M param)
    ├─ Rosemary training traces (RCF)
    └─ Distillation-resistant configs
```

### Key Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Ingestion** | Parse ChatGPT JSON export | `.json` file | Normalized records |
| **Indexer** | Build searchable metadata | Conversation records | SQLite DB + CSV |
| **Segmenter** | Classify message content | Raw text | Code/Prose/Meta tags |
| **DPO Generator** | Extract correction pairs | Tagged conversations | JSONL (DPO format) |
| **Fingerprinter** | Identify model variants | Response text + metadata | Model version tags |
| **Analyzer** | Compute temporal metrics | Tagged records | Time-series data |
| **Exporter** | Format for downstream use | Processed data | JSONL, Parquet, CSV |

---

## <a name="phase1"></a>Phase 1: Metadata Extraction

### 1.1 Conversation-Level Metadata

Extract from each conversation:

```python
metadata_schema = {
    "conversation_id": "str (UUID or sequence)",
    "title": "str",
    "created_at": "datetime",
    "updated_at": "datetime",
    "turn_count": "int (user + assistant messages)",
    "user_turns": "int",
    "assistant_turns": "int",
    "duration_minutes": "float",
    "total_tokens": "int (estimated)",
    "avg_user_message_length": "int (chars)",
    "avg_assistant_message_length": "int (chars)",
    "topic_primary": "str (inferred)",
    "topic_secondary": "list[str]",
    "artifact_count": {
        "code_blocks": "int",
        "terminal_outputs": "int",
        "tables": "int",
        "manifests": "int",
        "other": "int"
    },
    "correction_events": "int (manual or heuristic detected)",
    "has_profiling": "bool",
    "has_debugging": "bool",
    "information_gain": "float (0-1, from metrics analysis)",
    "malicious_compliance_score": "float (0-1, from analysis)",
    "tone_cluster": "str",
    "period": "int (1-5, temporal bucketing)"
}
```

### 1.2 Implementation Pseudocode

```python
import json
import sqlite3
from datetime import datetime
from collections import defaultdict

class MetadataExtractor:
    def __init__(self, export_path: str, db_path: str):
        self.export = json.load(open(export_path))
        self.db = sqlite3.connect(db_path)
        self.create_schema()
    
    def create_schema(self):
        """Create SQLite tables for indexed metadata."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME,
                turn_count INT,
                duration_minutes FLOAT,
                total_tokens INT,
                topic_primary TEXT,
                artifact_count INT,
                correction_events INT,
                information_gain REAL,
                tone_cluster TEXT,
                period INT
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,  -- 'user' or 'assistant'
                content TEXT,
                message_order INT,
                token_count INT,
                has_code BOOL,
                has_terminal BOOL,
                created_at DATETIME,
                FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                message_id TEXT,
                segment_type TEXT,  -- 'code', 'prose', 'meta', 'casual'
                content TEXT,
                token_count INT,
                language TEXT,  -- for code: 'python', 'yaml', 'json', etc.
                FOREIGN KEY(message_id) REFERENCES messages(message_id)
            )
        """)
    
    def extract_topics(self, conversation) -> tuple[str, list[str]]:
        """Infer primary and secondary topics from conversation."""
        title = conversation.get("title", "").lower()
        
        # Simple keyword-based classifier (upgrade to LLM later)
        topic_keywords = {
            "debugging": ["error", "bug", "traceback", "fix", "debug", "issue"],
            "architecture": ["design", "arch", "system", "framework", "structure"],
            "rcf_theory": ["rcf", "recursive", "categorical", "frame", "eigenrecursion"],
            "rlhf_impl": ["rlhf", "dpo", "ppo", "reward", "reinforcement"],
            "code_review": ["review", "refactor", "improve", "clean", "optimize"],
            "data_processing": ["data", "pipeline", "etl", "load", "process"],
            "deployment": ["deploy", "docker", "container", "production", "serve"],
            "meta": ["discuss", "talk about", "explain", "understand", "what is"]
        }
        
        primary = "general"
        secondary = []
        
        for topic, keywords in topic_keywords.items():
            if any(kw in title for kw in keywords):
                if not primary or primary == "general":
                    primary = topic
                else:
                    secondary.append(topic)
        
        return primary, secondary
    
    def count_artifacts(self, conversation) -> dict:
        """Count code blocks, terminals, tables, etc."""
        counts = defaultdict(int)
        
        for message in conversation.get("messages", []):
            content = message.get("content", "")
            
            # Count code blocks
            counts["code_blocks"] += content.count("```")
            
            # Count terminal indicators
            if any(x in content for x in ["$", ">>>", ">>>", "# Output:", "Error:"]):
                counts["terminal_outputs"] += 1
            
            # Count table indicators
            if "|" in content and content.count("|") > 4:
                counts["tables"] += 1
            
            # Count YAML/manifest indicators
            if any(x in content for x in ["kind:", "apiVersion:", "metadata:", "spec:"]):
                counts["manifests"] += 1
        
        return dict(counts)
    
    def detect_corrections(self, conversation) -> int:
        """Detect where user explicitly corrected the model."""
        correction_indicators = [
            "no that's wrong",
            "let me rephrase",
            "actually",
            "i meant",
            "not quite",
            "revise",
            "rewrite",
            "change that",
            "fix this",
            "incorrect"
        ]
        
        count = 0
        messages = conversation.get("messages", [])
        
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i > 0:
                content = msg.get("content", "").lower()
                if any(ind in content for ind in correction_indicators):
                    count += 1
        
        return count
    
    def extract_all(self):
        """Main extraction pipeline."""
        for conv in self.export.get("conversations", []):
            conv_id = conv.get("id", "unknown")
            title = conv.get("title", "Untitled")
            created_at = conv.get("create_time")
            
            messages = conv.get("messages", [])
            if not messages:
                continue
            
            # Compute metadata
            user_turns = sum(1 for m in messages if m.get("role") == "user")
            asst_turns = sum(1 for m in messages if m.get("role") == "assistant")
            turn_count = user_turns + asst_turns
            
            total_tokens = sum(len(m.get("content", "").split()) for m in messages)
            
            primary_topic, secondary_topics = self.extract_topics(conv)
            artifacts = self.count_artifacts(conv)
            corrections = self.detect_corrections(conv)
            
            # Insert into DB
            self.db.execute("""
                INSERT INTO conversations
                (conversation_id, title, created_at, turn_count, total_tokens,
                 topic_primary, artifact_count, correction_events)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (conv_id, title, created_at, turn_count, total_tokens,
                  primary_topic, sum(artifacts.values()), corrections))
            
            # Insert individual messages
            for order, msg in enumerate(messages):
                msg_id = f"{conv_id}_{order}"
                role = msg.get("role")
                content = msg.get("content", "")
                
                self.db.execute("""
                    INSERT INTO messages
                    (message_id, conversation_id, role, content, message_order)
                    VALUES (?, ?, ?, ?, ?)
                """, (msg_id, conv_id, role, content, order))
        
        self.db.commit()
        print(f"Indexed {len(self.export.get('conversations', []))} conversations")
```

### 1.3 Expected Output

After Phase 1 extraction:

```
moonshine/
├── data/
│   ├── moonshine.db  (SQLite index)
│   ├── metadata.csv  (conversation-level summary)
│   └── messages.parquet  (all messages + tags)
├── reports/
│   ├── phase1_summary.txt
│   └── topic_distribution.json
└── logs/
    └── extraction_2026-02-15.log
```

---

## <a name="phase2"></a>Phase 2: Error & Correction Analysis

### 2.1 DPO Pair Generation

Goal: Extract `(prompt, rejected, chosen)` tuples for RLHF training.

```python
dpo_pair_schema = {
    "prompt": "str (user message before correction)",
    "rejected": "str (model's initial response)",
    "chosen": "str (corrected/improved version)",
    "correction_type": "str enum: logic_error, syntax_error, incomplete, unclear, style",
    "confidence": "float (0-1, how confident pair is valid)",
    "source_conversation": "str (conv_id for traceability)",
    "message_indices": {
        "user_msg": "int",
        "assistant_msg": "int",
        "correction_msg": "int"
    }
}
```

### 2.2 Heuristic-Based Correction Detection

```python
class DPOPairGenerator:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.pairs = []
    
    def detect_correction_pattern(self, messages: list, start_idx: int) -> dict | None:
        """
        Detect pattern: [User] → [Assistant] → [User corrects] → [Assistant revised]
        """
        if start_idx + 3 >= len(messages):
            return None
        
        user_1 = messages[start_idx]
        asst_1 = messages[start_idx + 1]
        user_corr = messages[start_idx + 2]
        asst_2 = messages[start_idx + 3]
        
        # Validate roles
        if not (user_1["role"] == "user" and asst_1["role"] == "assistant" and
                user_corr["role"] == "user" and asst_2["role"] == "assistant"):
            return None
        
        # Check for correction indicators
        correction_text = user_corr.get("content", "").lower()
        correction_indicators = [
            "no that", "wrong", "let me", "actually", "i meant", 
            "not quite", "revise", "rewrite", "change", "fix", "incorrect"
        ]
        
        if not any(ind in correction_text for ind in correction_indicators):
            return None
        
        # Validate that correction references previous response
        prev_response = asst_1.get("content", "")
        if len(prev_response) < 50:  # Too short to be meaningful
            return None
        
        new_response = asst_2.get("content", "")
        if len(new_response) < 50:  # Too short to be meaningful
            return None
        
        # Calculate similarity (should be somewhat similar but not identical)
        similarity = self.text_similarity(prev_response, new_response)
        if similarity > 0.95:  # Too similar, probably just elaboration
            return None
        if similarity < 0.3:  # Too different, might not be correction
            return None
        
        return {
            "prompt": user_1.get("content"),
            "rejected": prev_response,
            "chosen": new_response,
            "correction_msg": user_corr.get("content"),
        }
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for quick filtering."""
        set1 = set(text1.split()[:100])  # First 100 words
        set2 = set(text2.split()[:100])
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0
    
    def infer_correction_type(self, rejected: str, chosen: str, 
                             correction_msg: str) -> str:
        """Classify type of correction."""
        correction_lower = correction_msg.lower()
        
        if any(x in correction_lower for x in ["syntax", "indentation", "bracket", "quote"]):
            return "syntax_error"
        
        if any(x in correction_lower for x in ["logic", "wrong", "incorrect", "error"]):
            return "logic_error"
        
        if len(chosen) > len(rejected) * 1.5:
            return "incomplete"
        
        if any(x in correction_lower for x in ["unclear", "confuse", "not clear"]):
            return "unclear"
        
        if any(x in correction_lower for x in ["style", "cleaner", "better", "improve"]):
            return "style"
        
        return "other"
    
    def generate_pairs(self, conversation: dict) -> list[dict]:
        """Extract all DPO pairs from a single conversation."""
        pairs = []
        messages = conversation.get("messages", [])
        
        i = 0
        while i < len(messages) - 3:
            pair = self.detect_correction_pattern(messages, i)
            
            if pair:
                pair["correction_type"] = self.infer_correction_type(
                    pair["rejected"],
                    pair["chosen"],
                    pair["correction_msg"]
                )
                pair["confidence"] = self.compute_confidence(pair)
                pairs.append(pair)
                i += 4  # Skip past this correction
            else:
                i += 1
        
        return pairs
    
    def compute_confidence(self, pair: dict) -> float:
        """Score confidence in DPO pair quality (0-1)."""
        score = 0.5  # Base score
        
        # Longer prompts are better
        prompt_len = len(pair["prompt"].split())
        if prompt_len > 20:
            score += 0.1
        
        # Significant difference in responses is good
        sim = self.text_similarity(pair["rejected"], pair["chosen"])
        if 0.3 < sim < 0.8:
            score += 0.2
        
        # Correction message is clear and explicit
        if len(pair["correction_msg"]) > 10:
            score += 0.1
        
        # Known high-signal correction types
        if pair["correction_type"] in ["logic_error", "incomplete"]:
            score += 0.1
        
        return min(score, 1.0)
    
    def export_pairs(self, output_path: str):
        """Export pairs in JSONL format for training."""
        import json
        
        with open(output_path, "w") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair) + "\n")
        
        print(f"Exported {len(self.pairs)} DPO pairs to {output_path}")
```

### 2.3 Manual Validation Step

For high-confidence pairs (confidence > 0.8):
- Automatic export to JSONL (ready for training)

For medium-confidence pairs (0.5 < confidence ≤ 0.8):
- Export to JSON with manual review flags
- Review 20% sample for quality assurance

For low-confidence pairs (confidence ≤ 0.5):
- Archive for future re-analysis with improved heuristics

---

## <a name="phase3"></a>Phase 3: Advanced Analytics

### 3.1 Model Version Fingerprinting

```python
class ModelFingerprinter:
    def __init__(self):
        self.signatures = {
            "gpt-4o": {
                "indicators": ["I appreciate", "I'd be happy", "😊", "collaboratively"],
                "emoji_freq": (0.02, 0.1),  # 2-10% emoji
                "avg_length": (150, 500),
                "formal_ratio": (0.1, 0.3),
            },
            "gpt-4.5": {
                "indicators": ["I'll", "structured", "step-by-step", "efficiency"],
                "emoji_freq": (0, 0.02),
                "avg_length": (200, 600),
                "formal_ratio": (0.3, 0.6),
            },
            "gpt-5-class": {
                "indicators": ["Let me think", "reasoning", "deeper", "consider"],
                "emoji_freq": (0, 0.01),
                "avg_length": (500, 2000),
                "formal_ratio": (0.5, 1.0),
            },
        }
    
    def fingerprint_message(self, content: str) -> tuple[str, float]:
        """Identify likely model variant + confidence."""
        scores = {}
        
        for model, sig in self.signatures.items():
            score = 0
            total_checks = 0
            
            # Check indicators
            for ind in sig["indicators"]:
                if ind.lower() in content.lower():
                    score += 1
            total_checks += len(sig["indicators"])
            
            # Check emoji frequency
            emoji_count = sum(1 for c in content if ord(c) > 127)
            emoji_freq = emoji_count / len(content) if content else 0
            freq_min, freq_max = sig["emoji_freq"]
            if freq_min <= emoji_freq <= freq_max:
                score += 1
            total_checks += 1
            
            # Normalize
            scores[model] = score / total_checks if total_checks > 0 else 0
        
        best_model = max(scores, key=scores.get)
        confidence = scores[best_model]
        
        return best_model, confidence
    
    def fingerprint_conversation(self, conversation: dict) -> dict:
        """Identify dominant model variant in conversation."""
        models = []
        confidences = []
        
        for msg in conversation.get("messages", []):
            if msg.get("role") == "assistant":
                model, conf = self.fingerprint_message(msg.get("content", ""))
                models.append(model)
                confidences.append(conf)
        
        if not models:
            return {"model": "unknown", "confidence": 0}
        
        # Take mode (most frequent) model
        from collections import Counter
        model_mode = Counter(models).most_common(1)[0][0]
        avg_conf = sum(confidences) / len(confidences)
        
        return {
            "model": model_mode,
            "confidence": avg_conf,
            "model_distribution": dict(Counter(models)),
        }
```

### 3.2 Tool Use Pattern Extraction

```python
class ToolUseAnalyzer:
    def __init__(self):
        self.patterns = []
    
    def analyze_artifact_usage(self, conversation: dict) -> dict:
        """Analyze how code/artifacts flow through conversation."""
        artifacts = {
            "code_requests": 0,
            "code_generations": 0,
            "execution_feedback": 0,
            "debugging_loops": 0,
            "total_artifacts": 0,
        }
        
        messages = conversation.get("messages", [])
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role")
            
            if "```" in content:
                artifacts["total_artifacts"] += 1
                
                if role == "user":
                    artifacts["code_requests"] += 1
                elif role == "assistant":
                    artifacts["code_generations"] += 1
            
            # Look for execution feedback (user posting output)
            if i > 0 and role == "user":
                prev_msg = messages[i-1]
                if "```" in prev_msg.get("content", "") and any(
                    x in content for x in ["Error:", "Output:", ">>>", "Traceback"]
                ):
                    artifacts["execution_feedback"] += 1
                    artifacts["debugging_loops"] += 1
        
        return artifacts
    
    def extract_workflows(self, conversations: list[dict]) -> list[dict]:
        """Identify common interaction patterns across conversations."""
        workflows = []
        
        for conv in conversations:
            usage = self.analyze_artifact_usage(conv)
            
            if usage["total_artifacts"] > 0:
                workflow = {
                    "conversation_id": conv.get("id"),
                    "title": conv.get("title"),
                    "artifact_usage": usage,
                    "workflow_type": self.classify_workflow(usage),
                }
                workflows.append(workflow)
        
        return workflows
    
    def classify_workflow(self, usage: dict) -> str:
        """Classify workflow type based on artifact patterns."""
        if usage["debugging_loops"] > 2:
            return "interactive_debugging"
        elif usage["code_generations"] > usage["code_requests"]:
            return "code_generation_focused"
        elif usage["code_requests"] > usage["code_generations"]:
            return "code_review_focused"
        else:
            return "mixed"
```

### 3.3 Temporal Language Evolution

```python
class LanguageEvolutionAnalyzer:
    def __init__(self):
        self.vocabulary = defaultdict(int)
        self.term_emergence = {}
    
    def extract_technical_terms(self, text: str) -> set[str]:
        """Extract technical terms (capitalized, CamelCase, underscored)."""
        import re
        
        terms = set()
        
        # CamelCase: RCF, HSGM, URST
        terms.update(re.findall(r'\b[A-Z]{2,}[a-z]?[A-Z]?[a-z]*\b', text))
        
        # Underscored: eigenrecursion, triaxial_backbone
        terms.update(re.findall(r'\b[a-z_]+_[a-z_]*\b', text))
        
        # All-caps: RCF, SOTA, RLHF
        terms.update(re.findall(r'\b[A-Z]{2,}\b', text))
        
        return terms
    
    def analyze_period_language(self, conversations: list[dict], 
                               period_starts: list[int]) -> dict:
        """Analyze vocabulary changes across time periods."""
        period_vocab = defaultdict(lambda: defaultdict(int))
        
        for period, start_idx in enumerate(period_starts):
            end_idx = period_starts[period + 1] if period + 1 < len(period_starts) else len(conversations)
            
            for conv in conversations[start_idx:end_idx]:
                for msg in conv.get("messages", []):
                    if msg.get("role") == "user":
                        terms = self.extract_technical_terms(msg.get("content", ""))
                        for term in terms:
                            period_vocab[period][term] += 1
        
        return {
            "vocabulary_by_period": dict(period_vocab),
            "new_terms_by_period": self.compute_term_emergence(period_vocab),
        }
    
    def compute_term_emergence(self, period_vocab: dict) -> dict:
        """Identify when each term first appeared."""
        emergence = {}
        all_terms = set()
        
        for period, vocab in period_vocab.items():
            all_terms.update(vocab.keys())
        
        for term in all_terms:
            for period in sorted(period_vocab.keys()):
                if term in period_vocab[period] and period_vocab[period][term] > 0:
                    emergence[term] = period
                    break
        
        return emergence
```

---

## <a name="schemas"></a>Data Formats & Schemas

### SQLite Schema

```sql
-- Main tables
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,
    created_at DATETIME,
    turn_count INT,
    duration_minutes FLOAT,
    total_tokens INT,
    topic_primary TEXT,
    artifact_count INT,
    correction_events INT,
    information_gain REAL,
    tone_cluster TEXT,
    period INT,
    model_version TEXT,
    model_confidence REAL
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    role TEXT,
    content TEXT,
    message_order INT,
    token_count INT,
    has_code BOOL,
    has_terminal BOOL,
    created_at DATETIME,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE segments (
    segment_id TEXT PRIMARY KEY,
    message_id TEXT,
    segment_type TEXT,
    content TEXT,
    token_count INT,
    language TEXT,
    FOREIGN KEY(message_id) REFERENCES messages(message_id)
);

CREATE TABLE dpo_pairs (
    pair_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    prompt TEXT,
    rejected TEXT,
    chosen TEXT,
    correction_type TEXT,
    confidence REAL,
    validated BOOL DEFAULT 0,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);
```

### JSONL Format (DPO Training)

```jsonl
{"prompt": "...", "rejected": "...", "chosen": "...", "correction_type": "logic_error", "confidence": 0.92}
{"prompt": "...", "rejected": "...", "chosen": "...", "correction_type": "incomplete", "confidence": 0.87}
```

### CSV Format (Metadata Summary)

```csv
conversation_id,title,turn_count,total_tokens,topic_primary,correction_events,information_gain,tone_cluster,period,model_version
conv_001,debugging RCF recursion,15,3240,debugging,2,0.61,clinical,2,gpt-4o
conv_002,RLHF pipeline design,22,5100,architecture,1,0.58,collaborative,3,gpt-4.5
```

---

## <a name="qa"></a>Quality Assurance

### Validation Checklist

Before exporting datasets:

- [ ] All DPO pairs have `confidence ≥ 0.7`
- [ ] Metadata completeness ≥ 95%
- [ ] No duplicate conversation IDs
- [ ] Token counts validated (sample check)
- [ ] Topic classifications manually spot-checked (10% sample)
- [ ] Correction events manually verified (20% sample)
- [ ] No PII or sensitive data leakage
- [ ] Reproducible pipeline (all random seeds fixed)

### Test Suite

```python
def test_metadata_extraction():
    """Verify metadata schema completeness."""
    assert all(key in metadata for key in required_keys)
    assert isinstance(metadata["turn_count"], int)
    assert 0 <= metadata["information_gain"] <= 1

def test_dpo_pair_quality():
    """Verify DPO pairs meet quality standards."""
    for pair in dpo_pairs:
        assert len(pair["prompt"]) > 10
        assert len(pair["rejected"]) > 10
        assert len(pair["chosen"]) > 10
        assert pair["confidence"] >= 0.7
        assert pair["correction_type"] in valid_types

def test_no_duplication():
    """Ensure no duplicate conversations or pairs."""
    conv_ids = [m["conversation_id"] for m in messages]
    assert len(conv_ids) == len(set(conv_ids))
```

---

## <a name="deployment"></a>Deployment & Scaling

### Single-Machine Pipeline

```bash
# Run full extraction locally
python moonshine/pipelines/extract_all.py \
    --input data/chatgpt_export.json \
    --output data/moonshine.db \
    --threads 4

# Generate DPO pairs
python moonshine/pipelines/dpo_generation.py \
    --db data/moonshine.db \
    --output data/dpo_pairs.jsonl \
    --min_confidence 0.7

# Export for training
python moonshine/pipelines/export.py \
    --db data/moonshine.db \
    --format pytorch \
    --output data/training_dataset/
```

### Distributed Scaling (Future)

For larger corpora:
- Partition conversations by ID ranges
- Process partitions on separate workers
- Merge results via SQLite federation
- Use async I/O for LLM fingerprinting

---

## Appendix: Dependencies

```
pandas
numpy
sqlite3 (bundled)
PyTorch
spaCy
transformers
tqdm
```

---

**Next Step:** See `Moonshine-Phase-Roadmap.md` for timeline and resource allocation.


---

## 8. Reports Folder Governance (2026-02-19)

### 8.1 Why This Exists

`reports/` now contains mixed artifact families (DB snapshots, canonical token forensics, dedup artifacts, exploratory reruns). Without explicit authority rules, operators can query stale DBs or compare incompatible parquet outputs.

### 8.2 Authority Rules

1. **Latest operational DB (current state):**
   - `reports/expansion_20260218/moonshine_corpus.db`
2. **Historical baseline DB:**
   - `reports/moonshine_corpus.db` (2026-02-17 snapshot)
3. **Canonical token-forensics lane only:**
   - `reports/canonical/*`
4. **Synthetic quarantine only:**
   - `reports/legacy_synthetic/*`

### 8.3 Parquet Schema Roles

#### Token Metrics Parquet
- Path: `reports/canonical/token_row_metrics.raw.parquet`
- Function: tokenizer/context-fit metrics by sample row
- Typical columns: token counts, context fit flags, truncation counts

#### Dedup Cluster Parquet
- Path: `reports/dedup_clusters.parquet`
- Function: exact/near/semantic dedup cluster assignments
- Typical columns: row identity, cluster ids/types, retention linkage

**Implementation implication:** these are not substitute datasets. They serve orthogonal analyses and should not be compared by file size.

### 8.4 Immediate Refactor Target (Phase 2)

Introduce explicit main authority lane:

```text
reports/main/
  moonshine_mash_active.db
  token_ledger.main.json
  merge_manifest.main.json
  reports_authority_manifest.json
```

### 8.5 Merge Contract (Provider Onboarding)

Before merging a new provider export:

1. Snapshot current main DB to `archive/main/<run_id>/`.
2. Ingest provider export into immutable provider-run DB.
3. Validate source hash + provenance ledger.
4. Merge into `reports/main/moonshine_mash_active.db` with idempotent upsert keys.
5. Update `reports_authority_manifest.json` to point downstream jobs to active DB.

### 8.6 Required Schema Additions for Multi-Provider DB

```sql
ALTER TABLE conversations ADD COLUMN provider TEXT;
ALTER TABLE conversations ADD COLUMN provider_run_id TEXT;
ALTER TABLE conversations ADD COLUMN source_file_sha256 TEXT;
ALTER TABLE conversations ADD COLUMN record_uid TEXT;

ALTER TABLE messages ADD COLUMN provider TEXT;
ALTER TABLE messages ADD COLUMN provider_run_id TEXT;
ALTER TABLE messages ADD COLUMN source_file_sha256 TEXT;
ALTER TABLE messages ADD COLUMN record_uid TEXT;
```

`record_uid` must be deterministic across reruns to guarantee idempotent merges.

### 8.7 Validation Command (Raw-Only Token Lane)

```powershell
python scripts/run_validation.py --reports-dir reports --strict --profile raw_only
```

This validates canonical token forensics lane; it does not replace DB merge validation for multi-provider mash.

---

**Governance Block Status:** âœ… Active  
**Effective Date:** 2026-02-19


### 8.8 Applied State Snapshot (2026-02-19)

Operational cleanup was executed and authority lane promoted.

```text
reports/main/
  moonshine_mash_active.db
  token_ledger.main.json
  merge_manifest.main.json
  reports_authority_manifest.json
```

Archived legacy root moonshine artifacts:

```text
archive/chatgpt/chatgpt_20260217_r1/
  moonshine_corpus.legacy_root_20260217.db
  moonshine_corpus_report.legacy_root_20260217.md
  moonshine_distillation_manifest.legacy_root_20260217.json
```

Implementation note:
- This was a bootstrap promotion from `reports/expansion_20260218/*` into `reports/main/*`.
- Next provider onboarding should use merge-upsert manifests with inserted/updated/skipped counts.


## 12. Main-Lane DPO Export Contract (2026-02-27)

This section supersedes earlier prototype assumptions for immediate production use.

### 12.1 Authoritative Source for Export

Use main DB as extraction source:

- `reports/main/moonshine_mash_active.db`

Do not export DPO pairs directly from raw JSON once rows are already normalized and provenance-tagged in main.

### 12.2 Export Unit

DPO pair candidate unit is a correction event trace resolved from message order:

- prior assistant response (`rejected`)
- user correction turn (correction signal)
- next assistant response (`chosen`)

Use `conversations` and `messages` linkage by `conversation_id`, preserving provider tags.

### 12.3 Ranking and Selection

Recommended ranking dimensions for candidate quality:

1. correction signal strength (explicit correction text)
2. response delta quality (rejected/chosen difference neither trivial nor unrelated)
3. provider provenance
4. quality context (`information_gain`, `malicious_compliance`, `correction_events`)

### 12.4 Balanced Initial Release Pack

For first mixed-provider DPO export:

- quota: `100` pairs total
- mix: `50` claude + `50` chatgpt
- rule: select highest-confidence valid pairs per provider
- output: JSONL with full provenance fields

### 12.5 Required Output Artifacts

- `reports/distillations/dpo/<run_id>/dpo_pairs_100.jsonl`
- `reports/distillations/dpo/<run_id>/dpo_manifest.json`
- `reports/distillations/dpo/<run_id>/selection_audit.md`

Manifest must include:

- source DB hash
- provider quotas and realized counts
- selection thresholds
- exclusion reasons
- timestamp and run id

### 12.6 Hard Rule

Main DB is the gold operational substrate for extraction. Display/render format can vary freely, but selection must remain provenance-locked to main records.


---

## 13. Schema Compliance Authority Addendum (2026-02-27)

This addendum locks schema compliance to the live merged system and resolves drift between planning-era pseudocode and production tables.

### 13.1 Source of Truth

- Schema authority document: `PROJECT_DATABASE_DOCUMENTATION.md`
- Live authority DB: `reports/main/moonshine_mash_active.db`
- Provider composition baseline: `chatgpt + claude`

### 13.2 Required Production Tables

The live schema contract is centered on three tables:

1. `conversations`
2. `messages`
3. `distilled_conversations`

Do not treat pseudocode-only legacy `segments` or early prototype layouts as canonical for current merge/export operations.

### 13.3 Provenance Envelope (Mandatory)

Rows participating in provider merges and downstream exports must preserve provenance metadata:

- `provider`
- `provider_run_id`
- `source_file_sha256`
- `source_path`
- `ingested_at`
- `record_uid`

`record_uid` uniqueness is a hard invariant for idempotent merge safety.

### 13.4 Compliance Checks Before Distillation Export

Before producing DPO/GRPO/SFT artifacts from main DB:

1. Verify table presence for `conversations`, `messages`, `distilled_conversations`.
2. Verify `record_uid` uniqueness in all three tables.
3. Verify provider composition and row counts against `reports/main/merge_manifest.main.json`.
4. Verify token authority against `reports/main/token_recount.main.json` (exact recount).
5. Emit export manifest with source DB hash and selection thresholds.

### 13.5 Scope Rule

Token totals and quality metrics must always be scope-labeled (heuristic vs canonical exact recount). Do not publish unlabeled token totals in release docs.

