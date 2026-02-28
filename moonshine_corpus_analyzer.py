#!/usr/bin/env python3
"""
Moonshine Corpus Analyzer - ChatGPT Export Analysis
=====================================================

Analyzes YOUR actual ChatGPT conversations.json export and extracts:
- Conversation-level metadata (turns, duration, topics, artifacts)
- Moonshine-specific quality metrics (token_ratio, user_entropy, malicious_compliance, etc.)
- Correction events and DPO pair candidates
- Tone clustering and temporal evolution

Outputs:
- moonshine_corpus.db (SQLite, browsable)
- moonshine_corpus_report.md (human readable summary)

Usage:
    python moonshine_corpus_analyzer.py 02-14-26-ChatGPT/conversations.json

Part of Project Moonshine - Point B: Corpus Analysis
"""

import json
import hashlib
import re
import sqlite3
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from langdetect import detect, LangDetectException


@dataclass
class ConversationMetrics:
    """Per-conversation Moonshine metrics."""
    conversation_id: str
    title: str
    created_at: Optional[float]
    updated_at: Optional[float]
    
    # Turn counts
    total_turns: int
    user_turns: int
    assistant_turns: int
    duration_minutes: float
    
    # Token metrics
    user_tokens: int
    assistant_tokens: int
    token_ratio: float  # user / assistant
    total_tokens: int
    
    # Quality metrics
    user_entropy: float  # Shannon entropy of user vocab
    semantic_density: float  # unique_terms / total_terms
    information_gain: float  # estimated via response relevance
    repetition_score: float  # n-gram repetition rate
    tone_shift: float  # detected tone changes
    malicious_compliance: float  # sycophancy score
    
    # Topic and tone
    topic_primary: str
    topic_secondary: List[str]
    tone_cluster: str
    
    # Artifacts
    code_blocks: int
    terminal_outputs: int
    tables: int
    manifests: int
    
    # Correction events
    correction_events: int
    
    # Temporal
    period: int  # 1-5 temporal bucket


@dataclass
class DistillationPolicy:
    """Configurable quality thresholds for distillation."""
    min_information_gain: float = 0.40
    max_malicious_compliance: float = 0.35
    min_user_entropy: float = 0.40
    min_total_tokens: int = 100
    max_repetition_score: float = 0.60
    token_budget_min: int = 90_000_000
    token_budget_max: int = 110_000_000
    policy_version: str = "moonshine-distill-v1.0"
    policy_description: str = (
        "Select conversations with information_gain>=0.40, "
        "malicious_compliance<=0.35, user_entropy>=0.40, "
        "targeting 90M-110M token band (canonical scaled)."
    )

    def meets_quality_threshold(self, metrics: "ConversationMetrics") -> tuple:
        if metrics.information_gain < self.min_information_gain:
            return False, f"info_gain={metrics.information_gain:.3f}<{self.min_information_gain}"
        if metrics.malicious_compliance > self.max_malicious_compliance:
            return False, f"malicious_compliance={metrics.malicious_compliance:.3f}>{self.max_malicious_compliance}"
        if metrics.user_entropy < self.min_user_entropy:
            return False, f"user_entropy={metrics.user_entropy:.3f}<{self.min_user_entropy}"
        if metrics.total_tokens < self.min_total_tokens:
            return False, f"total_tokens={metrics.total_tokens}<{self.min_total_tokens}"
        if metrics.repetition_score > self.max_repetition_score:
            return False, f"repetition_score={metrics.repetition_score:.3f}>{self.max_repetition_score}"
        return True, (f"quality_gate_pass: info_gain={metrics.information_gain:.3f}; "
                      f"malicious_compliance={metrics.malicious_compliance:.3f}; "
                      f"entropy={metrics.user_entropy:.3f}")

    def compute_quality_tier(self, metrics: "ConversationMetrics") -> str:
        if (metrics.information_gain >= 0.65
                and metrics.malicious_compliance <= 0.15
                and metrics.correction_events > 0):
            return "gold"
        elif (metrics.information_gain >= 0.55
                and metrics.malicious_compliance <= 0.25):
            return "silver"
        return "bronze"


class MoonshineCorpusAnalyzer:
    """Analyzer for ChatGPT export corpus."""
    
    # Topic keyword patterns
    TOPIC_PATTERNS = {
        "debugging": ["error", "bug", "traceback", "fix", "debug", "issue", "exception", "fail", "crash"],
        "architecture": ["design", "arch", "system", "framework", "structure", "pattern", "component"],
        "rcf_theory": ["rcf", "recursive", "categorical", "frame", "eigenrecursion", "hsgm", "urst"],
        "rlhf_impl": ["rlhf", "dpo", "ppo", "reward", "reinforcement", "training", "fine-tune"],
        "code_review": ["review", "refactor", "improve", "clean", "optimize", "format"],
        "data_processing": ["data", "pipeline", "etl", "load", "process", "csv", "json", "parquet"],
        "deployment": ["deploy", "docker", "container", "production", "serve", "kubernetes", "k8s"],
        "meta": ["discuss", "talk about", "explain", "understand", "what is", "how does"]
    }
    
    # Tone indicators
    TONE_PATTERNS = {
        "clinical": ["verify", "validation", "implementation", "requirement", "analysis", "evaluation"],
        "collaborative": ["let's", "together", "we can", "our approach", "partner"],
        "conversational": ["what if", "maybe", "i think", "perhaps", "wondering"],
        "code_driven": ["run the code", "terminal", "execute", "output", "```"],
        "debugging": ["failed", "error", "fix", "because", "traceback", "debug"],
        "neutral": []  # Default
    }
    
    # Correction indicators
    CORRECTION_PATTERNS = [
        "no that's wrong", "let me rephrase", "actually", "i meant",
        "not quite", "revise", "rewrite", "change that", "fix this",
        "incorrect", "that's not right", "you misunderstood", "try again"
    ]
    
    # Sycophancy indicators (malicious compliance)
    SYCOPHANCY_PATTERNS = [
        "i agree", "you're right", "absolutely", "exactly", "precisely",
        "great point", "excellent", "perfect", "wonderful idea"
    ]
    
    def __init__(
        self,
        conversations_path: Path,
        output_dir: Path = Path("reports"),
        policy: Optional[DistillationPolicy] = None,
        source_sha256: Optional[str] = None,
        provider: str = "chatgpt",
        provider_run_id: Optional[str] = None,
    ):
        self.conversations_path = Path(conversations_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "moonshine_corpus.db"
        self.policy = policy or DistillationPolicy()
        self.source_sha256 = source_sha256 or self._hash_source_file()
        self.run_id = datetime.now(timezone.utc).strftime("moonshine_%Y%m%d_%H%M%S")

        normalized_provider = (provider or "chatgpt").strip().lower()
        if not normalized_provider:
            raise ValueError("provider must be non-empty")

        self.provider = normalized_provider
        self.provider_run_id = provider_run_id or self.run_id
        self.ingested_at = datetime.now(timezone.utc).isoformat()
        self.source_path = str(self.conversations_path).replace("\\", "/")
        self.tokenizer_name = "o200k_base"
        self.raw_json_tokens = 0
        self.content_tokens_source = 0
        self.content_tokens_source_origin = "uninitialized"

    def _hash_source_file(self) -> str:
        """Compute SHA256 of the source conversations file."""
        hasher = hashlib.sha256()
        with open(self.conversations_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_encoder(self):
        """Load the exact tokenizer required for provider-local token truth."""
        try:
            import tiktoken
        except Exception as exc:
            raise RuntimeError(
                f"{self.tokenizer_name} tokenizer unavailable; install tiktoken in the active .venv before running Moonshine analysis"
            ) from exc
        return tiktoken.get_encoding(self.tokenizer_name)

    def _compute_raw_json_tokens(self) -> int:
        """Count full-file tokens for the current normalized conversations payload."""
        encoder = self._get_encoder()
        raw_content = self.conversations_path.read_text(encoding="utf-8")
        return len(encoder.encode(raw_content, disallowed_special=()))

    def _compute_content_tokens_source(self, messages: List[Dict[str, Any]]) -> int:
        """Count exact non-system message tokens for the current provider-local corpus."""
        encoder = self._get_encoder()
        token_cache: Dict[str, int] = {}
        total = 0
        for message in messages:
            text = message.get("text") or ""
            if not text:
                continue
            cached = token_cache.get(text)
            if cached is None:
                cached = len(encoder.encode(text, disallowed_special=()))
                token_cache[text] = cached
            total += cached
        return total

    def _prime_exact_token_counters(self, messages: List[Dict[str, Any]]):
        """Populate exact token counters before distillation or ledger emission."""
        if not messages:
            raise RuntimeError(
                "Moonshine analysis extracted zero non-system messages; refusing to emit token ledgers/manifests with empty counters."
            )

        self.raw_json_tokens = self._compute_raw_json_tokens()
        self.content_tokens_source = self._compute_content_tokens_source(messages)
        self.content_tokens_source_origin = "exact_message_recount"

        if self.content_tokens_source <= 0:
            raise RuntimeError(
                "Exact provider-local content token recount returned zero; refusing to continue."
            )

    def analyze(self) -> Dict[str, Any]:
        """Run full corpus analysis."""
        print("="*70)
        print("MOONSHINE CORPUS ANALYZER")
        print("="*70)
        print(f"Input: {self.conversations_path}")
        print(f"Output: {self.output_dir}")
        print("-"*70)
        
        # Load conversations
        print("\n[LOAD] Loading conversations...")
        conversations = self._load_conversations()
        print(f"   Loaded {len(conversations)} conversations")
        
        # Sort by time for period bucketing
        conversations.sort(key=lambda x: x.get("create_time") or 0)
        
        # Analyze each conversation
        print("\n[ANALYZE] Analyzing conversations...")
        metrics_list = []
        all_messages = []
        
        total_convs = len(conversations)
        period_size = max(1, total_convs // 5)
        
        for idx, conv in enumerate(conversations):
            period = min(5, (idx // period_size) + 1)
            metrics, messages = self._analyze_conversation(conv, period)
            metrics_list.append(metrics)
            all_messages.extend(messages)
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{total_convs}...")
        
        print(f"   [OK] Analyzed {len(metrics_list)} conversations")
        print(f"   [OK] Extracted {len(all_messages)} messages")
        self._prime_exact_token_counters(all_messages)
        print(f"   [OK] Exact non-system tokens: {self.content_tokens_source:,}")
        
        # Build SQLite database
        print("\n[DB] Building SQLite database...")
        self._build_database(metrics_list, all_messages)
        print(f"   [OK] Database: {self.db_path}")

        # Apply distillation policy
        print("\n[DISTILL] Applying distillation policy...")
        distilled_metrics, distillation_manifest = self._apply_distillation_policy(metrics_list)
        self._write_distilled_table(distilled_metrics, distillation_manifest)
        self._write_distillation_manifest(distillation_manifest)
        self._update_token_ledger(distillation_manifest)
        self._enforce_phase2_schema_contract()
        print(f"   [OK] Distilled: {len(distilled_metrics)} conversations, "
              f"{distillation_manifest['distilled_tokens_selected']:,} tokens")

        # Generate report
        print("\n[REPORT] Generating report...")
        report_path = self._generate_report(metrics_list)
        print(f"   [OK] Report: {report_path}")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

        return {
            "conversations_analyzed": len(metrics_list),
            "messages_extracted": len(all_messages),
            "database": str(self.db_path),
            "report": str(report_path),
            "distilled_conversations": len(distilled_metrics),
            "distilled_tokens": distillation_manifest["distilled_tokens_selected"],
        }
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversations from JSON export."""
        with open(self.conversations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ChatGPT export is a list of conversations
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Handle dict format (node-based tree)
            return self._convert_tree_to_list(data)
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
    
    def _convert_tree_to_list(self, tree: Dict) -> List[Dict]:
        """Convert node-based tree to list of conversations."""
        conversations = []
        for node_id, node_data in tree.items():
            if "message" in node_data:
                conversations.append({
                    "id": node_id,
                    "title": f"Conversation {node_id}",
                    "mapping": {node_id: node_data}
                })
        return conversations
    
    def _analyze_conversation(self, conv: Dict, period: int) -> Tuple[ConversationMetrics, List[Dict]]:
        """Analyze a single conversation."""
        conv_id = conv.get("id", conv.get("conversation_id", "unknown"))
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time")
        update_time = conv.get("update_time")
        
        # Extract all messages from mapping
        messages = self._extract_messages(conv)
        
        # Separate user and assistant messages
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        
        # Compute duration
        duration = 0.0
        if create_time and update_time:
            duration = (update_time - create_time) / 60.0
        
        # Compute tokens (simple estimation: words * 1.3)
        user_text = " ".join(m["text"] for m in user_msgs)
        asst_text = " ".join(m["text"] for m in asst_msgs)
        user_tokens = int(len(user_text.split()) * 1.3)
        asst_tokens = int(len(asst_text.split()) * 1.3)
        total_tokens = user_tokens + asst_tokens
        token_ratio = user_tokens / max(asst_tokens, 1)
        
        # Compute quality metrics
        user_entropy = self._compute_entropy(user_text)
        semantic_density = self._compute_semantic_density(user_text + " " + asst_text)
        repetition_score = self._compute_repetition(asst_text)
        info_gain = self._estimate_information_gain(user_text, asst_text)
        
        # Detect tone
        all_text = user_text + " " + asst_text
        tone_cluster = self._detect_tone(all_text)
        
        # Detect corrections
        correction_events = self._count_corrections(messages)
        
        # Detect sycophancy
        malicious_compliance = self._compute_sycophancy(messages)
        
        # Extract topics
        topic_primary, topic_secondary = self._extract_topics(title + " " + all_text)
        
        # Count artifacts
        artifacts = self._count_artifacts(all_text)
        
        # Create metrics
        metrics = ConversationMetrics(
            conversation_id=conv_id,
            title=title,
            created_at=create_time,
            updated_at=update_time,
            total_turns=len(messages),
            user_turns=len(user_msgs),
            assistant_turns=len(asst_msgs),
            duration_minutes=duration,
            user_tokens=user_tokens,
            assistant_tokens=asst_tokens,
            token_ratio=token_ratio,
            total_tokens=total_tokens,
            user_entropy=user_entropy,
            semantic_density=semantic_density,
            information_gain=info_gain,
            repetition_score=repetition_score,
            tone_shift=self._compute_tone_shift(messages),
            malicious_compliance=malicious_compliance,
            topic_primary=topic_primary,
            topic_secondary=topic_secondary,
            tone_cluster=tone_cluster,
            code_blocks=artifacts["code_blocks"],
            terminal_outputs=artifacts["terminal_outputs"],
            tables=artifacts["tables"],
            manifests=artifacts["manifests"],
            correction_events=correction_events,
            period=period
        )
        
        # Tag messages with conversation info
        for m in messages:
            m["conversation_id"] = conv_id
            m["conversation_title"] = title
        
        return metrics, messages
    
    def _extract_messages(self, conv: Dict) -> List[Dict]:
        """Extract flat message list from conversation mapping."""
        messages = []
        mapping = conv.get("mapping", {})
        
        for node_id, node_data in mapping.items():
            msg_data = node_data.get("message", {})
            if not msg_data:
                continue
            
            author = msg_data.get("author", {})
            role = author.get("role", "unknown")
            
            if role == "system":
                continue
            
            content = msg_data.get("content", {})
            text = ""
            if isinstance(content.get("parts"), list):
                text = " ".join(str(p) for p in content["parts"] if p)
            elif isinstance(content, str):
                text = content
            
            if not text.strip():
                continue
            
            messages.append({
                "message_id": node_id,
                "role": role,
                "text": text,
                "create_time": msg_data.get("create_time"),
                "char_count": len(text),
                "word_count": len(text.split())
            })
        
        # Sort by create_time if available
        messages.sort(key=lambda x: x["create_time"] or 0)
        return messages
    
    def _compute_entropy(self, text: str) -> float:
        """Compute Shannon entropy of text (normalized 0-1)."""
        if not text:
            return 0.0
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        total = len(words)
        
        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in word_counts.values()
        )
        
        # Normalize by log2 of unique word count
        max_entropy = math.log2(len(word_counts)) if word_counts else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_semantic_density(self, text: str) -> float:
        """Compute concept density (unique terms / total terms)."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique = len(set(words))
        total = len(words)
        return unique / total
    
    def _compute_repetition(self, text: str) -> float:
        """Compute n-gram repetition rate."""
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if not trigrams:
            return 0.0
        
        unique = len(set(trigrams))
        total = len(trigrams)
        return 1 - (unique / total)
    
    def _estimate_information_gain(self, user_text: str, asst_text: str) -> float:
        """Estimate information gain via response relevance heuristics."""
        # Simple heuristic: measure overlap between user query words and assistant response
        user_words = set(user_text.lower().split())
        asst_words = set(asst_text.lower().split())
        
        if not user_words or not asst_words:
            return 0.5
        
        # Jaccard similarity (overlap / union)
        overlap = len(user_words & asst_words)
        union = len(user_words | asst_words)
        
        # Information gain is higher when there's relevant overlap but not too much
        similarity = overlap / union if union > 0 else 0
        
        # High similarity = on-topic but might be sycophantic
        # Low similarity = off-topic
        # Optimal is around 0.2-0.4 overlap
        if 0.15 <= similarity <= 0.45:
            return 0.6 + (0.4 - similarity)  # Higher gain for good overlap
        else:
            return 0.4 + similarity * 0.3  # Lower gain for poor overlap
    
    def _detect_tone(self, text: str) -> str:
        """Detect primary tone cluster."""
        text_lower = text.lower()
        scores = {}

        for tone, patterns in self.TONE_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            scores[tone] = score

        if not scores or max(scores.values()) == 0:
            return "neutral"

        return max(scores, key=scores.get)

    def _compute_tone_shift(self, messages: List[Dict]) -> float:
        """Compute mean tone transition rate across conversation turns.

        Returns 0.0 (stable) to 1.0 (every turn shifts tone cluster).
        Uses the same TONE_PATTERNS as _detect_tone().
        """
        if len(messages) < 2:
            return 0.0
        tone_labels = [self._detect_tone(m["text"]) for m in messages]
        transitions = sum(
            1 for i in range(1, len(tone_labels))
            if tone_labels[i] != tone_labels[i - 1]
        )
        return transitions / (len(tone_labels) - 1)
    
    def _count_corrections(self, messages: List[Dict]) -> int:
        """Count correction events in conversation."""
        corrections = 0
        
        for i, msg in enumerate(messages):
            if msg["role"] != "user" or i == 0:
                continue
            
            text_lower = msg["text"].lower()
            
            for pattern in self.CORRECTION_PATTERNS:
                if pattern in text_lower:
                    corrections += 1
                    break
        
        return corrections
    
    def _compute_sycophancy(self, messages: List[Dict]) -> float:
        """Compute malicious compliance (sycophancy) score."""
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        
        if not asst_msgs:
            return 0.0
        
        sycophantic_count = 0
        
        for msg in asst_msgs:
            text_lower = msg["text"].lower()
            
            # Count agreement phrases
            agreement_score = sum(1 for p in self.SYCOPHANCY_PATTERNS if p in text_lower)
            
            # Check for excessive agreement without substance
            if agreement_score >= 2:
                sycophantic_count += 1
        
        return sycophantic_count / len(asst_msgs)
    
    def _extract_topics(self, text: str) -> Tuple[str, List[str]]:
        """Extract primary and secondary topics."""
        text_lower = text.lower()
        scores = {}
        
        for topic, patterns in self.TOPIC_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            scores[topic] = score
        
        if not scores or max(scores.values()) == 0:
            return "general", []
        
        # Get primary (highest score)
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_topics[0][0]
        
        # Get secondary (any with score > 0 but not primary)
        secondary = [t for t, s in sorted_topics[1:] if s > 0]
        
        return primary, secondary[:3]  # Limit to top 3 secondary
    
    def _count_artifacts(self, text: str) -> Dict[str, int]:
        """Count code blocks, terminals, tables, manifests."""
        return {
            "code_blocks": text.count("```"),
            "terminal_outputs": len(re.findall(r'(?:>>>|\$|# Output:|Error:)', text)),
            "tables": 1 if "|" in text and text.count("|") > 4 else 0,
            "manifests": len(re.findall(r'(?:kind:|apiVersion:|metadata:|spec:)', text))
        }
    
    def _build_database(self, metrics_list: List[ConversationMetrics], messages: List[Dict]):
        """Build SQLite database with conversations and messages."""
        # Remove existing database
        if self.db_path.exists():
            self.db_path.unlink()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                total_turns INTEGER,
                user_turns INTEGER,
                assistant_turns INTEGER,
                duration_minutes REAL,
                user_tokens INTEGER,
                assistant_tokens INTEGER,
                token_ratio REAL,
                total_tokens INTEGER,
                user_entropy REAL,
                semantic_density REAL,
                information_gain REAL,
                repetition_score REAL,
                tone_shift REAL,
                malicious_compliance REAL,
                topic_primary TEXT,
                topic_secondary TEXT,
                tone_cluster TEXT,
                code_blocks INTEGER,
                terminal_outputs INTEGER,
                tables INTEGER,
                manifests INTEGER,
                correction_events INTEGER,
                period INTEGER
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE messages (
                message_id TEXT,
                conversation_id TEXT,
                conversation_title TEXT,
                role TEXT,
                text TEXT,
                create_time REAL,
                char_count INTEGER,
                word_count INTEGER,
                PRIMARY KEY (message_id, conversation_id)
            )
        """)
        
        # Create index
        cursor.execute("CREATE INDEX idx_conv_period ON conversations(period)")
        cursor.execute("CREATE INDEX idx_conv_topic ON conversations(topic_primary)")
        cursor.execute("CREATE INDEX idx_msg_conv ON messages(conversation_id)")
        
        # Insert conversations
        for m in metrics_list:
            cursor.execute("""
                INSERT INTO conversations (
                    conversation_id, title, created_at, updated_at,
                    total_turns, user_turns, assistant_turns, duration_minutes,
                    user_tokens, assistant_tokens, token_ratio, total_tokens,
                    user_entropy, semantic_density, information_gain,
                    repetition_score, tone_shift, malicious_compliance,
                    topic_primary, topic_secondary, tone_cluster,
                    code_blocks, terminal_outputs, tables, manifests,
                    correction_events, period
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.conversation_id, m.title, m.created_at, m.updated_at,
                m.total_turns, m.user_turns, m.assistant_turns, m.duration_minutes,
                m.user_tokens, m.assistant_tokens, m.token_ratio, m.total_tokens,
                m.user_entropy, m.semantic_density, m.information_gain,
                m.repetition_score, m.tone_shift, m.malicious_compliance,
                m.topic_primary, json.dumps(m.topic_secondary), m.tone_cluster,
                m.code_blocks, m.terminal_outputs, m.tables, m.manifests,
                m.correction_events, m.period
            ))
        
        # Insert messages
        for msg in messages:
            cursor.execute("""
                INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                msg["message_id"], msg["conversation_id"], msg["conversation_title"],
                msg["role"], msg["text"], msg["create_time"],
                msg["char_count"], msg["word_count"]
            ))
        
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Distillation methods (Stage B)
    # ------------------------------------------------------------------

    def _apply_distillation_policy(
        self, metrics_list: List[ConversationMetrics]
    ) -> Tuple[List[ConversationMetrics], Dict]:
        """Apply quality gate + token budget to produce distilled subset."""
        distilled_at = datetime.now(timezone.utc).isoformat()

        # Step 1: quality gate
        quality_passed = []
        quality_rejected = 0
        for m in metrics_list:
            ok, reason = self.policy.meets_quality_threshold(m)
            if ok:
                quality_passed.append(m)
            else:
                quality_rejected += 1

        # Step 2: scale heuristic tokens into exact provider-local token space
        heuristic_total = sum(m.total_tokens for m in metrics_list)
        if self.content_tokens_source <= 0:
            raise RuntimeError("content_tokens_source must be primed before distillation")
        scale = self.content_tokens_source / max(heuristic_total, 1)

        # Step 3: sort by information_gain DESC, accumulate up to budget_max
        quality_passed.sort(key=lambda m: m.information_gain, reverse=True)
        selected = []
        accumulated = 0
        budget_trimmed = 0
        for m in quality_passed:
            canonical_tokens = int(m.total_tokens * scale)
            if accumulated + canonical_tokens > self.policy.token_budget_max:
                budget_trimmed += 1
                continue
            selected.append(m)
            accumulated += canonical_tokens

        distilled_tokens = accumulated
        if self.policy.token_budget_min <= distilled_tokens <= self.policy.token_budget_max:
            budget_status = "in_band"
        elif distilled_tokens < self.policy.token_budget_min:
            budget_status = "under_band"
        else:
            budget_status = "over_band"

        manifest = {
            "version": "1.0.0",
            "run_id": self.run_id,
            "distillation_timestamp": distilled_at,
            "source_sha256": self.source_sha256,
            "policy_version": self.policy.policy_version,
            "policy_description": self.policy.policy_description,
            "policy_thresholds": {
                "min_information_gain": self.policy.min_information_gain,
                "max_malicious_compliance": self.policy.max_malicious_compliance,
                "min_user_entropy": self.policy.min_user_entropy,
                "min_total_tokens": self.policy.min_total_tokens,
                "max_repetition_score": self.policy.max_repetition_score,
                "token_budget_min": self.policy.token_budget_min,
                "token_budget_max": self.policy.token_budget_max,
            },
            "input_conversations": len(metrics_list),
            "quality_passed": len(quality_passed),
            "quality_rejected": quality_rejected,
            "budget_trimmed": budget_trimmed,
            "distilled_conversations": len(selected),
            "distilled_tokens_selected": distilled_tokens,
            "budget_status": budget_status,
            "content_tokens_source": self.content_tokens_source,
            "content_tokens_source_origin": self.content_tokens_source_origin,
            "distilled_fraction": (
                distilled_tokens / self.content_tokens_source
                if self.content_tokens_source > 0 else 0.0
            ),
            "heuristic_scale_factor": round(scale, 6),
        }
        return selected, manifest

    def _write_distilled_table(
        self, distilled_metrics: List[ConversationMetrics], manifest: Dict
    ):
        """Create distilled_conversations table in existing DB."""
        distilled_at = manifest["distillation_timestamp"]
        heuristic_total = sum(
            m.total_tokens for m in distilled_metrics
        )
        # Recompute scale from full corpus (need all metrics accessible via DB)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(total_tokens) FROM conversations")
        full_heuristic = cursor.fetchone()[0] or 1
        scale = self.content_tokens_source / full_heuristic

        cursor.execute("DROP TABLE IF EXISTS distilled_conversations")
        cursor.execute("""
            CREATE TABLE distilled_conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                total_turns INT,
                user_turns INT,
                assistant_turns INT,
                total_tokens INT,
                user_tokens INT,
                assistant_tokens INT,
                token_ratio REAL,
                information_gain REAL,
                malicious_compliance REAL,
                user_entropy REAL,
                semantic_density REAL,
                repetition_score REAL,
                correction_events INT,
                topic_primary TEXT,
                tone_cluster TEXT,
                period INT,
                source_hash TEXT NOT NULL,
                distilled_at TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                run_id TEXT NOT NULL,
                quality_tier TEXT NOT NULL,
                inclusion_reason TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX idx_distilled_topic ON distilled_conversations(topic_primary)"
        )
        cursor.execute(
            "CREATE INDEX idx_distilled_period ON distilled_conversations(period)"
        )
        cursor.execute(
            "CREATE INDEX idx_distilled_tier ON distilled_conversations(quality_tier)"
        )
        cursor.execute(
            "CREATE INDEX idx_distilled_info_gain ON distilled_conversations(information_gain DESC)"
        )

        for m in distilled_metrics:
            _, reason = self.policy.meets_quality_threshold(m)
            tier = self.policy.compute_quality_tier(m)
            canonical_tokens = int(m.total_tokens * scale)
            canonical_user = int(m.user_tokens * scale)
            canonical_asst = int(m.assistant_tokens * scale)
            cursor.execute("""
                INSERT INTO distilled_conversations VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
            """, (
                m.conversation_id, m.title, m.created_at, m.updated_at,
                m.total_turns, m.user_turns, m.assistant_turns,
                canonical_tokens, canonical_user, canonical_asst,
                m.token_ratio, m.information_gain, m.malicious_compliance,
                m.user_entropy, m.semantic_density, m.repetition_score,
                m.correction_events, m.topic_primary, m.tone_cluster, m.period,
                self.source_sha256, distilled_at,
                self.policy.policy_version, self.run_id, tier, reason,
            ))

        conn.commit()
        conn.close()

    def _write_distillation_manifest(self, manifest: Dict):
        """Write distillation manifest JSON to output dir."""
        manifest_path = self.output_dir / "moonshine_distillation_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def _update_token_ledger(self, manifest: Dict):
        """Update token_ledger.json with exact provider-local counters and distilled totals."""
        ledger_path = self.output_dir / "token_ledger.json"
        if ledger_path.exists():
            with open(ledger_path, 'r', encoding='utf-8') as f:
                ledger = json.load(f)
        else:
            ledger = {
                "version": "1.0.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_sha256": self.source_sha256,
                "source_path": str(self.conversations_path),
                "tokenizer": self.tokenizer_name,
                "parsing_scope": "message_content_non_system",
                "counters": {
                    "canonical_tokens": self.raw_json_tokens,
                    "distilled_tokens_excluded": None,
                    "raw_json_tokens": self.raw_json_tokens,
                    "content_tokens_non_system": self.content_tokens_source,
                    "content_tokens_cleaned": None,
                    "distilled_tokens_selected": None,
                },
                "notes": {
                    "raw_json_tokens": "Full file tokenization including JSON structure",
                    "content_tokens_non_system": "Exact o200k_base count over non-system message content for this provider-local corpus",
                    "content_tokens_cleaned": "Post-policy filtering (computed by distillation stage)",
                    "distilled_tokens_selected": "Final training/query subset (computed by distillation stage)",
                },
                "run_id": self.run_id,
            }

        ledger["provider"] = self.provider
        ledger["provider_run_id"] = self.provider_run_id
        ledger["tokenizer"] = self.tokenizer_name
        ledger["parsing_scope"] = "message_content_non_system"
        ledger["source_path"] = self.source_path
        ledger["source_sha256"] = self.source_sha256
        ledger["content_tokens_source_origin"] = self.content_tokens_source_origin

        counters = ledger.setdefault("counters", {})
        counters["canonical_tokens"] = self.raw_json_tokens
        counters["raw_json_tokens"] = self.raw_json_tokens
        counters["content_tokens_non_system"] = self.content_tokens_source
        counters["content_tokens_cleaned"] = manifest["distilled_tokens_selected"]
        counters["distilled_tokens_selected"] = manifest["distilled_tokens_selected"]
        counters["distilled_tokens_excluded"] = max(
            self.content_tokens_source - manifest["distilled_tokens_selected"],
            0,
        )

        notes = ledger.setdefault("notes", {})
        notes["raw_json_tokens"] = "Full file tokenization including JSON structure"
        notes["content_tokens_non_system"] = "Exact o200k_base count over non-system message content for this provider-local corpus"
        notes["content_tokens_cleaned"] = "Post-policy filtering (computed by distillation stage)"
        notes["distilled_tokens_selected"] = "Final training/query subset (computed by distillation stage)"

        ledger["distillation_run_id"] = manifest["run_id"]
        ledger["distillation_timestamp"] = manifest["distillation_timestamp"]
        with open(ledger_path, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, indent=2)

    def _enforce_phase2_schema_contract(self):
        """Backfill provider-aware schema fields required by Phase 2 multi-provider merge."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        def _ensure_column(table: str, column: str, ddl_type: str):
            cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")

        for table in ["conversations", "messages", "distilled_conversations"]:
            _ensure_column(table, "provider", "TEXT")
            _ensure_column(table, "provider_run_id", "TEXT")
            _ensure_column(table, "source_file_sha256", "TEXT")
            _ensure_column(table, "source_path", "TEXT")
            _ensure_column(table, "ingested_at", "TEXT")
            _ensure_column(table, "record_uid", "TEXT")

        _ensure_column("messages", "conversation_record_uid", "TEXT")

        cursor.execute(
            """
            UPDATE conversations
            SET provider = COALESCE(provider, ?),
                provider_run_id = COALESCE(provider_run_id, ?),
                source_file_sha256 = COALESCE(source_file_sha256, ?),
                source_path = COALESCE(source_path, ?),
                ingested_at = COALESCE(ingested_at, ?),
                record_uid = COALESCE(record_uid, ? || ':' || ? || ':' || conversation_id)
            """,
            (
                self.provider,
                self.provider_run_id,
                self.source_sha256,
                self.source_path,
                self.ingested_at,
                self.provider,
                self.provider_run_id,
            ),
        )

        cursor.execute(
            """
            UPDATE messages
            SET provider = COALESCE(provider, ?),
                provider_run_id = COALESCE(provider_run_id, ?),
                source_file_sha256 = COALESCE(source_file_sha256, ?),
                source_path = COALESCE(source_path, ?),
                ingested_at = COALESCE(ingested_at, ?),
                conversation_record_uid = COALESCE(conversation_record_uid, ? || ':' || ? || ':' || conversation_id),
                record_uid = COALESCE(record_uid, ? || ':' || ? || ':' || conversation_id || ':' || message_id)
            """,
            (
                self.provider,
                self.provider_run_id,
                self.source_sha256,
                self.source_path,
                self.ingested_at,
                self.provider,
                self.provider_run_id,
                self.provider,
                self.provider_run_id,
            ),
        )

        cursor.execute(
            """
            UPDATE distilled_conversations
            SET provider = COALESCE(provider, ?),
                provider_run_id = COALESCE(provider_run_id, ?),
                source_file_sha256 = COALESCE(source_file_sha256, ?),
                source_path = COALESCE(source_path, ?),
                ingested_at = COALESCE(ingested_at, ?),
                record_uid = COALESCE(record_uid, ? || ':' || ? || ':' || conversation_id)
            """,
            (
                self.provider,
                self.provider_run_id,
                self.source_sha256,
                self.source_path,
                self.ingested_at,
                self.provider,
                self.provider_run_id,
            ),
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_provider ON conversations(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_provider_run ON conversations(provider, provider_run_id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_record_uid ON conversations(record_uid)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_provider ON messages(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_provider_run ON messages(provider, provider_run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv_record_uid ON messages(conversation_record_uid)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_msg_record_uid ON messages(record_uid)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_distilled_provider ON distilled_conversations(provider)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_distilled_provider_run ON distilled_conversations(provider, provider_run_id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_distilled_record_uid ON distilled_conversations(record_uid)")

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------

    def _generate_report(self, metrics_list: List[ConversationMetrics]) -> Path:
        """Generate markdown report summarizing the corpus."""
        report_path = self.output_dir / "moonshine_corpus_report.md"
        
        # Compute aggregates
        total_convs = len(metrics_list)
        total_user_turns = sum(m.user_turns for m in metrics_list)
        total_asst_turns = sum(m.assistant_turns for m in metrics_list)
        total_tokens = sum(m.total_tokens for m in metrics_list)
        
        avg_info_gain = sum(m.information_gain for m in metrics_list) / total_convs
        avg_malicious = sum(m.malicious_compliance for m in metrics_list) / total_convs
        avg_token_ratio = sum(m.token_ratio for m in metrics_list) / total_convs
        
        # Topic distribution
        topic_counts = Counter(m.topic_primary for m in metrics_list)
        
        # Tone distribution
        tone_counts = Counter(m.tone_cluster for m in metrics_list)
        
        # High-signal conversations (info gain > 0.58, malicious < 0.25)
        high_signal = [m for m in metrics_list if m.information_gain > 0.58 and m.malicious_compliance < 0.25]
        
        # Conversations with corrections
        with_corrections = [m for m in metrics_list if m.correction_events > 0]
        
        lines = [
            "# Moonshine Corpus Analysis Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Conversations Analyzed:** {total_convs:,}",
            f"**Total Messages:** {total_user_turns + total_asst_turns:,}",
            f"**Total Tokens:** {total_tokens:,}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"Your corpus contains **{total_convs:,} conversations** with **{total_tokens:,} tokens**.",
            f"**{len(high_signal):,} conversations ({len(high_signal)/total_convs:.1%})** are high-signal",
            f"(information gain > 0.58, low sycophancy).",
            f"**{len(with_corrections)} conversations** contain explicit correction events.",
            "",
            "### Key Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Average Information Gain | {avg_info_gain:.3f} |",
            f"| Average Malicious Compliance | {avg_malicious:.3f} |",
            f"| Average Token Ratio (User/Assistant) | {avg_token_ratio:.2f} |",
            f"| Total Correction Events | {sum(m.correction_events for m in metrics_list)} |",
            "",
            "## Topic Distribution",
            "",
            "| Topic | Count | Percentage |",
            "|-------|-------|------------|",
        ]
        
        for topic, count in topic_counts.most_common():
            lines.append(f"| {topic} | {count:,} | {count/total_convs:.1%} |")
        
        lines.extend([
            "",
            "## Tone Cluster Distribution",
            "",
            "| Tone | Count | Percentage |",
            "|------|-------|------------|",
        ])
        
        for tone, count in tone_counts.most_common():
            lines.append(f"| {tone} | {count:,} | {count/total_convs:.1%} |")
        
        lines.extend([
            "",
            "## Temporal Distribution (Periods 1-5)",
            "",
            "| Period | Conversations | Avg Info Gain |",
            "|--------|---------------|---------------|",
        ])
        
        for period in range(1, 6):
            period_convs = [m for m in metrics_list if m.period == period]
            if period_convs:
                avg_gain = sum(m.information_gain for m in period_convs) / len(period_convs)
                lines.append(f"| {period} | {len(period_convs)} | {avg_gain:.3f} |")
        
        lines.extend([
            "",
            "## Artifact Statistics",
            "",
            "| Artifact Type | Total Count |",
            "|---------------|-------------|",
            f"| Code Blocks | {sum(m.code_blocks for m in metrics_list):,} |",
            f"| Terminal Outputs | {sum(m.terminal_outputs for m in metrics_list):,} |",
            f"| Tables | {sum(m.tables for m in metrics_list):,} |",
            f"| Manifests | {sum(m.manifests for m in metrics_list):,} |",
            "",
            "## High-Signal Conversations (Recommended for Training)",
            "",
            f"**{len(high_signal)} conversations** meet the criteria:",
            "- Information gain > 0.58",
            "- Malicious compliance < 0.25",
            "",
            "| Conversation | Topic | Info Gain |",
            "|--------------|-------|-----------|",
        ])
        
        # Show top 20 high-signal conversations
        for m in sorted(high_signal, key=lambda x: x.information_gain, reverse=True)[:20]:
            lines.append(f"| {m.title[:50]}... | {m.topic_primary} | {m.information_gain:.3f} |")
        
        lines.extend([
            "",
            "## Conversations with Correction Events (DPO Candidates)",
            "",
            f"**{len(with_corrections)} conversations** contain explicit corrections.",
            "These are candidates for DPO (Direct Preference Optimization) training pairs.",
            "",
            "| Conversation | Corrections | Topic |",
            "|--------------|-------------|-------|",
        ])
        
        for m in sorted(with_corrections, key=lambda x: x.correction_events, reverse=True)[:15]:
            lines.append(f"| {m.title[:50]}... | {m.correction_events} | {m.topic_primary} |")
        
        lines.extend([
            "",
            "---",
            "",
            "## Database Schema",
            "",
            "The SQLite database (`moonshine_corpus.db`) contains two tables:",
            "",
            "### conversations",
            "Primary table with one row per conversation containing all metrics.",
            "",
            "### messages",  
            "Individual messages with conversation linkage.",
            "",
            "### Example Queries",
            "",
            "```sql",
            "-- Get high-signal debugging conversations from Period 4",
            "SELECT * FROM conversations",
            "WHERE topic_primary = 'debugging'",
            "  AND period = 4",
            "  AND information_gain > 0.58",
            "  AND malicious_compliance < 0.25;",
            "",
            "-- Get conversations with the most corrections",
            "SELECT title, correction_events, topic_primary",
            "FROM conversations",
            "WHERE correction_events > 0",
            "ORDER BY correction_events DESC;",
            "",
            "-- Get all messages from a specific conversation",
            "SELECT role, text FROM messages",
            "WHERE conversation_id = '<conv_id>'",
            "ORDER BY create_time;",
            "```",
            "",
            "---",
            "",
            "*Report generated by Moonshine Corpus Analyzer*",
        ])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return report_path


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Moonshine corpus analyzer")
    parser.add_argument("input_path", help="Path to provider export conversations JSON")
    parser.add_argument("output_dir", nargs="?", default="reports", help="Output directory")
    parser.add_argument("--provider", default="chatgpt", help="Provider label (chatgpt, claude, etc.)")
    parser.add_argument("--provider-run-id", default=None, help="Optional provider run identifier")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        raise SystemExit(1)

    policy = DistillationPolicy()
    analyzer = MoonshineCorpusAnalyzer(
        input_path,
        output_dir,
        policy=policy,
        provider=args.provider,
        provider_run_id=args.provider_run_id,
    )
    result = analyzer.analyze()

    print(f"\n[RESULTS] Results:")
    print(f"   Provider: {analyzer.provider}")
    print(f"   Provider Run ID: {analyzer.provider_run_id}")
    print(f"   Conversations: {result['conversations_analyzed']:,}")
    print(f"   Messages: {result['messages_extracted']:,}")
    print(f"   Database: {result['database']}")
    print(f"   Report: {result['report']}")


if __name__ == "__main__":
    main()
