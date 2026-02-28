"""
SOTA++ Token Forensics Agents

Seven specialized agents for production-grade dataset intelligence:
1. DataProfilerAgent - Schema, splits, sources, language distribution
2. MultiTokenizerAgent - Cross-tokenizer analysis and cost modeling
3. QualityScoringAgent - Readability, entropy, repetition, boilerplate
4. SafetyPIIAgent - PII detection and safety risk scoring
5. DedupAgent - Exact, near-dup, and semantic deduplication
6. CostModelAgent - Training and inference cost projections
7. VerifierAgent - Quality gates and integrity checks

Part of Project Decentralize SOTA - Drop 3: Dataset Intelligence Toolkit
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import warnings

import numpy as np
import pandas as pd

# Conditional imports with graceful fallbacks
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    warnings.warn("tiktoken not available - using fallback tokenizer")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not available - limited tokenizer support")

from token_forensics_orchestrator import BaseAgent, OrchestratorConfig


# ============================================================================
# AGENT 1: DATA PROFILER
# ============================================================================

class DataProfilerAgent(BaseAgent):
    """
    Builds comprehensive schema map, split map, source map, language map,
    and null/malformed diagnostics.
    """
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Profile dataset structure and metadata."""
        
        dataset_path = Path(shared_context["dataset_path"])
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract conversation tree structure
        conversations = self._extract_conversations(data)
        
        # Build profile
        profile = {
            "schema": self._build_schema_map(conversations),
            "splits": self._build_split_map(conversations),
            "sources": self._build_source_map(conversations),
            "languages": self._build_language_map(conversations),
            "diagnostics": self._build_diagnostics(conversations),
            "row_count": len(conversations),
            "sample_rows": conversations[:5]  # First 5 for inspection
        }
        
        # Emit artifacts
        self.emit_artifact("data_profile.json", profile)
        
        # Generate markdown report
        md_report = self._generate_markdown_report(profile)
        self.emit_artifact("data_profile.md", md_report, format="md")
        
        # Store conversations for downstream agents
        shared_context["conversations"] = conversations
        
        return profile
    
    def _extract_conversations(self, data: Any) -> List[Dict[str, Any]]:
        """Extract flattened conversation messages from ChatGPT export format.
        
        Handles two export formats:
        1. Mapping-style dict: {"node_id": {message, parent, children}} - direct tree
        2. List-style ChatGPT export: [{"mapping": {...}, "title": ...}, ...] - conversations list
        """
        
        conversations = []
        
        if isinstance(data, dict):
            if "mapping" in data or all(k in data for k in ["id", "title"]):
                pass
            elif all(isinstance(v, dict) and "message" in v for v in data.values() if isinstance(v, dict)):
                pass
            else:
                pass
        
        if isinstance(data, list):
            for conv in data:
                if isinstance(conv, dict) and "mapping" in conv:
                    conversations.extend(self._extract_from_mapping(conv.get("mapping", {}), conv.get("id", "unknown")))
        elif isinstance(data, dict):
            first_value = next(iter(data.values()), None) if data else None
            if first_value and isinstance(first_value, dict):
                if "mapping" in first_value or "title" in first_value:
                    pass
                elif "message" in first_value:
                    conversations.extend(self._extract_from_mapping(data, "direct_tree"))
                else:
                    for conv_id, conv_data in data.items():
                        if isinstance(conv_data, dict):
                            if "mapping" in conv_data:
                                conversations.extend(self._extract_from_mapping(conv_data.get("mapping", {}), conv_id))
                            elif "message" in conv_data:
                                conversations.extend(self._extract_from_mapping({conv_id: conv_data}, conv_id))
            else:
                conversations.extend(self._extract_from_mapping(data, "unknown"))
        
        return conversations
    
    def _extract_from_mapping(self, mapping: Dict[str, Any], conversation_id: str) -> List[Dict[str, Any]]:
        """Extract messages from a ChatGPT mapping tree structure."""
        
        messages = []
        
        for node_id, node_data in mapping.items():
            if not isinstance(node_data, dict):
                continue
                
            message = node_data.get("message", {})
            
            if not message:
                continue
            
            author = message.get("author", {})
            role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"
            
            content = message.get("content", {})
            
            text = ""
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if isinstance(parts, list):
                    text = " ".join(str(part) for part in parts if part)
                elif isinstance(content.get("text"), str):
                    text = content.get("text", "")
            elif isinstance(content, str):
                text = content
            
            if not text or not text.strip():
                continue
            
            messages.append({
                "sample_id": node_id,
                "conversation_id": conversation_id,
                "role": role,
                "text": text,
                "create_time": message.get("create_time"),
                "parent": node_data.get("parent"),
                "children": node_data.get("children", []),
                "raw_bytes": len(text.encode('utf-8')),
                "char_count": len(text),
            })
        
        return messages
    
    def _build_schema_map(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer schema from data structure."""
        
        if not conversations:
            return {"error": "No conversations found"}
        
        sample = conversations[0]
        
        return {
            "fields": list(sample.keys()),
            "field_types": {k: type(v).__name__ for k, v in sample.items()},
            "nullable_fields": [
                k for k in sample.keys() 
                if any(c.get(k) is None for c in conversations)
            ]
        }
    
    def _build_split_map(self, conversations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count messages by split (role)."""
        
        split_counts = Counter(c["role"] for c in conversations)
        
        return dict(split_counts)
    
    def _build_source_map(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map source attribution."""
        
        # For ChatGPT export, source is uniform
        return {
            "source": "ChatGPT Export",
            "export_format": "conversation_tree",
            "unique_sources": 1
        }
    
    def _build_language_map(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect language distribution (heuristic-based)."""
        
        # Simple heuristic: English if mostly ASCII
        def is_english(text: str) -> bool:
            try:
                text.encode('ascii')
                return True
            except UnicodeEncodeError:
                return False
        
        language_counts = Counter(
            "en" if is_english(c["text"]) else "other"
            for c in conversations
        )
        
        return {
            "detected_languages": dict(language_counts),
            "primary_language": language_counts.most_common(1)[0][0],
            "language_confidence": language_counts.most_common(1)[0][1] / len(conversations)
        }
    
    def _build_diagnostics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Null rate, malformed row detection."""
        
        total = len(conversations)
        
        null_counts = {
            field: sum(1 for c in conversations if c.get(field) is None)
            for field in conversations[0].keys()
        }
        
        # Malformed: missing text or invalid structure
        malformed = sum(
            1 for c in conversations 
            if not c.get("text") or c["char_count"] == 0
        )
        
        return {
            "null_counts": null_counts,
            "null_rates": {k: v/total for k, v in null_counts.items()},
            "malformed_count": malformed,
            "malformed_rate": malformed / total
        }
    
    def _generate_markdown_report(self, profile: Dict[str, Any]) -> str:
        """Generate human-readable markdown report."""
        
        report = [
            "# Data Profile Report",
            "",
            "## Dataset Overview",
            f"- **Total Rows:** {profile['row_count']:,}",
            f"- **Source:** {profile['sources']['source']}",
            f"- **Primary Language:** {profile['languages']['primary_language']} ({profile['languages']['language_confidence']:.1%} confidence)",
            "",
            "## Schema",
            "```json",
            json.dumps(profile['schema'], indent=2),
            "```",
            "",
            "## Splits",
            ""
        ]
        
        for split, count in profile['splits'].items():
            report.append(f"- **{split}:** {count:,} ({count/profile['row_count']:.1%})")
        
        report.extend([
            "",
            "## Diagnostics",
            f"- **Malformed Rows:** {profile['diagnostics']['malformed_count']} ({profile['diagnostics']['malformed_rate']:.1%})",
            "",
            "### Null Rates by Field",
            ""
        ])
        
        for field, rate in profile['diagnostics']['null_rates'].items():
            if rate > 0:
                report.append(f"- `{field}`: {rate:.1%}")
        
        return "\n".join(report)


# ============================================================================
# AGENT 2: MULTI-TOKENIZER
# ============================================================================

class MultiTokenizerAgent(BaseAgent):
    """
    Compute per-row tokenization using multiple tokenizers.
    Report drift across tokenizers and context-fit statistics.
    """
    
    def __init__(self, config: OrchestratorConfig):
        super().__init__(config)
        self.tokenizers = self._init_tokenizers()
    
    def _init_tokenizers(self) -> Dict[str, Any]:
        """Initialize multiple tokenizers."""
        tokenizers = {}
        
        # OpenAI (tiktoken)
        if HAS_TIKTOKEN:
            try:
                tokenizers["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
                tokenizers["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception as e:
                print(f"⚠️  tiktoken init failed: {e}")
        
        # Llama (SentencePiece via transformers)
        if HAS_TRANSFORMERS:
            try:
                tokenizers["llama-2-7b"] = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            except Exception as e:
                print(f"⚠️  Llama tokenizer init failed: {e}")
        
        # Fallback: whitespace tokenizer
        if not tokenizers:
            print("⚠️  No production tokenizers available. Using fallback.")
            tokenizers["whitespace"] = "fallback"
        
        return tokenizers
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize all conversations with multiple tokenizers."""
        
        conversations = shared_context.get("conversations", [])
        
        if not conversations:
            raise ValueError("No conversations in shared context. DataProfiler must run first.")
        
        # Tokenize each row with each tokenizer
        tokenization_results = []
        
        for conv in conversations:
            text = conv["text"]
            row_result = {
                "sample_id": conv["sample_id"],
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "char_count": len(text)
            }
            
            for tokenizer_name, tokenizer in self.tokenizers.items():
                token_count = self._tokenize(text, tokenizer, tokenizer_name)
                row_result[f"tokens_{tokenizer_name}"] = token_count
            
            # Context fit analysis (assuming 4k, 8k, 32k, 128k windows)
            primary_tokenizer = list(self.tokenizers.keys())[0]
            token_count = row_result.get(f"tokens_{primary_tokenizer}", 0)
            
            row_result.update({
                "context_4k_fit": token_count <= 4096,
                "context_8k_fit": token_count <= 8192,
                "context_32k_fit": token_count <= 32768,
                "context_128k_fit": token_count <= 131072,
                "truncation_at_4k": max(0, token_count - 4096),
                "truncation_at_8k": max(0, token_count - 8192),
                "truncation_at_32k": max(0, token_count - 32768),
            })
            
            tokenization_results.append(row_result)
        
        # Aggregate statistics
        df = pd.DataFrame(tokenization_results)
        
        aggregate_stats = {
            "row_count": len(df),
            "tokenizer_stats": self._compute_tokenizer_stats(df),
            "context_fit_stats": self._compute_context_fit_stats(df),
            "drift_analysis": self._analyze_tokenizer_drift(df)
        }
        
        # Emit artifacts
        self.emit_artifact("tokenization_results.json", {
            "per_row": tokenization_results,
            "aggregate": aggregate_stats
        })
        
        # Save per-row metrics as parquet
        output_path = self.config.output_dir / "token_row_metrics.parquet"
        df.to_parquet(output_path, index=False)
        print(f"📊 {self.name} → token_row_metrics.parquet")
        
        # Tokenizer benchmark CSV
        benchmark_df = self._create_tokenizer_benchmark(df)
        benchmark_path = self.config.output_dir / "tokenizer_benchmark.csv"
        benchmark_df.to_csv(benchmark_path, index=False)
        print(f"📊 {self.name} → tokenizer_benchmark.csv")
        
        # Store for downstream agents
        shared_context["tokenization_results"] = tokenization_results
        shared_context["tokenization_df"] = df
        
        return aggregate_stats
    
    def _tokenize(self, text: str, tokenizer: Any, tokenizer_name: str) -> int:
        """Tokenize text and return token count."""
        
        if tokenizer == "fallback":
            # Whitespace tokenizer fallback
            return len(text.split())
        
        if hasattr(tokenizer, "encode"):
            # tiktoken or transformers
            try:
                return len(tokenizer.encode(text))
            except Exception:
                return len(text.split())  # Fallback
        
        return len(text.split())
    
    def _compute_tokenizer_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics per tokenizer."""
        
        stats = {}
        
        token_cols = [col for col in df.columns if col.startswith("tokens_")]
        
        for col in token_cols:
            tokenizer_name = col.replace("tokens_", "")
            
            stats[tokenizer_name] = {
                "total_tokens": int(df[col].sum()),
                "mean_tokens": float(df[col].mean()),
                "median_tokens": float(df[col].median()),
                "p90_tokens": float(df[col].quantile(0.90)),
                "p95_tokens": float(df[col].quantile(0.95)),
                "p99_tokens": float(df[col].quantile(0.99)),
                "max_tokens": int(df[col].max()),
                "max_token_sample_id": df.loc[df[col].idxmax(), "sample_id"]
            }
        
        return stats
    
    def _compute_context_fit_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute context window fit statistics."""
        
        total = len(df)
        
        return {
            "fit_4k_rate": float((df["context_4k_fit"].sum() / total)),
            "fit_8k_rate": float((df["context_8k_fit"].sum() / total)),
            "fit_32k_rate": float((df["context_32k_fit"].sum() / total)),
            "fit_128k_rate": float((df["context_128k_fit"].sum() / total)),
            "truncation_4k_mean": float(df["truncation_at_4k"].mean()),
            "truncation_8k_mean": float(df["truncation_at_8k"].mean()),
            "truncation_32k_mean": float(df["truncation_at_32k"].mean()),
        }
    
    def _analyze_tokenizer_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drift between tokenizers."""
        
        token_cols = [col for col in df.columns if col.startswith("tokens_")]
        
        if len(token_cols) < 2:
            return {"note": "Need 2+ tokenizers for drift analysis"}
        
        # Pairwise drift
        drift = {}
        
        for i, col1 in enumerate(token_cols):
            for col2 in token_cols[i+1:]:
                pair_name = f"{col1.replace('tokens_', '')} vs {col2.replace('tokens_', '')}"
                
                diff = (df[col1] - df[col2]).abs()
                
                drift[pair_name] = {
                    "mean_abs_diff": float(diff.mean()),
                    "max_abs_diff": int(diff.max()),
                    "diff_rate": float((diff > 0).sum() / len(df))
                }
        
        return drift
    
    def _create_tokenizer_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tokenizer comparison benchmark table."""
        
        token_cols = [col for col in df.columns if col.startswith("tokens_")]
        
        benchmark_data = []
        
        for col in token_cols:
            tokenizer_name = col.replace("tokens_", "")
            
            benchmark_data.append({
                "tokenizer": tokenizer_name,
                "total_tokens": int(df[col].sum()),
                "mean_tokens_per_row": float(df[col].mean()),
                "median_tokens_per_row": float(df[col].median()),
                "p95_tokens_per_row": float(df[col].quantile(0.95)),
                "max_tokens_per_row": int(df[col].max()),
            })
        
        return pd.DataFrame(benchmark_data)


# ============================================================================
# AGENT 3: QUALITY SCORING
# ============================================================================

class QualityScoringAgent(BaseAgent):
    """
    Compute quality signals: readability, repetition, entropy, 
    boilerplate score, contamination heuristics.
    """
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Score quality for each conversation."""
        
        conversations = shared_context.get("conversations", [])
        
        if not conversations:
            raise ValueError("No conversations in shared context")
        
        quality_scores = []
        
        for conv in conversations:
            text = conv["text"]
            
            score = {
                "sample_id": conv["sample_id"],
                "readability_score": self._compute_readability(text),
                "repetition_score": self._compute_repetition(text),
                "entropy_score": self._compute_entropy(text),
                "boilerplate_score": self._compute_boilerplate(text),
                "quality_score": 0.0  # Composite score
            }
            
            # Composite quality score (weighted average)
            score["quality_score"] = (
                0.3 * score["readability_score"] +
                0.3 * (1 - score["repetition_score"]) +  # Lower repetition is better
                0.2 * score["entropy_score"] +
                0.2 * (1 - score["boilerplate_score"])  # Lower boilerplate is better
            )
            
            quality_scores.append(score)
        
        # Aggregate statistics
        df = pd.DataFrame(quality_scores)
        
        aggregate = {
            "mean_quality_score": float(df["quality_score"].mean()),
            "median_quality_score": float(df["quality_score"].median()),
            "quality_distribution": {
                "high_quality_rate": float((df["quality_score"] >= 0.7).sum() / len(df)),
                "medium_quality_rate": float(((df["quality_score"] >= 0.4) & (df["quality_score"] < 0.7)).sum() / len(df)),
                "low_quality_rate": float((df["quality_score"] < 0.4).sum() / len(df)),
            }
        }
        
        # Emit artifacts
        self.emit_artifact("quality_scores.json", {
            "per_row": quality_scores,
            "aggregate": aggregate
        })
        
        # Quality risk report
        risk_report = self._generate_quality_risk_report(df)
        self.emit_artifact("quality_risk_report.json", risk_report)
        
        # Store for downstream
        shared_context["quality_scores"] = quality_scores
        shared_context["quality_df"] = df
        
        return aggregate
    
    def _compute_readability(self, text: str) -> float:
        """Flesch Reading Ease approximation."""
        
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?') + 1
        
        if not words or not sentences:
            return 0.0
        
        # Approximation: avg word length as syllable proxy
        avg_word_len = sum(len(w) for w in words) / len(words)
        avg_sentence_len = len(words) / sentences
        
        # Normalized score (0-1)
        score = max(0, min(1, 1 - (avg_word_len - 4) / 10 - (avg_sentence_len - 15) / 50))
        
        return score
    
    def _compute_repetition(self, text: str) -> float:
        """Measure repetitiveness (trigram repetition rate)."""
        
        words = text.lower().split()
        
        if len(words) < 3:
            return 0.0
        
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        
        if not trigrams:
            return 0.0
        
        unique_trigrams = len(set(trigrams))
        repetition_rate = 1 - (unique_trigrams / len(trigrams))
        
        return repetition_rate
    
    def _compute_entropy(self, text: str) -> float:
        """Shannon entropy of character distribution (normalized)."""
        
        if not text:
            return 0.0
        
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        entropy = -sum(
            (count / total_chars) * np.log2(count / total_chars)
            for count in char_counts.values()
        )
        
        # Normalize by max possible entropy (log2 of alphabet size)
        max_entropy = np.log2(len(char_counts)) if char_counts else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_boilerplate(self, text: str) -> float:
        """Detect boilerplate patterns."""
        
        boilerplate_phrases = [
            "click here",
            "subscribe",
            "cookie policy",
            "terms of service",
            "privacy policy",
            "all rights reserved",
            "copyright",
            "disclaimer"
        ]
        
        text_lower = text.lower()
        
        matches = sum(1 for phrase in boilerplate_phrases if phrase in text_lower)
        
        # Score: 0 = no boilerplate, 1 = high boilerplate
        return min(1.0, matches / 3)
    
    def _generate_quality_risk_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate quality risk assessment."""
        
        # Identify problematic rows
        low_quality = df[df["quality_score"] < 0.4]
        high_repetition = df[df["repetition_score"] > 0.6]
        low_entropy = df[df["entropy_score"] < 0.3]
        
        return {
            "total_rows": len(df),
            "low_quality_count": len(low_quality),
            "high_repetition_count": len(high_repetition),
            "low_entropy_count": len(low_entropy),
            "risk_flags": {
                "low_quality_rate": len(low_quality) / len(df),
                "high_repetition_rate": len(high_repetition) / len(df),
                "low_entropy_rate": len(low_entropy) / len(df),
            },
            "recommendations": self._generate_quality_recommendations(df)
        }
    
    def _generate_quality_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable quality recommendations."""
        
        recommendations = []
        
        low_quality_rate = (df["quality_score"] < 0.4).sum() / len(df)
        
        if low_quality_rate > 0.2:
            recommendations.append(
                f"High low-quality rate ({low_quality_rate:.1%}). "
                "Consider filtering rows with quality_score < 0.4."
            )
        
        high_rep_rate = (df["repetition_score"] > 0.6).sum() / len(df)
        
        if high_rep_rate > 0.15:
            recommendations.append(
                f"High repetition rate ({high_rep_rate:.1%}). "
                "Consider deduplication or repetition filtering."
            )
        
        if not recommendations:
            recommendations.append("Quality metrics within acceptable ranges.")
        
        return recommendations


# ============================================================================
# AGENT 4: SAFETY & PII
# ============================================================================

class SafetyPIIAgent(BaseAgent):
    """
    Detect PII and sensitive classes with confidence and span-level evidence.
    """
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for PII and safety risks."""
        
        conversations = shared_context.get("conversations", [])
        
        if not conversations:
            raise ValueError("No conversations in shared context")
        
        pii_results = []
        
        for conv in conversations:
            text = conv["text"]
            
            result = {
                "sample_id": conv["sample_id"],
                **self._detect_pii(text),
                "safety_risk_score": self._compute_safety_risk(text)
            }
            
            pii_results.append(result)
        
        # Aggregate
        df = pd.DataFrame(pii_results)
        
        aggregate = {
            "total_rows": len(df),
            "pii_present_count": int(df["pii_present"].sum()),
            "pii_rate": float(df["pii_present"].mean()),
            "pii_type_distribution": self._compute_pii_type_distribution(df),
            "safety_risk_distribution": {
                "high_risk_count": int((df["safety_risk_score"] > 0.7).sum()),
                "medium_risk_count": int(((df["safety_risk_score"] > 0.3) & (df["safety_risk_score"] <= 0.7)).sum()),
                "low_risk_count": int((df["safety_risk_score"] <= 0.3).sum()),
            }
        }
        
        # Emit artifacts
        self.emit_artifact("pii_safety_results.json", {
            "per_row": pii_results,
            "aggregate": aggregate
        })
        
        self.emit_artifact("pii_safety_report.json", aggregate)
        
        # Store for downstream
        shared_context["pii_results"] = pii_results
        shared_context["pii_df"] = df
        
        return aggregate
    
    def _detect_pii(self, text: str) -> Dict[str, Any]:
        """Heuristic-based PII detection."""
        
        pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        
        detected_types = []
        
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, text):
                detected_types.append(pii_type)
        
        return {
            "pii_present": len(detected_types) > 0,
            "pii_types": detected_types,
            "pii_count": len(detected_types)
        }
    
    def _compute_safety_risk(self, text: str) -> float:
        """Compute safety risk score (0-1)."""
        
        # Heuristic: presence of sensitive keywords
        risk_keywords = [
            "password", "secret", "confidential", "private key",
            "api key", "token", "credentials"
        ]
        
        text_lower = text.lower()
        
        matches = sum(1 for keyword in risk_keywords if keyword in text_lower)
        
        # Normalize
        risk_score = min(1.0, matches / 3)
        
        return risk_score
    
    def _compute_pii_type_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count occurrences of each PII type."""
        
        pii_type_counts = Counter()
        
        for types in df["pii_types"]:
            for pii_type in types:
                pii_type_counts[pii_type] += 1
        
        return dict(pii_type_counts)


# ============================================================================
# AGENT 5: DEDUPLICATION
# ============================================================================

class DedupAgent(BaseAgent):
    """
    Exact hash, near-dup (MinHash), semantic dup clusters.
    Emit cluster IDs and retained-canonical suggestions.
    """
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-level deduplication."""
        
        conversations = shared_context.get("conversations", [])
        
        if not conversations:
            raise ValueError("No conversations in shared context")
        
        dedup_results = []
        
        # Exact deduplication (SHA256)
        exact_dup_groups = self._exact_dedup(conversations)
        
        # Near-dup (MinHash simulation)
        near_dup_clusters = self._near_dedup(conversations)
        
        # Semantic dup (embedding-based, placeholder)
        semantic_clusters = self._semantic_dedup(conversations)
        
        # Combine results
        for conv in conversations:
            sample_id = conv["sample_id"]
            text = conv["text"]
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            dedup_results.append({
                "sample_id": sample_id,
                "text_sha256": text_hash,
                "exact_dup_group": exact_dup_groups.get(text_hash, text_hash),
                "near_dup_cluster_id": near_dup_clusters.get(sample_id, sample_id),
                "semantic_cluster_id": semantic_clusters.get(sample_id, sample_id),
                "is_exact_dup": exact_dup_groups.get(text_hash, text_hash) != text_hash,
                "is_canonical": exact_dup_groups.get(text_hash, text_hash) == text_hash
            })
        
        # Aggregate statistics
        df = pd.DataFrame(dedup_results)
        
        aggregate = {
            "total_rows": len(df),
            "exact_dup_count": int(df["is_exact_dup"].sum()),
            "exact_dup_rate": float(df["is_exact_dup"].mean()),
            "unique_exact_groups": len(df["exact_dup_group"].unique()),
            "unique_near_dup_clusters": len(df["near_dup_cluster_id"].unique()),
            "unique_semantic_clusters": len(df["semantic_cluster_id"].unique()),
            "near_dup_rate": float(1 - len(df["near_dup_cluster_id"].unique()) / len(df)),
            "semantic_dup_rate": float(1 - len(df["semantic_cluster_id"].unique()) / len(df)),
        }
        
        # Emit artifacts
        self.emit_artifact("dedup_results.json", {
            "per_row": dedup_results,
            "aggregate": aggregate
        })
        
        # Dedup clusters parquet
        output_path = self.config.output_dir / "dedup_clusters.parquet"
        df.to_parquet(output_path, index=False)
        print(f"📊 {self.name} → dedup_clusters.parquet")
        
        # Store for downstream
        shared_context["dedup_results"] = dedup_results
        shared_context["dedup_df"] = df
        
        return aggregate
    
    def _exact_dedup(self, conversations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Exact deduplication via SHA256."""
        
        hash_to_first_id = {}
        
        for conv in conversations:
            text = conv["text"]
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            if text_hash not in hash_to_first_id:
                hash_to_first_id[text_hash] = text_hash
        
        return hash_to_first_id
    
    def _near_dedup(self, conversations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Near-duplicate detection (MinHash simulation)."""
        
        # Simplified: group by first 100 chars (simulates MinHash bucketing)
        prefix_to_cluster = {}
        
        for conv in conversations:
            text = conv["text"]
            prefix = text[:100]
            
            if prefix not in prefix_to_cluster:
                prefix_to_cluster[prefix] = conv["sample_id"]
        
        # Map each sample to its cluster
        sample_to_cluster = {}
        
        for conv in conversations:
            text = conv["text"]
            prefix = text[:100]
            sample_to_cluster[conv["sample_id"]] = prefix_to_cluster[prefix]
        
        return sample_to_cluster
    
    def _semantic_dedup(self, conversations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Semantic deduplication (placeholder - requires embeddings)."""
        
        # Placeholder: each sample is its own cluster
        return {conv["sample_id"]: conv["sample_id"] for conv in conversations}


# ============================================================================
# AGENT 6: COST MODEL
# ============================================================================

class CostModelAgent(BaseAgent):
    """
    Convert token distributions into training and inference cost projections
    by model family and context size.
    """
    
    COST_TABLE = {
        # Training costs ($ per 1M tokens)
        "training": {
            "gpt-3.5-turbo": 8.0,
            "gpt-4": 30.0,
            "llama-2-7b": 2.0,
            "llama-2-13b": 4.0,
            "llama-2-70b": 15.0,
        },
        # Inference costs ($ per 1M tokens)
        "inference": {
            "gpt-3.5-turbo": 0.5,
            "gpt-4": 30.0,
            "llama-2-7b": 0.2,
            "llama-2-13b": 0.4,
            "llama-2-70b": 2.0,
        }
    }
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Project training and inference costs."""
        
        tokenization_df = shared_context.get("tokenization_df")
        
        if tokenization_df is None:
            raise ValueError("No tokenization results in shared context")
        
        # Extract token columns
        token_cols = [col for col in tokenization_df.columns if col.startswith("tokens_")]
        
        if not token_cols:
            raise ValueError("No token counts found in tokenization results")
        
        # Compute costs per tokenizer
        cost_projections = {}
        
        for col in token_cols:
            tokenizer_name = col.replace("tokens_", "")
            total_tokens = tokenization_df[col].sum()
            
            # Map tokenizer to model family
            model_family = self._map_tokenizer_to_model(tokenizer_name)
            
            train_cost_per_m = self.COST_TABLE["training"].get(model_family, 5.0)
            infer_cost_per_m = self.COST_TABLE["inference"].get(model_family, 1.0)
            
            train_cost = (total_tokens / 1_000_000) * train_cost_per_m
            infer_cost = (total_tokens / 1_000_000) * infer_cost_per_m
            
            cost_projections[tokenizer_name] = {
                "total_tokens": int(total_tokens),
                "model_family": model_family,
                "estimated_train_cost_usd": round(train_cost, 2),
                "estimated_infer_cost_usd": round(infer_cost, 2),
                "cost_per_1m_tokens_train": train_cost_per_m,
                "cost_per_1m_tokens_infer": infer_cost_per_m,
            }
        
        # Aggregate
        aggregate = {
            "projections_by_tokenizer": cost_projections,
            "total_projected_train_cost_range": {
                "min_usd": min(p["estimated_train_cost_usd"] for p in cost_projections.values()),
                "max_usd": max(p["estimated_train_cost_usd"] for p in cost_projections.values()),
            },
            "total_projected_infer_cost_range": {
                "min_usd": min(p["estimated_infer_cost_usd"] for p in cost_projections.values()),
                "max_usd": max(p["estimated_infer_cost_usd"] for p in cost_projections.values()),
            }
        }
        
        # Emit artifact
        self.emit_artifact("cost_projection.json", aggregate)
        
        # Store for downstream
        shared_context["cost_projections"] = cost_projections
        
        return aggregate
    
    def _map_tokenizer_to_model(self, tokenizer_name: str) -> str:
        """Map tokenizer name to model family for cost lookup."""
        
        mapping = {
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "llama-2-7b": "llama-2-7b",
            "llama-2-13b": "llama-2-13b",
            "llama-2-70b": "llama-2-70b",
            "whitespace": "llama-2-7b",  # Default fallback
        }
        
        return mapping.get(tokenizer_name, "llama-2-7b")


# ============================================================================
# AGENT 7: VERIFIER
# ============================================================================

class VerifierAgent(BaseAgent):
    """
    Cross-check counts, schema integrity, null rates, checksum reproducibility,
    and report consistency. Block completion if quality gates fail.
    """
    
    REQUIRED_ARTIFACTS = [
        "data_profile.json",
        "tokenization_results.json",
        "quality_scores.json",
        "pii_safety_results.json",
        "dedup_results.json",
        "cost_projection.json",
        "token_row_metrics.parquet",
        "tokenizer_benchmark.csv",
        "dedup_clusters.parquet"
    ]
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive verification checks."""
        
        verification_results = {
            "artifacts_check": self._verify_artifacts(),
            "schema_integrity_check": self._verify_schema_integrity(shared_context),
            "count_consistency_check": self._verify_count_consistency(shared_context),
            "reproducibility_check": self._verify_reproducibility(shared_context),
            "quality_gates": self._verify_quality_gates(shared_context)
        }
        
        # Overall pass/fail
        all_checks_passed = all(
            check.get("passed", False) 
            for check in verification_results.values()
        )
        
        verification_results["overall_status"] = "PASS" if all_checks_passed else "FAIL"
        
        # Emit verification report
        self.emit_artifact("verification_report.json", verification_results)
        
        # If failed, raise error to halt pipeline
        if not all_checks_passed:
            failed_checks = [
                name for name, check in verification_results.items()
                if not check.get("passed", False)
            ]
            raise ValueError(f"Verification failed on: {', '.join(failed_checks)}")
        
        return verification_results
    
    def _verify_artifacts(self) -> Dict[str, Any]:
        """Check that all required artifacts were generated."""
        
        missing_artifacts = []
        
        for artifact_name in self.REQUIRED_ARTIFACTS:
            artifact_path = self.config.output_dir / artifact_name
            
            if not artifact_path.exists():
                missing_artifacts.append(artifact_name)
        
        return {
            "passed": len(missing_artifacts) == 0,
            "required_count": len(self.REQUIRED_ARTIFACTS),
            "found_count": len(self.REQUIRED_ARTIFACTS) - len(missing_artifacts),
            "missing_artifacts": missing_artifacts
        }
    
    def _verify_schema_integrity(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify schema consistency across agents."""
        
        # All agents should have processed same number of rows
        conversations = shared_context.get("conversations", [])
        expected_count = len(conversations)
        
        quality_df = shared_context.get("quality_df")
        pii_df = shared_context.get("pii_df")
        dedup_df = shared_context.get("dedup_df")
        
        counts = {
            "conversations": expected_count,
            "quality_scores": len(quality_df) if quality_df is not None else 0,
            "pii_results": len(pii_df) if pii_df is not None else 0,
            "dedup_results": len(dedup_df) if dedup_df is not None else 0,
        }
        
        # Check consistency
        counts_match = len(set(counts.values())) == 1
        
        return {
            "passed": counts_match,
            "row_counts": counts,
            "expected_count": expected_count
        }
    
    def _verify_count_consistency(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify count consistency across different analysis outputs."""
        
        # Extract counts from different sources
        data_profile_path = self.config.output_dir / "data_profile.json"
        
        with open(data_profile_path, 'r') as f:
            data_profile = json.load(f)
        
        profile_count = data_profile.get("row_count", 0)
        conversations_count = len(shared_context.get("conversations", []))
        
        counts_match = profile_count == conversations_count
        
        return {
            "passed": counts_match,
            "data_profile_count": profile_count,
            "conversations_count": conversations_count
        }
    
    def _verify_reproducibility(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify reproducibility hash can be generated."""
        
        # Check that all JSON artifacts have stable structure
        try:
            hasher = hashlib.sha256()
            
            for artifact_path in sorted(self.config.output_dir.glob("*.json")):
                if artifact_path.name not in ["repro_manifest.json", "verification_report.json"]:
                    with open(artifact_path, 'rb') as f:
                        hasher.update(f.read())
            
            repro_hash = hasher.hexdigest()
            
            return {
                "passed": True,
                "reproducibility_hash": repro_hash
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _verify_quality_gates(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify quality metrics meet minimum thresholds."""
        
        quality_df = shared_context.get("quality_df")
        
        if quality_df is None:
            return {"passed": False, "error": "No quality scores available"}
        
        # Quality gates
        mean_quality = quality_df["quality_score"].mean()
        low_quality_rate = (quality_df["quality_score"] < 0.3).sum() / len(quality_df)
        
        gates = {
            "mean_quality_above_0.5": bool(mean_quality >= 0.5),
            "low_quality_rate_below_0.3": bool(low_quality_rate < 0.3),
        }
        
        all_gates_passed = bool(all(gates.values()))
        
        return {
            "passed": all_gates_passed,
            "gates": gates,
            "mean_quality_score": float(mean_quality),
            "low_quality_rate": float(low_quality_rate)
        }


# Export all agents
__all__ = [
    "DataProfilerAgent",
    "MultiTokenizerAgent",
    "QualityScoringAgent",
    "SafetyPIIAgent",
    "DedupAgent",
    "CostModelAgent",
    "VerifierAgent"
]
