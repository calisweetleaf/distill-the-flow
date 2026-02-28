"""
dataset_forensics: Production-grade SOTA++ token forensics system.

A comprehensive toolkit for analyzing, tokenizing, and quality-assessing
datasets for large language model training and evaluation.

Modules:
    ingestion: Streaming JSON/JSONL parsing for large datasets
    normalization: Text cleaning and unicode normalization
    tokenization: Multi-tokenizer support (tiktoken, HuggingFace)
    metadata: Row-level metadata extraction
    dedup: Exact and near-duplicate detection
    quality: Quality scoring and anomaly detection
    cli: Command-line interface with structured logging

Example:
    >>> from dataset_forensics import TokenizationEngine, MetadataExtractor
    >>> tokenizer = TokenizationEngine("cl100k_base")
    >>> metadata = MetadataExtractor(tokenizer)
"""

__version__ = "1.0.0"
__author__ = "Dataset Forensics Team"

from dataset_forensics.ingestion import (
    StreamingJSONParser,
    StreamingJSONLParser,
    ChatGPTExportParser,
    FileShardIterator,
)

from dataset_forensics.normalization import (
    TextNormalizer,
    UnicodeNormalizer,
    WhitespaceNormalizer,
)

from dataset_forensics.tokenization import (
    TokenizationEngine,
    TokenizerBackend,
    TiktokenBackend,
    HuggingFaceBackend,
    CharacterBackend,
    TokenizationResult,
)

from dataset_forensics.metadata import (
    MetadataExtractor,
    RowMetadata,
    LanguageDetector,
    compute_sha256,
)

from dataset_forensics.dedup import (
    DedupEngine,
    ExactDedup,
    MinHashDedup,
    SimHashDedup,
)

from dataset_forensics.quality import (
    QualityAnalyzer,
    EntropyAnalyzer,
    RepetitionDetector,
    AnomalyDetector,
    QualityScore,
)

__all__ = [
    # Ingestion
    "StreamingJSONParser",
    "StreamingJSONLParser",
    "ChatGPTExportParser",
    "FileShardIterator",
    # Normalization
    "TextNormalizer",
    "UnicodeNormalizer",
    "WhitespaceNormalizer",
    # Tokenization
    "TokenizationEngine",
    "TokenizerBackend",
    "TiktokenBackend",
    "HuggingFaceBackend",
    "CharacterBackend",
    "TokenizationResult",
    # Metadata
    "MetadataExtractor",
    "RowMetadata",
    "LanguageDetector",
    "compute_sha256",
    # Deduplication
    "DedupEngine",
    "ExactDedup",
    "MinHashDedup",
    "SimHashDedup",
    # Quality
    "QualityAnalyzer",
    "EntropyAnalyzer",
    "RepetitionDetector",
    "AnomalyDetector",
    "QualityScore",
]
