"""
Row-level metadata extraction for dataset forensics.

Extracts:
- SHA256 checksums
- Character/word/line counts
- Language detection
- Token counts and context window fits
- Duplication info
- Quality scores
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import Counter
import json

from dataset_forensics.tokenization import TokenizationEngine, TokenizationResult

logger = logging.getLogger(__name__)


@dataclass
class RowMetadata:
    """
    Complete row metadata for a dataset sample.
    
    All fields defined in the specification.
    """
    # Identification
    sample_id: str
    split: str = "train"
    source: str = "unknown"
    source_license: str = "unknown"
    
    # Content hashes
    text_sha256: str = ""
    raw_bytes: int = 0
    
    # Text statistics
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    
    # Language detection
    language: str = "unknown"
    language_confidence: float = 0.0
    
    # Tokenization (primary tokenizer)
    tokenizer_name: str = ""
    tokenizer_version: str = ""
    token_count: int = 0
    special_token_count: int = 0
    unk_token_count: int = 0
    
    # Context window fits
    context_4k_fit: bool = False
    context_8k_fit: bool = False
    context_32k_fit: bool = False
    truncation_at_4k: bool = False
    truncation_at_8k: bool = False
    truncation_at_32k: bool = False
    
    # Deduplication
    exact_dup_group: Optional[str] = None
    near_dup_cluster_id: Optional[str] = None
    
    # Quality scores
    quality_score: float = 0.0
    entropy_score: float = 0.0
    repetition_score: float = 0.0
    safety_risk_score: float = 0.0
    anomaly_flags: List[str] = field(default_factory=list)
    
    # Additional metrics (for internal use)
    _additional: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (serializable)."""
        result = asdict(self)
        # Remove internal fields
        result.pop('_additional', None)
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RowMetadata':
        """Create from dictionary."""
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def compute_sha256(text: str) -> str:
    """
    Compute SHA256 hash of text.
    
    Args:
        text: Input text
        
    Returns:
        Hex-encoded SHA256 hash
    """
    if isinstance(text, str):
        data = text.encode('utf-8')
    else:
        data = text
    return hashlib.sha256(data).hexdigest()


def compute_xxhash(text: str) -> str:
    """
    Compute fast XXH3 hash (if available).
    
    Args:
        text: Input text
        
    Returns:
        Hex-encoded hash
    """
    try:
        import xxhash
        if isinstance(text, str):
            data = text.encode('utf-8')
        else:
            data = text
        return xxhash.xxh3_64(data).hexdigest()
    except ImportError:
        # Fall back to truncated SHA256
        return compute_sha256(text)[:16]


class LanguageDetector:
    """
    Language detection with multiple backend support.
    
    Supports:
    - fasttext (fast, accurate)
    - langdetect (pure Python)
    - cld3 (Google Compact Language Detector)
    """
    
    def __init__(self, backend: str = "fasttext", min_confidence: float = 0.5):
        self.backend = backend
        self.min_confidence = min_confidence
        self._detector = None
        self._available = False
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the detection backend."""
        if self.backend == "fasttext":
            try:
                import fasttext
                # Try to load a pre-trained model
                self._detector = fasttext.load_model('lid.176.ftz')
                self._available = True
            except Exception as e:
                logger.warning(f"fasttext not available: {e}")
                self._try_fallback()
                
        elif self.backend == "langdetect":
            try:
                from langdetect import detect, detect_langs
                self._detector = detect
                self._available = True
            except ImportError:
                logger.warning("langdetect not available")
                self._try_fallback()
                
        elif self.backend == "cld3":
            try:
                import cld3
                self._detector = cld3
                self._available = True
            except ImportError:
                logger.warning("cld3 not available")
                self._try_fallback()
        else:
            self._try_fallback()
    
    def _try_fallback(self):
        """Try to use a fallback detection method."""
        try:
            from langdetect import detect
            self.backend = "langdetect"
            self._detector = detect
            self._available = True
            logger.info("Using langdetect as fallback")
        except ImportError:
            self._available = False
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            (language_code, confidence)
        """
        if not text or len(text) < 50:
            return ("unknown", 0.0)
        
        if not self._available:
            return ("unknown", 0.0)
        
        try:
            if self.backend == "fasttext":
                predictions = self._detector.predict(text.replace('\n', ' '))
                lang = predictions[0][0].replace('__label__', '')
                confidence = predictions[1][0]
                return (lang, confidence)
                
            elif self.backend == "langdetect":
                from langdetect import detect_langs
                langs = detect_langs(text)
                if langs:
                    return (langs[0].lang, langs[0].prob)
                return ("unknown", 0.0)
                
            elif self.backend == "cld3":
                result = self._detector.get_language(text)
                if result:
                    return (result.language, result.probability)
                return ("unknown", 0.0)
                
        except Exception as e:
            logger.debug(f"Language detection error: {e}")
        
        return ("unknown", 0.0)
    
    def is_available(self) -> bool:
        """Check if detector is available."""
        return self._available


class MetadataExtractor:
    """
    Extract comprehensive metadata from text samples.
    
    Integrates with tokenization engine and provides
    all required row-level metadata fields.
    """
    
    CONTEXT_SIZES = {
        '4k': 4096,
        '8k': 8192,
        '32k': 32768,
    }
    
    def __init__(
        self,
        tokenizer: Optional[TokenizationEngine] = None,
        language_detector: Optional[LanguageDetector] = None,
        enable_language_detection: bool = True,
    ):
        """
        Initialize metadata extractor.
        
        Args:
            tokenizer: TokenizationEngine instance
            language_detector: LanguageDetector instance
            enable_language_detection: Whether to detect language
        """
        self.tokenizer = tokenizer or TokenizationEngine("cl100k_base")
        self.language_detector = language_detector
        
        if enable_language_detection and self.language_detector is None:
            self.language_detector = LanguageDetector()
    
    def extract(
        self,
        text: str,
        sample_id: Optional[str] = None,
        split: str = "train",
        source: str = "unknown",
        source_license: str = "unknown",
    ) -> RowMetadata:
        """
        Extract complete metadata for a text sample.
        
        Args:
            text: Input text
            sample_id: Unique identifier (generated if None)
            split: Dataset split
            source: Source identifier
            source_license: License information
            
        Returns:
            RowMetadata with all fields populated
        """
        # Generate ID if not provided
        if sample_id is None:
            sample_id = str(uuid.uuid4())
        
        # Basic text statistics
        char_count = len(text)
        word_count = len(text.split())
        line_count = text.count('\n') + 1
        
        # Compute hash
        text_sha256 = compute_sha256(text)
        raw_bytes = len(text.encode('utf-8'))
        
        # Tokenization
        token_result = self.tokenizer.tokenize(text)
        context_fits = self.tokenizer.check_all_context_windows(text)
        
        # Language detection
        language = "unknown"
        language_confidence = 0.0
        if self.language_detector and len(text) >= 50:
            try:
                language, language_confidence = self.language_detector.detect(text)
            except Exception as e:
                logger.debug(f"Language detection failed: {e}")
        
        # Build metadata
        metadata = RowMetadata(
            sample_id=sample_id,
            split=split,
            source=source,
            source_license=source_license,
            text_sha256=text_sha256,
            raw_bytes=raw_bytes,
            char_count=char_count,
            word_count=word_count,
            line_count=line_count,
            language=language,
            language_confidence=language_confidence,
            tokenizer_name=token_result.tokenizer_name,
            tokenizer_version=token_result.tokenizer_version,
            token_count=token_result.token_count,
            special_token_count=token_result.special_token_count,
            unk_token_count=token_result.unk_token_count,
            context_4k_fit=context_fits.get('context_4k_fit', False),
            context_8k_fit=context_fits.get('context_8k_fit', False),
            context_32k_fit=context_fits.get('context_32k_fit', False),
            truncation_at_4k=context_fits.get('truncation_at_4k', False),
            truncation_at_8k=context_fits.get('truncation_at_8k', False),
            truncation_at_32k=context_fits.get('truncation_at_32k', False),
        )
        
        return metadata
    
    def extract_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
        split: str = "train",
        source: str = "unknown",
    ) -> List[RowMetadata]:
        """
        Extract metadata for a batch of texts.
        
        Args:
            texts: List of input texts
            sample_ids: List of IDs (generated if None)
            split: Dataset split
            source: Source identifier
            
        Returns:
            List of RowMetadata
        """
        if sample_ids is None:
            sample_ids = [str(uuid.uuid4()) for _ in texts]
        
        results = []
        for text, sid in zip(texts, sample_ids):
            metadata = self.extract(text, sid, split, source)
            results.append(metadata)
        
        return results
    
    def compute_text_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text.
        
        Args:
            text: Input text
            
        Returns:
            Entropy in bits per character
        """
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        total = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * (prob.bit_length() - 1)
        
        return entropy
    
    def compute_token_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of token distribution.
        
        Args:
            text: Input text
            
        Returns:
            Entropy in bits per token
        """
        try:
            result = self.tokenizer.tokenize(text)
            if not result.tokens:
                return 0.0
            
            token_counts = Counter(result.tokens)
            total = len(result.tokens)
            entropy = 0.0
            
            for count in token_counts.values():
                if count > 0:
                    prob = count / total
                    # log2 approximation
                    entropy -= prob * (prob.bit_length() - 1)
            
            return entropy
        except Exception as e:
            logger.debug(f"Token entropy calculation failed: {e}")
            return 0.0


def calculate_repetition_score(text: str, ngram_sizes: List[int] = None) -> float:
    """
    Calculate repetition score for text.
    
    Higher scores indicate more repetitive content.
    
    Args:
        text: Input text
        ngram_sizes: N-gram sizes to check (default: [1, 2, 3, 4])
        
    Returns:
        Repetition score (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    ngram_sizes = ngram_sizes or [1, 2, 3, 4]
    scores = []
    
    for n in ngram_sizes:
        if len(text) < n:
            continue
        
        # Generate n-grams
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams:
            continue
        
        # Calculate repetition ratio
        unique = len(set(ngrams))
        total = len(ngrams)
        repetition = 1.0 - (unique / total)
        scores.append(repetition)
    
    return sum(scores) / len(scores) if scores else 0.0


def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extract basic text features.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of text features
    """
    return {
        'char_count': len(text),
        'byte_count': len(text.encode('utf-8')),
        'word_count': len(text.split()),
        'line_count': text.count('\n') + 1,
        'sentence_count': text.count('.') + text.count('!') + text.count('?'),
        'avg_word_length': sum(len(w) for w in text.split()) / max(len(text.split()), 1),
        'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
        'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
        'whitespace_ratio': sum(c.isspace() for c in text) / max(len(text), 1),
        'punctuation_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1),
    }
