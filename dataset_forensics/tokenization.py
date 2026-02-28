"""
Multi-tokenizer support for dataset forensics.

Supports:
- tiktoken (cl100k_base, p50k_base, etc.)
- HuggingFace tokenizers
- Character-level fallback tokenizer
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    text: str
    tokens: List[int]
    token_count: int
    tokenizer_name: str
    tokenizer_version: str
    char_count: int
    special_token_count: int = 0
    unk_token_count: int = 0
    compression_ratio: float = 0.0
    processing_time_ms: float = 0.0
    
    @property
    def tokens_per_char(self) -> float:
        """Average tokens per character."""
        if self.char_count > 0:
            return self.token_count / self.char_count
        return 0.0
    
    @property
    def chars_per_token(self) -> float:
        """Average characters per token."""
        if self.token_count > 0:
            return self.char_count / self.token_count
        return 0.0


@dataclass
class TokenizerBenchmark:
    """Benchmark results for a tokenizer."""
    tokenizer_name: str
    total_chars: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    samples_processed: int = 0
    errors: int = 0
    
    @property
    def avg_tokens_per_sample(self) -> float:
        if self.samples_processed > 0:
            return self.total_tokens / self.samples_processed
        return 0.0
    
    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms > 0:
            return self.total_tokens / (self.total_time_ms / 1000)
        return 0.0
    
    @property
    def chars_per_second(self) -> float:
        if self.total_time_ms > 0:
            return self.total_chars / (self.total_time_ms / 1000)
        return 0.0
    
    @property
    def avg_compression_ratio(self) -> float:
        if self.total_chars > 0:
            return self.total_tokens / self.total_chars
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tokenizer_name': self.tokenizer_name,
            'total_chars': self.total_chars,
            'total_tokens': self.total_tokens,
            'samples_processed': self.samples_processed,
            'errors': self.errors,
            'avg_tokens_per_sample': self.avg_tokens_per_sample,
            'tokens_per_second': self.tokens_per_second,
            'chars_per_second': self.chars_per_second,
            'avg_compression_ratio': self.avg_compression_ratio,
            'total_time_sec': self.total_time_ms / 1000,
        }


class TokenizerBackend(ABC):
    """Abstract base class for tokenizer backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tokenizer name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Tokenizer version/encoding."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens without full encoding."""
        pass
    
    def get_vocab_size(self) -> Optional[int]:
        """Get vocabulary size if available."""
        return None
    
    def get_special_tokens(self) -> Set[str]:
        """Get special token strings."""
        return set()
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return True


class TiktokenBackend(TokenizerBackend):
    """tiktoken backend for OpenAI tokenizers."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoder = None
        self._special_tokens: Set[str] = set()
        
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(encoding_name)
            self._special_tokens = set(self._encoder.special_tokens_set)
        except ImportError:
            logger.warning("tiktoken not installed, TiktokenBackend unavailable")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
    
    @property
    def name(self) -> str:
        return f"tiktoken_{self.encoding_name}"
    
    @property
    def version(self) -> str:
        return self.encoding_name
    
    def is_available(self) -> bool:
        return self._encoder is not None
    
    def encode(self, text: str) -> List[int]:
        if not self._encoder:
            raise RuntimeError("tiktoken encoder not available")
        return self._encoder.encode(text, disallowed_special=())
    
    def decode(self, tokens: List[int]) -> str:
        if not self._encoder:
            raise RuntimeError("tiktoken encoder not available")
        return self._encoder.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        if not self._encoder:
            raise RuntimeError("tiktoken encoder not available")
        return len(self._encoder.encode(text, disallowed_special=()))
    
    def get_vocab_size(self) -> Optional[int]:
        if self._encoder:
            return self._encoder.n_vocab
        return None
    
    def get_special_tokens(self) -> Set[str]:
        return self._special_tokens


class HuggingFaceBackend(TokenizerBackend):
    """HuggingFace transformers tokenizer backend."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._tokenizer = None
        self._unk_token_id: Optional[int] = None
        
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self._tokenizer.unk_token_id is not None:
                self._unk_token_id = self._tokenizer.unk_token_id
        except ImportError:
            logger.warning("transformers not installed, HuggingFaceBackend unavailable")
        except Exception as e:
            logger.warning(f"Failed to load HF tokenizer {model_name}: {e}")
    
    @property
    def name(self) -> str:
        return f"hf_{self.model_name}"
    
    @property
    def version(self) -> str:
        return self.model_name
    
    def is_available(self) -> bool:
        return self._tokenizer is not None
    
    def encode(self, text: str) -> List[int]:
        if not self._tokenizer:
            raise RuntimeError("HF tokenizer not available")
        return self._tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, tokens: List[int]) -> str:
        if not self._tokenizer:
            raise RuntimeError("HF tokenizer not available")
        return self._tokenizer.decode(tokens, skip_special_tokens=True)
    
    def count_tokens(self, text: str) -> int:
        if not self._tokenizer:
            raise RuntimeError("HF tokenizer not available")
        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def get_vocab_size(self) -> Optional[int]:
        if self._tokenizer:
            return len(self._tokenizer)
        return None
    
    def get_special_tokens(self) -> Set[str]:
        if not self._tokenizer:
            return set()
        special = set()
        for attr in ['pad_token', 'eos_token', 'bos_token', 'unk_token', 'cls_token', 'sep_token', 'mask_token']:
            token = getattr(self._tokenizer, attr, None)
            if token:
                special.add(token)
        return special


class CharacterBackend(TokenizerBackend):
    """Character-level tokenizer fallback."""
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
        
    @property
    def name(self) -> str:
        return "character"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    def encode(self, text: str) -> List[int]:
        """Encode as Unicode code points."""
        return [ord(c) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode from Unicode code points."""
        return ''.join(chr(t) for t in tokens if 0 <= t <= 0x10FFFF)
    
    def count_tokens(self, text: str) -> int:
        """Estimate based on character count."""
        return max(1, int(len(text) / self.chars_per_token))
    
    def get_vocab_size(self) -> Optional[int]:
        return 0x10FFFF + 1  # Full Unicode range


class TokenizationEngine:
    """
    Main tokenization engine supporting multiple backends.
    
    Provides:
    - Multi-tokenizer support
    - Context window fitting
    - Benchmarking
    - Statistics collection
    """
    
    CONTEXT_WINDOWS = {
        '4k': 4096,
        '8k': 8192,
        '32k': 32768,
        '128k': 131072,
    }
    
    def __init__(
        self,
        backend: Union[str, TokenizerBackend] = "cl100k_base",
        context_windows: Optional[List[int]] = None,
    ):
        """
        Initialize tokenization engine.
        
        Args:
            backend: Tokenizer backend name or instance
            context_windows: List of context window sizes to track
        """
        if isinstance(backend, str):
            self.backend = self._create_backend(backend)
        else:
            self.backend = backend
        
        self.context_windows = context_windows or [4096, 8192, 32768]
        self.benchmark = TokenizerBenchmark(self.backend.name)
        self._special_tokens = self.backend.get_special_tokens()
        
        if not self.backend.is_available():
            logger.warning(f"Backend {self.backend.name} not available, falling back to character")
            self.backend = CharacterBackend()
    
    def _create_backend(self, name: str) -> TokenizerBackend:
        """Create tokenizer backend by name."""
        if name.startswith("tiktoken_") or name in ["cl100k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2"]:
            encoding = name.replace("tiktoken_", "")
            return TiktokenBackend(encoding)
        elif name.startswith("hf_"):
            model = name[3:]
            return HuggingFaceBackend(model)
        elif name == "character":
            return CharacterBackend()
        else:
            # Try as tiktoken encoding
            return TiktokenBackend(name)
    
    def tokenize(self, text: str) -> TokenizationResult:
        """
        Tokenize text and return full result.
        
        Args:
            text: Text to tokenize
            
        Returns:
            TokenizationResult with tokens and metadata
        """
        start_time = time.perf_counter()
        
        try:
            tokens = self.backend.encode(text)
            token_count = len(tokens)
            
            # Count special tokens
            special_count = sum(1 for t in tokens if self._is_special_token(t))
            
            # Estimate unknown tokens (implementation depends on backend)
            unk_count = self._count_unknown_tokens(tokens)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Update benchmark
            self.benchmark.total_chars += len(text)
            self.benchmark.total_tokens += token_count
            self.benchmark.total_time_ms += processing_time
            self.benchmark.samples_processed += 1
            
            # Calculate compression ratio
            compression = token_count / max(len(text), 1)
            
            return TokenizationResult(
                text=text,
                tokens=tokens,
                token_count=token_count,
                tokenizer_name=self.backend.name,
                tokenizer_version=self.backend.version,
                char_count=len(text),
                special_token_count=special_count,
                unk_token_count=unk_count,
                compression_ratio=compression,
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            self.benchmark.errors += 1
            logger.error(f"Tokenization error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Quick token count without full tokenization."""
        return self.backend.count_tokens(text)
    
    def _is_special_token(self, token_id: int) -> bool:
        """Check if token ID is a special token."""
        # This is backend-specific; for now use a heuristic
        return token_id < 100  # Most special tokens have low IDs
    
    def _count_unknown_tokens(self, tokens: List[int]) -> int:
        """Count unknown/UNK tokens."""
        vocab_size = self.backend.get_vocab_size()
        if vocab_size:
            return sum(1 for t in tokens if t >= vocab_size)
        return 0
    
    def fits_context_window(self, text: str, window_size: int) -> Tuple[bool, int]:
        """
        Check if text fits in context window.
        
        Args:
            text: Text to check
            window_size: Context window size
            
        Returns:
            (fits: bool, token_count: int)
        """
        token_count = self.count_tokens(text)
        return token_count <= window_size, token_count
    
    def check_all_context_windows(self, text: str) -> Dict[str, Any]:
        """
        Check text against all configured context windows.
        
        Returns:
            Dict with fit status for each window size
        """
        token_count = self.count_tokens(text)
        results = {}
        
        for size_name, size_value in self.CONTEXT_WINDOWS.items():
            if size_value in self.context_windows:
                fits = token_count <= size_value
                results[f"context_{size_name}_fit"] = fits
                results[f"truncation_at_{size_name}"] = not fits
        
        results['token_count'] = token_count
        return results
    
    def truncate_to_context_window(
        self,
        text: str,
        window_size: int,
        truncation_side: str = "right",
    ) -> str:
        """
        Truncate text to fit in context window.
        
        Args:
            text: Text to truncate
            window_size: Maximum tokens
            truncation_side: "left" or "right"
            
        Returns:
            Truncated text
        """
        tokens = self.backend.encode(text)
        
        if len(tokens) <= window_size:
            return text
        
        if truncation_side == "left":
            truncated = tokens[-window_size:]
        else:
            truncated = tokens[:window_size]
        
        return self.backend.decode(truncated)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        return self.benchmark.to_dict()
    
    def reset_benchmark(self):
        """Reset benchmark statistics."""
        self.benchmark = TokenizerBenchmark(self.backend.name)


def estimate_tokens(text: str, method: str = "char_ratio") -> int:
    """
    Estimate token count without full tokenization.
    
    Args:
        text: Input text
        method: Estimation method ("char_ratio", "words", "hybrid")
        
    Returns:
        Estimated token count
    """
    if method == "char_ratio":
        # GPT models average ~4 chars per token
        return max(1, len(text) // 4)
    elif method == "words":
        # ~1.3 tokens per word on average
        words = len(text.split())
        return max(1, int(words * 1.3))
    elif method == "hybrid":
        # Combine character and word estimates
        char_est = len(text) // 4
        word_est = int(len(text.split()) * 1.3)
        return max(1, (char_est + word_est) // 2)
    else:
        raise ValueError(f"Unknown estimation method: {method}")


def calculate_entropy(tokens: List[int]) -> float:
    """
    Calculate Shannon entropy of token distribution.
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Shannon entropy in bits
    """
    if not tokens:
        return 0.0
    
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    
    for count in counts.values():
        prob = count / total
        entropy -= prob * (prob.bit_length() - 1)  # log2 approximation
    
    return entropy
