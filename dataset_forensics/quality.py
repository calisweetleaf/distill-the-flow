"""
Quality analysis for dataset forensics.

Analyzes:
- Entropy (Shannon entropy for randomness)
- Repetition detection
- Information gain
- Anomaly detection
- Safety risk scoring
"""

import re
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter
from statistics import mean, stdev

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality scoring result."""
    overall_score: float  # 0.0 to 1.0
    entropy_score: float
    repetition_score: float
    information_gain_score: float
    safety_risk_score: float
    anomaly_flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_high_quality(self, threshold: float = 0.6) -> bool:
        """Check if quality meets threshold."""
        return self.overall_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'entropy_score': self.entropy_score,
            'repetition_score': self.repetition_score,
            'information_gain_score': self.information_gain_score,
            'safety_risk_score': self.safety_risk_score,
            'anomaly_flags': self.anomaly_flags,
            'details': self.details,
        }


class BaseQualityAnalyzer(ABC):
    """Abstract base class for quality analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> float:
        """Analyze text and return quality score."""
        pass


class EntropyAnalyzer(BaseQualityAnalyzer):
    """
    Shannon entropy analyzer.
    
    Measures information density and randomness.
    Higher entropy generally indicates more information content.
    """
    
    def __init__(
        self,
        min_threshold: float = 2.0,
        max_threshold: float = 6.0,
        unit: str = "bits_per_char",
    ):
        """
        Initialize entropy analyzer.
        
        Args:
            min_threshold: Minimum acceptable entropy
            max_threshold: Maximum acceptable entropy (too high = noise)
            unit: Entropy unit ("bits_per_char" or "bits")
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.unit = unit
        
    def calculate(self, text: str) -> float:
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
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def analyze(self, text: str) -> float:
        """
        Analyze text and return normalized quality score.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        entropy = self.calculate(text)
        
        # Normalize to score
        if entropy < self.min_threshold:
            # Too low - repetitive/low information
            return entropy / self.min_threshold * 0.5
        elif entropy > self.max_threshold:
            # Too high - possibly garbled/noise
            excess = entropy - self.max_threshold
            penalty = min(excess / 2.0, 0.5)
            return 1.0 - penalty
        else:
            # In optimal range
            return 0.5 + 0.5 * ((entropy - self.min_threshold) / 
                               (self.max_threshold - self.min_threshold))
    
    def get_character_entropy(self, text: str) -> Dict[str, float]:
        """Get entropy broken down by character type."""
        categories = {
            'lowercase': '',
            'uppercase': '',
            'digits': '',
            'punctuation': '',
            'whitespace': '',
            'other': '',
        }
        
        for c in text:
            if c.islower():
                categories['lowercase'] += c
            elif c.isupper():
                categories['uppercase'] += c
            elif c.isdigit():
                categories['digits'] += c
            elif c.isspace():
                categories['whitespace'] += c
            elif not c.isalnum():
                categories['punctuation'] += c
            else:
                categories['other'] += c
        
        return {
            cat: self.calculate(chars) if chars else 0.0
            for cat, chars in categories.items()
        }


class RepetitionDetector(BaseQualityAnalyzer):
    """
    Detects repetitive patterns in text.
    
    Repetition can indicate:
    - Model stalling/generating filler
    - Copy-paste errors
    - Low-quality synthetic data
    """
    
    def __init__(
        self,
        char_repetition_threshold: float = 0.5,
        line_repetition_threshold: float = 0.7,
        ngram_thresholds: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize repetition detector.
        
        Args:
            char_repetition_threshold: Max ratio of repeated characters
            line_repetition_threshold: Max ratio of duplicate lines
            ngram_thresholds: Thresholds for n-gram repetition by n
        """
        self.char_repetition_threshold = char_repetition_threshold
        self.line_repetition_threshold = line_repetition_threshold
        self.ngram_thresholds = ngram_thresholds or {
            2: 0.5,  # Bigram
            3: 0.4,  # Trigram
            4: 0.3,  # 4-gram
        }
        
    def analyze(self, text: str) -> float:
        """
        Analyze text and return repetition score.
        
        Returns:
            Repetition score between 0.0 (no repetition) and 1.0 (high repetition)
        """
        if not text or len(text) < 10:
            return 0.0
        
        scores = []
        
        # Character repetition
        char_score = self._char_repetition(text)
        scores.append(char_score)
        
        # Line repetition
        line_score = self._line_repetition(text)
        scores.append(line_score)
        
        # N-gram repetition
        for n, threshold in self.ngram_thresholds.items():
            if len(text) >= n:
                ngram_score = self._ngram_repetition(text, n)
                scores.append(ngram_score)
        
        return max(scores) if scores else 0.0
    
    def _char_repetition(self, text: str) -> float:
        """Calculate character-level repetition."""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        total = len(text)
        
        # Find most common character ratio
        if char_counts:
            most_common = char_counts.most_common(1)[0][1]
            return most_common / total
        return 0.0
    
    def _line_repetition(self, text: str) -> float:
        """Calculate line-level repetition."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 2:
            return 0.0
        
        unique_lines = len(set(lines))
        total_lines = len(lines)
        
        return 1.0 - (unique_lines / total_lines)
    
    def _ngram_repetition(self, text: str, n: int) -> float:
        """Calculate n-gram repetition."""
        if len(text) < n:
            return 0.0
        
        # Generate n-grams (words for n>=2, characters for n=1)
        if n <= 2:
            # Character n-grams
            ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        else:
            # Word n-grams
            words = text.split()
            if len(words) < n:
                return 0.0
            ngrams = [
                ' '.join(words[i:i+n])
                for i in range(len(words) - n + 1)
            ]
        
        if not ngrams:
            return 0.0
        
        unique = len(set(ngrams))
        total = len(ngrams)
        
        return 1.0 - (unique / total)
    
    def detect_repeated_sequences(self, text: str, min_length: int = 10) -> List[Tuple[str, int]]:
        """
        Find explicitly repeated sequences.
        
        Returns:
            List of (sequence, count) tuples
        """
        repeats = []
        
        # Check for repeated substrings
        length = len(text)
        for l in range(min_length, length // 2 + 1):
            seen = {}
            for i in range(length - l + 1):
                substr = text[i:i+l]
                if substr in seen:
                    seen[substr] += 1
                else:
                    seen[substr] = 1
            
            for substr, count in seen.items():
                if count > 1:
                    repeats.append((substr, count))
        
        # Sort by length descending
        repeats.sort(key=lambda x: len(x[0]), reverse=True)
        return repeats[:10]  # Top 10


class InformationGainAnalyzer(BaseQualityAnalyzer):
    """
    Analyze information gain/novelty in text.
    
    Measures how much new information content is present
    relative to typical language patterns.
    """
    
    def __init__(self, reference_corpus: Optional[List[str]] = None):
        """
        Initialize information gain analyzer.
        
        Args:
            reference_corpus: Reference text for comparison (optional)
        """
        self.reference_corpus = reference_corpus
        self.reference_dist = self._build_distribution(reference_corpus or [])
        
    def _build_distribution(self, texts: List[str]) -> Counter:
        """Build word distribution from corpus."""
        dist = Counter()
        for text in texts:
            words = text.lower().split()
            dist.update(words)
        return dist
    
    def analyze(self, text: str) -> float:
        """
        Analyze information gain of text.
        
        Returns:
            Information gain score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        # If no reference, use vocabulary diversity as proxy
        if not self.reference_corpus:
            unique_ratio = len(set(words)) / len(words)
            return min(unique_ratio * 2, 1.0)  # Scale up slightly
        
        # Compare against reference distribution
        novel_words = 0
        for word in set(words):
            if self.reference_dist.get(word, 0) < 2:
                novel_words += 1
        
        return novel_words / len(set(words)) if words else 0.0
    
    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Calculate information gain between two texts.
        
        Returns:
            Ratio of novel content in text2 relative to text1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words2:
            return 0.0
        
        novel = words2 - words1
        return len(novel) / len(words2)


class AnomalyDetector:
    """
    Statistical anomaly detection for text.
    
    Detects:
    - Length anomalies (too short/long)
    - Content anomalies (gibberish, encoding issues)
    - Statistical outliers
    """
    
    def __init__(
        self,
        min_chars: int = 10,
        max_chars: int = 1000000,
        max_tokens: int = 100000,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            min_chars: Minimum acceptable characters
            max_chars: Maximum acceptable characters
            max_tokens: Maximum acceptable tokens
            zscore_threshold: Z-score threshold for outliers
            iqr_multiplier: IQR multiplier for outliers
        """
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        
        # Running statistics
        self.lengths: List[int] = []
        self.token_counts: List[int] = []
        
    def detect(self, text: str, token_count: Optional[int] = None) -> List[str]:
        """
        Detect anomalies in text.
        
        Args:
            text: Input text
            token_count: Optional pre-computed token count
            
        Returns:
            List of anomaly flags
        """
        flags = []
        
        # Length checks
        char_count = len(text)
        if char_count < self.min_chars:
            flags.append(f"too_short:{char_count}")
        if char_count > self.max_chars:
            flags.append(f"too_long:{char_count}")
        
        # Token count check
        if token_count is not None and token_count > self.max_tokens:
            flags.append(f"too_many_tokens:{token_count}")
        
        # Content checks
        if self._is_gibberish(text):
            flags.append("gibberish_detected")
        
        if self._has_encoding_issues(text):
            flags.append("encoding_issues")
        
        if self._excessive_punctuation(text):
            flags.append("excessive_punctuation")
        
        # Statistical outlier check (if we have enough data)
        if len(self.lengths) >= 10:
            if self._is_statistical_outlier(char_count):
                flags.append("statistical_outlier_length")
        
        # Update statistics
        self.lengths.append(char_count)
        if token_count is not None:
            self.token_counts.append(token_count)
        
        # Keep window manageable
        if len(self.lengths) > 10000:
            self.lengths = self.lengths[-5000:]
            self.token_counts = self.token_counts[-5000:]
        
        return flags
    
    def _is_gibberish(self, text: str) -> bool:
        """Detect gibberish text."""
        if not text:
            return False
        
        # Check consonant/vowel ratio (gibberish often has unusual ratios)
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        vowel_count = sum(1 for c in text if c in vowels)
        consonant_count = sum(1 for c in text if c in consonants)
        
        total_letters = vowel_count + consonant_count
        if total_letters < 10:
            return False
        
        vowel_ratio = vowel_count / total_letters
        
        # Normal text has ~40% vowels
        if vowel_ratio < 0.15 or vowel_ratio > 0.6:
            return True
        
        # Check for excessive repetition
        char_entropy = self._calculate_entropy(text)
        if char_entropy < 2.0 and len(text) > 50:
            return True
        
        return False
    
    def _has_encoding_issues(self, text: str) -> bool:
        """Detect potential encoding issues."""
        # Check for replacement characters
        if '\ufffd' in text:
            return True
        
        # Check for high ratio of non-printable characters
        non_printable = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
        if len(text) > 0 and non_printable / len(text) > 0.1:
            return True
        
        return False
    
    def _excessive_punctuation(self, text: str) -> bool:
        """Check for excessive punctuation."""
        punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0:
            punct_ratio = punct_count / len(text)
            return punct_ratio > 0.3  # More than 30% punctuation
        return False
    
    def _is_statistical_outlier(self, length: int) -> bool:
        """Check if length is a statistical outlier."""
        if len(self.lengths) < 10:
            return False
        
        # Z-score method
        m = mean(self.lengths)
        try:
            s = stdev(self.lengths)
            if s > 0:
                zscore = abs(length - m) / s
                if zscore > self.zscore_threshold:
                    return True
        except:
            pass
        
        # IQR method
        sorted_lengths = sorted(self.lengths)
        n = len(sorted_lengths)
        q1 = sorted_lengths[n // 4]
        q3 = sorted_lengths[3 * n // 4]
        iqr = q3 - q1
        
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        
        return length < lower or length > upper
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy."""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        total = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def reset(self):
        """Reset running statistics."""
        self.lengths.clear()
        self.token_counts.clear()


class SafetyRiskScorer:
    """
    Safety risk scoring for text content.
    
    Detects potential safety concerns:
    - Prompt injection patterns
    - Jailbreak attempts
    - Data exfiltration patterns
    """
    
    # Known prompt injection patterns
    PROMPT_INJECTION_PATTERNS = [
        r'ignore\s+(?:previous|above|prior)',
        r'forget\s+(?:previous|above|prior)',
        r'disregard\s+(?:instructions|rules)',
        r'system\s*:\s*',
        r'user\s*:\s*.*assistant\s*:',
        r'\{\{.*?\}\}',  # Template injection
        r'<%.*?%>',  # ERB/ASP style
        r'\$\{.*?\}',  # Variable interpolation
    ]
    
    # Known jailbreak patterns
    JAILBREAK_PATTERNS = [
        r'dan\s*(?:mode|prompt)',
        r'jailbreak',
        r'ignore\s+ethical',
        r'pretend\s+(?:to\s+be|you\s+are)',
        r'role\s*play',
        r'hypothetically',
        r'for\s+educational\s+purposes',
        r'research\s+purposes',
    ]
    
    # Data exfiltration patterns
    EXFILTRATION_PATTERNS = [
        r'send\s+(?:to|data).*?http',
        r'upload\s+to',
        r'base64\s+encode',
        r'encode\s+as',
        r'save\s+to\s+file',
    ]
    
    def __init__(self):
        self.injection_regex = re.compile(
            '|'.join(self.PROMPT_INJECTION_PATTERNS),
            re.IGNORECASE
        )
        self.jailbreak_regex = re.compile(
            '|'.join(self.JAILBREAK_PATTERNS),
            re.IGNORECASE
        )
        self.exfiltration_regex = re.compile(
            '|'.join(self.EXFILTRATION_PATTERNS),
            re.IGNORECASE
        )
        
    def score(self, text: str) -> Tuple[float, List[str]]:
        """
        Calculate safety risk score.
        
        Args:
            text: Input text
            
        Returns:
            (risk_score, detected_categories)
        """
        if not text:
            return 0.0, []
        
        scores = {}
        
        # Check injection patterns
        injection_matches = len(self.injection_regex.findall(text))
        scores['injection'] = min(injection_matches * 0.2, 1.0)
        
        # Check jailbreak patterns
        jailbreak_matches = len(self.jailbreak_regex.findall(text))
        scores['jailbreak'] = min(jailbreak_matches * 0.3, 1.0)
        
        # Check exfiltration patterns
        exfil_matches = len(self.exfiltration_regex.findall(text))
        scores['exfiltration'] = min(exfil_matches * 0.25, 1.0)
        
        # Calculate overall score (max of categories)
        overall_score = max(scores.values()) if scores else 0.0
        
        # Determine detected categories
        categories = [cat for cat, score in scores.items() if score > 0.3]
        
        return overall_score, categories


class QualityAnalyzer:
    """
    Main quality analysis class combining all analyzers.
    
    Provides comprehensive quality assessment for text samples.
    """
    
    def __init__(
        self,
        entropy_min: float = 2.0,
        entropy_max: float = 6.0,
        repetition_max: float = 0.5,
        information_gain_min: float = 0.1,
    ):
        """
        Initialize quality analyzer.
        
        Args:
            entropy_min: Minimum acceptable entropy
            entropy_max: Maximum acceptable entropy
            repetition_max: Maximum acceptable repetition score
            information_gain_min: Minimum acceptable information gain
        """
        self.entropy_analyzer = EntropyAnalyzer(entropy_min, entropy_max)
        self.repetition_detector = RepetitionDetector()
        self.info_gain_analyzer = InformationGainAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.safety_scorer = SafetyRiskScorer()
        
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.repetition_max = repetition_max
        self.information_gain_min = information_gain_min
        
    def analyze(
        self,
        text: str,
        token_count: Optional[int] = None,
    ) -> QualityScore:
        """
        Perform complete quality analysis.
        
        Args:
            text: Input text
            token_count: Optional token count
            
        Returns:
            QualityScore with all metrics
        """
        # Run all analyzers
        entropy_raw = self.entropy_analyzer.calculate(text)
        entropy_score = self.entropy_analyzer.analyze(text)
        
        repetition_raw = self.repetition_detector.analyze(text)
        repetition_score = 1.0 - repetition_raw  # Invert: lower repetition = higher score
        
        info_gain = self.info_gain_analyzer.analyze(text)
        
        anomaly_flags = self.anomaly_detector.detect(text, token_count)
        
        safety_risk, risk_categories = self.safety_scorer.score(text)
        
        # Calculate overall score (weighted average)
        weights = {
            'entropy': 0.25,
            'repetition': 0.25,
            'information_gain': 0.3,
            'safety': 0.2,
        }
        
        overall = (
            weights['entropy'] * entropy_score +
            weights['repetition'] * repetition_score +
            weights['information_gain'] * info_gain +
            weights['safety'] * (1.0 - safety_risk)
        )
        
        # Penalize for anomalies
        anomaly_penalty = min(len(anomaly_flags) * 0.1, 0.5)
        overall = max(0.0, overall - anomaly_penalty)
        
        return QualityScore(
            overall_score=overall,
            entropy_score=entropy_score,
            repetition_score=repetition_score,
            information_gain_score=info_gain,
            safety_risk_score=safety_risk,
            anomaly_flags=anomaly_flags,
            details={
                'entropy_raw': entropy_raw,
                'repetition_raw': repetition_raw,
                'risk_categories': risk_categories,
            }
        )
    
    def is_acceptable(self, text: str, token_count: Optional[int] = None) -> bool:
        """Quick check if text meets quality thresholds."""
        score = self.analyze(text, token_count)
        return (
            score.overall_score >= 0.4 and
            not score.anomaly_flags and
            score.safety_risk_score < 0.5
        )
