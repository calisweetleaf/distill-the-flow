"""
Text normalization utilities.

Handles:
- Unicode normalization (NFC, NFD, NFKC, NFKD)
- Whitespace normalization
- Control character removal
- Line ending normalization
"""

import re
import unicodedata
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Pattern

logger = logging.getLogger(__name__)


class UnicodeForm(Enum):
    """Unicode normalization forms."""
    NFC = "NFC"
    NFD = "NFD"
    NFKC = "NFKC"
    NFKD = "NFKD"


class LineEnding(Enum):
    """Line ending styles."""
    LF = "\n"
    CRLF = "\r\n"
    NATIVE = None  # Use platform default


@dataclass
class NormalizationConfig:
    """Configuration for text normalization."""
    unicode_form: UnicodeForm = UnicodeForm.NFC
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    remove_zero_width: bool = True
    line_ending: LineEnding = LineEnding.LF
    max_line_length: Optional[int] = None
    collapse_repeated_newlines: bool = True
    strip_leading_trailing: bool = True


class UnicodeNormalizer:
    """
    Unicode normalization handler.
    
    Provides NFC/NFD/NFKC/NFKD normalization to ensure
    consistent character representation.
    """
    
    # Zero-width characters to optionally remove
    ZERO_WIDTH_CHARS = {
        '\u200B',  # Zero Width Space
        '\u200C',  # Zero Width Non-Joiner
        '\u200D',  # Zero Width Joiner
        '\uFEFF',  # Zero Width No-Break Space (BOM)
        '\u2060',  # Word Joiner
        '\u00AD',  # Soft Hyphen
    }
    
    # Control characters to remove (except whitespace)
    CONTROL_CHARS = set(chr(i) for i in range(32)) - {
        '\t', '\n', '\r', '\x0b', '\x0c'  # Keep whitespace controls
    }
    
    def __init__(self, form: UnicodeForm = UnicodeForm.NFC):
        self.form = form
        
    def normalize(self, text: str) -> str:
        """
        Apply Unicode normalization.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        return unicodedata.normalize(self.form.value, text)
    
    def is_normalized(self, text: str) -> bool:
        """Check if text is already in the target normalization form."""
        return unicodedata.is_normalized(self.form.value, text)
    
    def get_composition_info(self, text: str) -> dict:
        """Get information about character composition."""
        categories = {}
        for char in text:
            cat = unicodedata.category(char)
            categories[cat] = categories.get(cat, 0) + 1
        return categories


class WhitespaceNormalizer:
    """
    Whitespace normalization handler.
    
    Handles:
    - Tab to space conversion
    - Multiple space collapsing
    - Line ending normalization
    - Leading/trailing whitespace removal
    """
    
    # Pattern for multiple whitespace characters
    MULTI_SPACE_PATTERN: Pattern = re.compile(r'[ \t]+')
    MULTI_NEWLINE_PATTERN: Pattern = re.compile(r'\n{3,}')
    LEADING_TRAILING_WS_PATTERN: Pattern = re.compile(r'^[ \t]+|[ \t]+$', re.MULTILINE)
    
    def __init__(
        self,
        tab_width: int = 4,
        preserve_indent: bool = True,
        line_ending: LineEnding = LineEnding.LF,
    ):
        self.tab_width = tab_width
        self.preserve_indent = preserve_indent
        self.line_ending = line_ending
        
    def normalize(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Whitespace-normalized text
        """
        if not text:
            return text
        
        # Normalize line endings first
        text = self._normalize_line_endings(text)
        
        # Handle tabs
        if self.preserve_indent:
            text = self._convert_tabs_preserve_indent(text)
        else:
            text = text.replace('\t', ' ' * self.tab_width)
        
        # Collapse multiple spaces (but not at line starts if preserving indent)
        text = self.MULTI_SPACE_PATTERN.sub(' ', text)
        
        # Collapse repeated newlines
        text = self.MULTI_NEWLINE_PATTERN.sub('\n\n', text)
        
        return text
    
    def _normalize_line_endings(self, text: str) -> str:
        """Normalize line endings to specified format."""
        # First normalize all to LF
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Then convert to target if not LF
        if self.line_ending == LineEnding.CRLF:
            text = text.replace('\n', '\r\n')
        
        return text
    
    def _convert_tabs_preserve_indent(self, text: str) -> str:
        """Convert tabs to spaces while preserving indentation structure."""
        lines = text.split('\n')
        result = []
        
        for line in lines:
            # Find leading tabs
            stripped = line.lstrip('\t')
            leading_tabs = len(line) - len(stripped)
            
            if leading_tabs > 0:
                # Convert leading tabs to spaces
                spaces = ' ' * (leading_tabs * self.tab_width)
                line = spaces + stripped
            
            # Convert remaining tabs in line
            line = line.replace('\t', ' ' * self.tab_width)
            result.append(line)
        
        return '\n'.join(result)
    
    def collapse_lines(self, text: str, max_consecutive: int = 2) -> str:
        """Collapse consecutive empty lines."""
        pattern = re.compile(rf'\n{{{max_consecutive + 1},}}')
        replacement = '\n' * max_consecutive
        return pattern.sub(replacement, text)


class TextNormalizer:
    """
    Main text normalization class combining all normalization steps.
    
    Provides a comprehensive normalization pipeline for text data.
    """
    
    # Common problematic patterns
    CONTROL_CHAR_PATTERN: Pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    BOM_PATTERN: Pattern = re.compile('[\ufeff\ufffe\uffff]')
    REPLACEMENT_CHAR_PATTERN: Pattern = re.compile('\ufffd')
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self.unicode_normalizer = UnicodeNormalizer(self.config.unicode_form)
        self.whitespace_normalizer = WhitespaceNormalizer(
            line_ending=self.config.line_ending
        )
        
    def normalize(self, text: str) -> str:
        """
        Apply full normalization pipeline.
        
        Pipeline order:
        1. Unicode normalization
        2. Control character removal
        3. Zero-width character removal
        4. Whitespace normalization
        5. Leading/trailing trim
        
        Args:
            text: Input text
            
        Returns:
            Fully normalized text
        """
        if not text:
            return ""
        
        # 1. Unicode normalization
        text = self.unicode_normalizer.normalize(text)
        
        # 2. Remove control characters
        if self.config.remove_control_chars:
            text = self.CONTROL_CHAR_PATTERN.sub('', text)
        
        # 3. Remove zero-width characters
        if self.config.remove_zero_width:
            for char in UnicodeNormalizer.ZERO_WIDTH_CHARS:
                text = text.replace(char, '')
        
        # 4. BOM removal
        text = self.BOM_PATTERN.sub('', text)
        
        # 5. Whitespace normalization
        if self.config.normalize_whitespace:
            text = self.whitespace_normalizer.normalize(text)
        
        # 6. Leading/trailing trim
        if self.config.strip_leading_trailing:
            text = text.strip()
        
        # 7. Collapse repeated newlines
        if self.config.collapse_repeated_newlines:
            text = self.whitespace_normalizer.collapse_lines(text, max_consecutive=2)
        
        # 8. Max line length enforcement
        if self.config.max_line_length:
            text = self._enforce_max_line_length(text, self.config.max_line_length)
        
        return text
    
    def _enforce_max_line_length(self, text: str, max_length: int) -> str:
        """Break overly long lines."""
        lines = text.split('\n')
        result = []
        
        for line in lines:
            while len(line) > max_length:
                # Find break point
                break_point = max_length
                # Try to break at word boundary
                while break_point > max_length * 0.8 and line[break_point] != ' ':
                    break_point -= 1
                if break_point <= max_length * 0.8:
                    break_point = max_length  # Hard break
                
                result.append(line[:break_point])
                line = line[break_point:].lstrip()
            
            result.append(line)
        
        return '\n'.join(result)
    
    def normalize_batch(self, texts: list) -> list:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]
    
    def get_normalization_stats(self, original: str, normalized: str) -> dict:
        """Get statistics about the normalization performed."""
        return {
            'original_length': len(original),
            'normalized_length': len(normalized),
            'bytes_removed': len(original.encode('utf-8')) - len(normalized.encode('utf-8')),
            'length_change': len(normalized) - len(original),
            'was_modified': original != normalized,
        }


def normalize_text(
    text: str,
    unicode_form: str = "NFC",
    normalize_whitespace: bool = True,
    remove_control: bool = True,
) -> str:
    """
    Convenience function for quick text normalization.
    
    Args:
        text: Input text
        unicode_form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
        normalize_whitespace: Whether to normalize whitespace
        remove_control: Whether to remove control characters
        
    Returns:
        Normalized text
    """
    config = NormalizationConfig(
        unicode_form=UnicodeForm(unicode_form),
        normalize_whitespace=normalize_whitespace,
        remove_control_chars=remove_control,
    )
    normalizer = TextNormalizer(config)
    return normalizer.normalize(text)


def is_valid_utf8(text: str) -> bool:
    """Check if text contains valid UTF-8 (no replacement characters)."""
    return '\ufffd' not in text


def detect_encoding_issues(text: str) -> list:
    """Detect potential encoding issues in text."""
    issues = []
    
    # Check for replacement characters
    if '\ufffd' in text:
        issues.append('replacement_characters')
    
    # Check for BOM
    if text.startswith('\ufeff'):
        issues.append('leading_bom')
    
    # Check for mixed line endings
    has_crlf = '\r\n' in text
    has_lf_only = bool(re.search(r'[^\r]\n', text)) or text.startswith('\n')
    has_cr_only = '\r' in text.replace('\r\n', '')
    
    if sum([has_crlf, has_lf_only, has_cr_only]) > 1:
        issues.append('mixed_line_endings')
    
    # Check for control characters
    if any(ord(c) < 32 and c not in '\t\n\r' for c in text):
        issues.append('control_characters')
    
    # Check for invalid Unicode sequences
    try:
        text.encode('utf-8').decode('utf-8', 'strict')
    except UnicodeError:
        issues.append('invalid_utf8')
    
    return issues


def fix_common_encoding_issues(text: str) -> str:
    """Fix common encoding issues found in text data."""
    # Fix Windows-1252 mojibake (common for smart quotes)
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201C': '"',  # Left double quote
        '\u201D': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u00A0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text
