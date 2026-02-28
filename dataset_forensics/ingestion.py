"""
Streaming JSON/JSONL ingestion for large datasets.

Handles ChatGPT export format, gzipped files, and memory-efficient streaming.
"""

import gzip
import ijson
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union
import os

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics for ingestion process."""
    files_processed: int = 0
    rows_parsed: int = 0
    rows_failed: int = 0
    bytes_read: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def rows_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.rows_parsed / self.duration_seconds
        return 0.0


class StreamingParser(ABC):
    """Abstract base class for streaming parsers."""
    
    @abstractmethod
    def parse(self, filepath: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """Parse file and yield records."""
        pass
    
    @abstractmethod
    def parse_bytes(self, data: bytes) -> Generator[Dict[str, Any], None, None]:
        """Parse bytes and yield records."""
        pass


class StreamingJSONParser(StreamingParser):
    """
    Streaming JSON parser for large arrays using ijson.
    
    Handles:
    - Large JSON arrays (GB scale)
    - Malformed JSON with graceful fallback
    - Progress tracking
    """
    
    def __init__(self, item_path: str = "item", buffer_size: int = 65536):
        self.item_path = item_path
        self.buffer_size = buffer_size
        self.stats = IngestionStats()
        
    def parse(self, filepath: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """
        Stream parse a JSON file.
        
        Args:
            filepath: Path to JSON file (may be gzipped)
            
        Yields:
            Dictionary records from the JSON array
        """
        filepath = Path(filepath)
        self.stats = IngestionStats()
        self.stats.start_time = datetime.utcnow()
        
        logger.info(f"Starting JSON parse: {filepath}")
        
        try:
            if filepath.suffix == '.gz':
                yield from self._parse_gzipped(filepath)
            else:
                yield from self._parse_regular(filepath)
        except Exception as e:
            logger.warning(f"Primary parse failed for {filepath}: {e}")
            logger.info(f"Attempting fallback line-by-line parse...")
            yield from self._fallback_parse(filepath)
        finally:
            self.stats.end_time = datetime.utcnow()
            logger.info(f"JSON parse complete: {self.stats.rows_parsed} rows in {self.stats.duration_seconds:.2f}s")
    
    def _parse_regular(self, filepath: Path) -> Generator[Dict[str, Any], None, None]:
        """Parse regular (non-gzipped) JSON file."""
        with open(filepath, 'rb') as f:
            self.stats.files_processed += 1
            for record in ijson.items(f, self.item_path, use_float=True):
                self.stats.rows_parsed += 1
                self.stats.bytes_read += len(json.dumps(record))
                yield record
    
    def _parse_gzipped(self, filepath: Path) -> Generator[Dict[str, Any], None, None]:
        """Parse gzipped JSON file."""
        with gzip.open(filepath, 'rb') as f:
            self.stats.files_processed += 1
            for record in ijson.items(f, self.item_path, use_float=True):
                self.stats.rows_parsed += 1
                self.stats.bytes_read += len(json.dumps(record))
                yield record
    
    def _fallback_parse(self, filepath: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Fallback line-by-line parser for malformed JSON.
        
        Attempts to extract JSON objects from each line.
        """
        opener = gzip.open if filepath.suffix == '.gz' else open
        mode = 'rt' if filepath.suffix == '.gz' else 'r'
        encoding = 'utf-8' if filepath.suffix != '.gz' else None
        
        with opener(filepath, mode, encoding=encoding, errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line in ('[', ']', ','):
                    continue
                    
                # Remove trailing comma if present
                if line.endswith(','):
                    line = line[:-1]
                    
                try:
                    record = json.loads(line)
                    self.stats.rows_parsed += 1
                    yield record
                except json.JSONDecodeError as e:
                    self.stats.rows_failed += 1
                    if self.stats.rows_failed <= 5:  # Log first 5 errors
                        logger.debug(f"Line {line_num} parse error: {e}")
                    continue
    
    def parse_bytes(self, data: bytes) -> Generator[Dict[str, Any], None, None]:
        """Parse JSON from bytes."""
        try:
            for record in ijson.items(data, self.item_path, use_float=True):
                self.stats.rows_parsed += 1
                yield record
        except Exception as e:
            logger.error(f"Bytes parse error: {e}")
            raise


class StreamingJSONLParser(StreamingParser):
    """
    Streaming JSONL (JSON Lines) parser.
    
    Handles:
    - Standard JSONL format (one JSON object per line)
    - Gzipped JSONL files
    - Line-by-line error recovery
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
        self.stats = IngestionStats()
        
    def parse(self, filepath: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """
        Stream parse a JSONL file.
        
        Args:
            filepath: Path to JSONL file (may be gzipped)
            
        Yields:
            Dictionary records from each line
        """
        filepath = Path(filepath)
        self.stats = IngestionStats()
        self.stats.start_time = datetime.utcnow()
        
        logger.info(f"Starting JSONL parse: {filepath}")
        
        opener = gzip.open if filepath.suffix == '.gz' else open
        mode = 'rt' if filepath.suffix == '.gz' else 'r'
        
        with opener(filepath, mode, encoding=self.encoding, errors='replace') as f:
            self.stats.files_processed += 1
            self.stats.bytes_read = os.path.getsize(filepath)
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                    self.stats.rows_parsed += 1
                    yield record
                except json.JSONDecodeError as e:
                    self.stats.rows_failed += 1
                    if self.stats.rows_failed <= 10:
                        logger.debug(f"Line {line_num} JSON decode error: {e}")
                    continue
        
        self.stats.end_time = datetime.utcnow()
        logger.info(f"JSONL parse complete: {self.stats.rows_parsed} rows in {self.stats.duration_seconds:.2f}s")
    
    def parse_bytes(self, data: bytes) -> Generator[Dict[str, Any], None, None]:
        """Parse JSONL from bytes."""
        text = data.decode(self.encoding, errors='replace')
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                self.stats.rows_parsed += 1
                yield record
            except json.JSONDecodeError:
                self.stats.rows_failed += 1
                continue
    
    def parse_string(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """Parse JSONL from string."""
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                self.stats.rows_parsed += 1
                yield record
            except json.JSONDecodeError:
                self.stats.rows_failed += 1
                continue


class ChatGPTExportParser:
    """
    Parser for ChatGPT export JSON format.
    
    Handles both:
    - Full export: Array of conversation objects
    - Single conversation: Individual conversation object
    """
    
    def __init__(self):
        self.json_parser = StreamingJSONParser(item_path="item")
        self.stats = IngestionStats()
        
    def parse(self, filepath: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """
        Parse ChatGPT export file.
        
        Args:
            filepath: Path to ChatGPT export JSON
            
        Yields:
            Conversation objects with normalized structure
        """
        filepath = Path(filepath)
        self.stats.start_time = datetime.utcnow()
        
        logger.info(f"Parsing ChatGPT export: {filepath}")
        
        for convo in self.json_parser.parse(filepath):
            normalized = self._normalize_conversation(convo)
            if normalized:
                self.stats.rows_parsed += 1
                yield normalized
        
        self.stats.end_time = datetime.utcnow()
        logger.info(f"ChatGPT export parse complete: {self.stats.rows_parsed} conversations")
    
    def _normalize_conversation(self, convo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize conversation to standard format.
        
        Extracts:
        - id: Conversation ID
        - title: Conversation title
        - create_time: Creation timestamp
        - messages: List of message objects
        """
        if not isinstance(convo, dict):
            return None
        
        # Extract basic metadata
        convo_id = convo.get('id', 'unknown')
        title = convo.get('title', 'Untitled')
        create_time = self._extract_timestamp(convo)
        
        # Extract messages from mapping structure
        messages = self._extract_messages(convo)
        
        return {
            'id': convo_id,
            'title': title,
            'create_time': create_time,
            'update_time': convo.get('update_time'),
            'messages': messages,
            'mapping': convo.get('mapping', {}),
            '_source_format': 'chatgpt_export',
        }
    
    def _extract_timestamp(self, convo: Dict[str, Any]) -> Optional[float]:
        """Extract timestamp from conversation or its messages."""
        # Try conversation-level timestamps
        for key in ['create_time', 'update_time']:
            ts = convo.get(key)
            if ts is not None:
                try:
                    return float(ts)
                except (TypeError, ValueError):
                    continue
        
        # Try message-level timestamps
        mapping = convo.get('mapping', {})
        if isinstance(mapping, dict):
            for node in mapping.values():
                if not isinstance(node, dict):
                    continue
                msg = node.get('message', {})
                if isinstance(msg, dict):
                    for key in ['create_time', 'update_time']:
                        ts = msg.get(key)
                        if ts is not None:
                            try:
                                return float(ts)
                            except (TypeError, ValueError):
                                continue
        return None
    
    def _extract_messages(self, convo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract messages from conversation mapping."""
        messages = []
        mapping = convo.get('mapping', {})
        
        if not isinstance(mapping, dict):
            return messages
        
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            
            msg = node.get('message')
            if not isinstance(msg, dict):
                continue
            
            # Extract role
            author = msg.get('author', {})
            role = author.get('role', 'unknown')
            
            # Extract content
            content_obj = msg.get('content', {})
            if isinstance(content_obj, dict):
                parts = content_obj.get('parts', [])
                content = ' '.join(str(p) for p in parts if p)
                content_type = content_obj.get('content_type', 'text')
            else:
                content = str(content_obj)
                content_type = 'text'
            
            # Extract timestamp
            msg_time = None
            for key in ['create_time', 'update_time']:
                ts = msg.get(key)
                if ts is not None:
                    try:
                        msg_time = float(ts)
                        break
                    except (TypeError, ValueError):
                        continue
            
            messages.append({
                'id': node_id,
                'role': role,
                'content': content,
                'content_type': content_type,
                'timestamp': msg_time,
                'author': author,
                'metadata': msg.get('metadata', {}),
            })
        
        # Sort by timestamp if available
        messages.sort(key=lambda m: (m['timestamp'] is None, m['timestamp'] or 0))
        return messages
    
    def extract_text_content(self, convo: Dict[str, Any]) -> str:
        """Extract all text content from conversation as single string."""
        texts = []
        for msg in convo.get('messages', []):
            content = msg.get('content', '')
            if content:
                texts.append(f"[{msg.get('role', 'unknown')}]: {content}")
        return '\n\n'.join(texts)
    
    def extract_code_blocks(self, convo: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract code blocks from conversation."""
        import re
        code_blocks = []
        
        for msg in convo.get('messages', []):
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')
            
            # Match markdown code blocks
            pattern = r'```(\w*)\n(.*?)```'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for lang, code in matches:
                code_blocks.append({
                    'language': lang.strip().lower() or 'text',
                    'code': code,
                    'role': role,
                    'message_id': msg.get('id'),
                })
        
        return code_blocks


class FileShardIterator:
    """
    Iterator over file shards matching patterns.
    
    Handles:
    - Directory traversal with pattern matching
    - Glob patterns
    - Automatic format detection
    """
    
    def __init__(
        self,
        paths: List[Union[str, Path]],
        patterns: List[str] = None,
        recursive: bool = True,
    ):
        self.paths = [Path(p) for p in paths]
        self.patterns = patterns or ['*.json', '*.jsonl', '*.jsonl.gz']
        self.recursive = recursive
        self.stats = IngestionStats()
        
    def __iter__(self) -> Generator[Path, None, None]:
        """Iterate over all matching files."""
        seen: Set[Path] = set()
        
        for base_path in self.paths:
            if not base_path.exists():
                logger.warning(f"Path does not exist: {base_path}")
                continue
            
            if base_path.is_file():
                if base_path not in seen:
                    seen.add(base_path)
                    self.stats.files_processed += 1
                    yield base_path
            elif base_path.is_dir():
                yield from self._iterate_directory(base_path, seen)
    
    def _iterate_directory(
        self,
        directory: Path,
        seen: Set[Path]
    ) -> Generator[Path, None, None]:
        """Iterate over files in directory matching patterns."""
        for pattern in self.patterns:
            if self.recursive:
                matches = directory.rglob(pattern)
            else:
                matches = directory.glob(pattern)
            
            for filepath in matches:
                if filepath.is_file() and filepath not in seen:
                    seen.add(filepath)
                    self.stats.files_processed += 1
                    yield filepath
    
    def get_files(self) -> List[Path]:
        """Get list of all matching files."""
        return list(self)
    
    def count_files(self) -> int:
        """Count matching files."""
        return sum(1 for _ in self)


class AutoDetectParser(StreamingParser):
    """
    Auto-detecting parser that selects appropriate parser based on file extension.
    """
    
    def __init__(self):
        self.json_parser = StreamingJSONParser()
        self.jsonl_parser = StreamingJSONLParser()
        self.chatgpt_parser = ChatGPTExportParser()
        self.stats = IngestionStats()
        
    def parse(self, filepath: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """Parse file with auto-detected format."""
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        stem = filepath.stem.lower()
        
        # Detect gzipped files
        if ext == '.gz':
            ext = Path(stem).suffix.lower()
            stem = Path(stem).stem.lower()
        
        # Choose parser based on extension
        if ext == '.jsonl':
            logger.debug(f"Using JSONL parser for {filepath}")
            yield from self.jsonl_parser.parse(filepath)
        elif ext == '.json':
            # Check if it's a ChatGPT export
            if 'conversation' in stem.lower() or 'chatgpt' in stem.lower():
                logger.debug(f"Using ChatGPT parser for {filepath}")
                yield from self.chatgpt_parser.parse(filepath)
            else:
                logger.debug(f"Using JSON parser for {filepath}")
                yield from self.json_parser.parse(filepath)
        else:
            # Default to JSONL for unknown extensions
            logger.debug(f"Defaulting to JSONL parser for {filepath}")
            yield from self.jsonl_parser.parse(filepath)
    
    def parse_bytes(self, data: bytes) -> Generator[Dict[str, Any], None, None]:
        """Parse bytes with auto-detection."""
        # Try JSONL first (more common for streaming)
        try:
            yield from self.jsonl_parser.parse_bytes(data)
        except Exception:
            # Fall back to JSON
            yield from self.json_parser.parse_bytes(data)


def coerce_timestamp(raw: Any) -> Optional[float]:
    """Coerce value to timestamp float."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def format_timestamp(ts: Optional[float]) -> str:
    """Format timestamp for display."""
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except (TypeError, ValueError, OSError):
        return "invalid"


def get_month_key(ts: Optional[float]) -> str:
    """Get YYYY-MM key from timestamp."""
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m")
    except (TypeError, ValueError, OSError):
        return "unknown"
