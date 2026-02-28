"""
Command-line interface for dataset forensics.

Provides:
- Config-driven execution
- Deterministic run IDs
- Structured logging
- Progress tracking
- Output generation
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import package modules
from dataset_forensics.ingestion import (
    AutoDetectParser,
    FileShardIterator,
    IngestionStats,
)
from dataset_forensics.normalization import TextNormalizer, NormalizationConfig, UnicodeForm, LineEnding
from dataset_forensics.tokenization import TokenizationEngine, TokenizerBenchmark
from dataset_forensics.metadata import MetadataExtractor, RowMetadata, compute_sha256
from dataset_forensics.dedup import HybridDedupEngine
from dataset_forensics.quality import QualityAnalyzer


# Configure structured logging
class StructuredLogFormatter(logging.Formatter):
    """JSON structured log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'run_id'):
            log_data['run_id'] = record.run_id
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class SimpleLogFormatter(logging.Formatter):
    """Simple human-readable log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        run_id = getattr(record, 'run_id', '')
        context = getattr(record, 'context', '')
        
        if run_id:
            return f"[{timestamp}] [{record.levelname}] [{run_id}] {record.getMessage()}"
        return f"[{timestamp}] [{record.levelname}] {record.getMessage()}"


def setup_logging(config: Dict[str, Any], run_id: str) -> logging.Logger:
    """
    Set up logging with proper formatters and handlers.
    
    Args:
        config: Logging configuration
        run_id: Run identifier for log correlation
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, config.get('level', 'INFO').upper())
    log_format = config.get('format', 'structured')
    
    logger = logging.getLogger('dataset_forensics')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    if log_format == 'structured':
        formatter = StructuredLogFormatter()
    else:
        formatter = SimpleLogFormatter()
    
    # Console handler
    if config.get('console', {}).get('enabled', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.get('file', {}).get('enabled', True):
        log_dir = Path(config['file'].get('path', 'reports/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"run_{run_id}_{datetime.utcnow():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_run_id(config: Dict[str, Any]) -> str:
    """
    Generate deterministic run ID based on timestamp and config hash.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Deterministic run ID string
    """
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Hash config for determinism
    config_str = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    return f"{timestamp}_{config_hash}"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_file_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_dependency_versions() -> Dict[str, str]:
    """Get versions of key dependencies."""
    deps = {}
    
    packages = [
        'pandas',
        'pyarrow',
        'numpy',
        'tiktoken',
        'transformers',
        'ijson',
        'yaml',
    ]
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            deps[pkg] = getattr(module, '__version__', 'unknown')
        except ImportError:
            deps[pkg] = 'not_installed'
    
    # Python version
    deps['python'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    return deps


class ForensicsPipeline:
    """
    Main forensics processing pipeline.

    Orchestrates ingestion, normalization, tokenization,
    deduplication, quality analysis, and output generation.
    """

    def __init__(self, config: Dict[str, Any], run_id: str, logger: logging.Logger, max_rows: Optional[int] = None):
        self.config = config
        self.run_id = run_id
        self.logger = logger
        self.max_rows = max_rows  # None means process all rows
        
        # Initialize components
        self.parser = AutoDetectParser()
        self.normalizer = self._init_normalizer()
        self.tokenizers = self._init_tokenizers()
        self.metadata_extractor = None  # Will use primary tokenizer
        self.dedup_engine = self._init_dedup()
        self.quality_analyzer = self._init_quality()
        
        # Storage for results
        self.row_metadata: List[RowMetadata] = []
        self.tokenizer_benchmarks: Dict[str, TokenizerBenchmark] = {}
        self.input_checksums: Dict[str, str] = {}
        
    def _init_normalizer(self) -> TextNormalizer:
        """Initialize text normalizer from config."""
        norm_config = self.config.get('normalization', {})
        unicode_form_str = norm_config.get('unicode_form', 'NFC')
        line_ending_str = norm_config.get('line_endings', 'LF')
        config = NormalizationConfig(
            unicode_form=UnicodeForm(unicode_form_str),
            normalize_whitespace=norm_config.get('normalize_whitespace', True),
            remove_control_chars=norm_config.get('remove_control_chars', True),
            remove_zero_width=norm_config.get('remove_zero_width', True),
            line_ending=LineEnding(line_ending_str) if line_ending_str in ('LF', 'CRLF') else LineEnding.LF,
        )
        return TextNormalizer(config)
    
    def _init_tokenizers(self) -> Dict[str, TokenizationEngine]:
        """Initialize tokenizers from config."""
        tokenizers = {}
        
        for tok_config in self.config.get('tokenizers', []):
            name = tok_config['name']
            backend = tok_config.get('backend', 'tiktoken')
            
            if backend == 'tiktoken':
                encoding = tok_config.get('version', name)
                tokenizers[name] = TokenizationEngine(encoding)
            elif backend == 'huggingface':
                model = tok_config.get('model', name)
                tokenizers[name] = TokenizationEngine(f"hf_{model}")
            elif backend == 'character':
                tokenizers[name] = TokenizationEngine('character')
        
        return tokenizers
    
    def _init_dedup(self) -> Optional[HybridDedupEngine]:
        """Initialize deduplication engine."""
        dedup_config = self.config.get('deduplication', {})
        
        if not dedup_config.get('exact', {}).get('enabled', True) and \
           not dedup_config.get('near', {}).get('enabled', True):
            return None
        
        # Normalize algorithm name: 'minhash_lsh' -> 'minhash'
        raw_algo = dedup_config.get('near', {}).get('algorithm', 'minhash')
        near_method = 'minhash' if 'minhash' in raw_algo else raw_algo
        return HybridDedupEngine(
            enable_exact=dedup_config.get('exact', {}).get('enabled', True),
            enable_near=dedup_config.get('near', {}).get('enabled', True),
            near_method=near_method,
        )
    
    def _init_quality(self) -> QualityAnalyzer:
        """Initialize quality analyzer."""
        quality_config = self.config.get('quality', {})
        
        return QualityAnalyzer(
            entropy_min=quality_config.get('entropy', {}).get('min', 2.0),
            entropy_max=quality_config.get('entropy', {}).get('max', 6.0),
            repetition_max=quality_config.get('repetition', {}).get('max_character_ratio', 0.5),
            information_gain_min=quality_config.get('information_gain', {}).get('min', 0.1),
        )
    
    def discover_inputs(self) -> List[Path]:
        """Discover input files based on configuration."""
        input_config = self.config.get('input', {})
        paths = input_config.get('paths', ['data/input'])
        patterns = input_config.get('patterns', ['*.json', '*.jsonl', '*.jsonl.gz'])
        
        iterator = FileShardIterator(paths, patterns)
        files = list(iterator)
        
        self.logger.info(f"Discovered {len(files)} input files")
        return files
    
    def process_file(self, filepath: Path) -> List[RowMetadata]:
        """Process a single input file."""
        self.logger.info(f"Processing file: {filepath}")
        
        # Compute checksum
        self.input_checksums[str(filepath)] = compute_file_checksum(filepath)
        
        results = []
        default_tokenizer = self.config.get('default_tokenizer', 'cl100k_base')
        primary_tokenizer = self.tokenizers.get(default_tokenizer)
        
        if not primary_tokenizer:
            primary_tokenizer = list(self.tokenizers.values())[0]
        
        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor(primary_tokenizer)
        
        row_count = 0
        for record in self.parser.parse(filepath):
            # Respect max_rows limit if set
            if self.max_rows is not None and row_count >= self.max_rows:
                self.logger.info(f"Reached max_rows limit ({self.max_rows}), stopping.")
                break

            try:
                # Extract text from record
                text = self._extract_text(record)
                if not text:
                    continue

                # Normalize text
                text = self.normalizer.normalize(text)

                # Generate sample ID
                sample_id = f"{self.run_id}_{filepath.name}_{row_count}"

                # Extract metadata
                metadata = self.metadata_extractor.extract(
                    text=text,
                    sample_id=sample_id,
                    source=str(filepath),
                )

                # Store original text for benchmarking
                metadata._additional['text'] = text

                # Run deduplication
                if self.dedup_engine:
                    dedup_result = self.dedup_engine.add(sample_id, text)
                    if dedup_result.is_duplicate:
                        metadata.exact_dup_group = dedup_result.duplicate_of
                        metadata.near_dup_cluster_id = dedup_result.cluster_id

                # Run quality analysis
                quality = self.quality_analyzer.analyze(text, metadata.token_count)
                metadata.quality_score = quality.overall_score
                metadata.entropy_score = quality.entropy_score
                metadata.repetition_score = quality.repetition_score
                metadata.safety_risk_score = quality.safety_risk_score
                metadata.anomaly_flags = quality.anomaly_flags

                results.append(metadata)
                row_count += 1

                # Progress logging
                if row_count % self.config.get('processing', {}).get('progress_interval', 1000) == 0:
                    self.logger.info(f"Processed {row_count} rows from {filepath.name}")

            except Exception as e:
                self.logger.error(f"Error processing record {row_count}: {e}")
                continue
        
        self.logger.info(f"Completed {filepath.name}: {row_count} rows")
        return results
    
    def _extract_text(self, record: Dict[str, Any]) -> str:
        """Extract text content from a record."""
        # Handle different formats
        if 'text' in record:
            return record['text']
        elif 'content' in record:
            return record['content']
        elif 'messages' in record:
            # ChatGPT export format
            messages = record.get('messages', [])
            texts = []
            for msg in messages:
                content = msg.get('content', '')
                if content:
                    role = msg.get('role', 'unknown')
                    texts.append(f"[{role}]: {content}")
            return '\n\n'.join(texts)
        elif 'mapping' in record:
            # Alternative ChatGPT format
            mapping = record.get('mapping', {})
            texts = []
            for node_id, node in mapping.items():
                if isinstance(node, dict):
                    msg = node.get('message', {})
                    if isinstance(msg, dict):
                        content = msg.get('content', {})
                        if isinstance(content, dict):
                            parts = content.get('parts', [])
                            text = ' '.join(str(p) for p in parts if p)
                        else:
                            text = str(content)
                        if text:
                            role = msg.get('author', {}).get('role', 'unknown')
                            texts.append(f"[{role}]: {text}")
            return '\n\n'.join(texts)
        else:
            # Fallback: stringify entire record
            return json.dumps(record, default=str)
    
    def run_benchmarks(self) -> Dict[str, TokenizerBenchmark]:
        """Run tokenizer benchmarks on a sample."""
        benchmarks = {}
        
        # Use first few texts for benchmarking
        sample_texts = [m._additional.get('text', '') for m in self.row_metadata[:100]]
        
        for name, tokenizer in self.tokenizers.items():
            for text in sample_texts:
                try:
                    tokenizer.tokenize(text)
                except Exception as e:
                    self.logger.debug(f"Benchmark error for {name}: {e}")
            
            benchmarks[name] = tokenizer.get_stats()
        
        return benchmarks
    
    def generate_outputs(self, output_dir: Path):
        """Generate all output files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_config = self.config.get('output', {})
        
        # 1. Generate token_row_metrics.parquet
        self._generate_parquet(output_dir / output_config.get('metrics_parquet', 'token_row_metrics.parquet'))
        
        # 2. Generate tokenizer_benchmark.csv
        self._generate_benchmark_csv(output_dir / output_config.get('benchmark_csv', 'tokenizer_benchmark.csv'))
        
        # 3. Generate repro_manifest.json
        self._generate_manifest(output_dir / output_config.get('manifest_json', 'repro_manifest.json'))
    
    def _generate_parquet(self, filepath: Path):
        """Generate Parquet file with row metadata."""
        try:
            import pandas as pd
            
            # Convert to DataFrame
            data = [m.to_dict() for m in self.row_metadata]
            df = pd.DataFrame(data)
            
            # Write parquet
            compression = self.config.get('output', {}).get('parquet_compression', 'zstd')
            df.to_parquet(filepath, compression=compression, index=False)
            
            self.logger.info(f"Generated: {filepath}")
        except ImportError:
            self.logger.warning("pandas/pyarrow not installed, skipping parquet generation")
            # Fallback to JSON
            json_path = filepath.with_suffix('.jsonl')
            with open(json_path, 'w', encoding='utf-8') as f:
                for m in self.row_metadata:
                    f.write(json.dumps(m.to_dict(), default=str) + '\n')
            self.logger.info(f"Generated JSONL fallback: {json_path}")
    
    def _generate_benchmark_csv(self, filepath: Path):
        """Generate CSV with tokenizer benchmarks."""
        try:
            import pandas as pd
            
            data = []
            for name, stats in self.tokenizer_benchmarks.items():
                row = stats.to_dict()
                row['tokenizer'] = name
                data.append(row)
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
                self.logger.info(f"Generated: {filepath}")
        except ImportError:
            self.logger.warning("pandas not installed, skipping benchmark CSV")
    
    def _generate_manifest(self, filepath: Path):
        """Generate reproducibility manifest."""
        manifest = {
            'run_id': self.run_id,
            'timestamp': datetime.utcnow().isoformat(),
            'config_hash': hashlib.sha256(
                json.dumps(self.config, sort_keys=True, default=str).encode()
            ).hexdigest(),
            'input_files': self.input_checksums,
            'output_files': {},
            'git_commit': self._get_git_commit(),
            'dependency_versions': get_dependency_versions(),
            'stats': {
                'total_rows': len(self.row_metadata),
                'dedup_stats': self.dedup_engine.get_stats() if self.dedup_engine else None,
            }
        }
        
        # Compute output file checksums
        for output_file in filepath.parent.iterdir():
            if output_file.is_file():
                manifest['output_files'][str(output_file.name)] = compute_file_checksum(output_file)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        self.logger.info(f"Generated: {filepath}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def run(self):
        """Execute the full pipeline."""
        self.logger.info(f"Starting forensics pipeline: {self.run_id}")
        
        # Discover inputs
        input_files = self.discover_inputs()
        if not input_files:
            self.logger.error("No input files found!")
            return
        
        # Process each file
        for filepath in input_files:
            results = self.process_file(filepath)
            self.row_metadata.extend(results)
        
        self.logger.info(f"Total rows processed: {len(self.row_metadata)}")
        
        # Run benchmarks
        self.logger.info("Running tokenizer benchmarks...")
        self.tokenizer_benchmarks = self.run_benchmarks()
        
        # Generate outputs
        output_dir = Path(self.config.get('output', {}).get('directory', 'reports'))
        self.logger.info(f"Generating outputs to: {output_dir}")
        self.generate_outputs(output_dir)
        
        self.logger.info(f"Pipeline complete: {self.run_id}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='dataset_forensics',
        description='Production-grade token forensics system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dataset_forensics --config config.yaml
  python dataset_forensics/cli.py --config config.yaml
  python -m dataset_forensics --config config.yaml --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='dataset_forensics/config.yaml',
        help='Path to configuration file (default: dataset_forensics/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        help='Override generated run ID'
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of conversations/rows to process (default: all)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Load configuration
    try:
        config = load_config(parsed_args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {parsed_args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        return 1
    
    # Override logging level if verbose
    if parsed_args.verbose:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = 'DEBUG'
    
    # Generate or use provided run ID
    run_id = parsed_args.run_id or generate_run_id(config)
    
    # Set up logging
    logger = setup_logging(config.get('logging', {}), run_id)
    logger.info(f"Dataset Forensics v1.0.0 - Run ID: {run_id}")
    
    # Run pipeline
    try:
        pipeline = ForensicsPipeline(config, run_id, logger, max_rows=parsed_args.max_rows)
        pipeline.run()
        return 0
    except Exception as e:
        logger.exception("Pipeline failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
