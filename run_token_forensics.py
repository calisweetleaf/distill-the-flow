#!/usr/bin/env python3
"""
SOTA++ Token Forensics Pipeline - Main Execution Script

Run comprehensive token-level forensics on training datasets.
Part of Project Decentralize SOTA - Drop 3: Dataset Intelligence Toolkit

Usage:
    python run_token_forensics.py <input_file> [--output-dir reports]

Example:
    python run_token_forensics.py 02-14-26-ChatGPT/conversations.json

Author: Daeron (Christian Trey Levi Rowell)
License: Sovereign Anti-Exploitation Software License
"""

import sys
import argparse
from pathlib import Path

from token_forensics_orchestrator import TokenForensicsOrchestrator, OrchestratorConfig
from token_forensics_agents import (
    DataProfilerAgent,
    MultiTokenizerAgent,
    QualityScoringAgent,
    SafetyPIIAgent,
    DedupAgent,
    CostModelAgent,
    VerifierAgent
)


def parse_args():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="SOTA++ Token Forensics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on ChatGPT export
  python run_token_forensics.py 02-14-26-ChatGPT/conversations.json

  # Custom output directory
  python run_token_forensics.py data.json --output-dir ./analysis

  # Debug mode (fail fast)
  python run_token_forensics.py data.json --fail-fast

Output artifacts (in output-dir):
  - data_profile.json & .md
  - tokenization_results.json
  - token_row_metrics.parquet
  - tokenizer_benchmark.csv
  - quality_scores.json
  - quality_risk_report.json
  - pii_safety_results.json
  - pii_safety_report.json
  - dedup_results.json
  - dedup_clusters.parquet
  - cost_projection.json
  - verification_report.json
  - repro_manifest.json
"""
    )
    
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to dataset file (JSON format)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports/)"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first agent failure (default: run all agents)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=7,
        help="Max parallel workers (default: 7)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--disable-agents",
        nargs="+",
        choices=["DataProfilerAgent", "MultiTokenizerAgent", "QualityScoringAgent", 
                 "SafetyPIIAgent", "DedupAgent", "CostModelAgent"],
        help="Agents to disable (Verifier always runs)"
    )
    
    return parser.parse_args()


def main():
    """Main execution entry point."""
    
    args = parse_args()
    
    # Validate input
    if not args.input_file.exists():
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Configure
    # Default: None means all agents enabled
    enabled_agents = None
    
    # If user disabled specific agents, build enabled list
    if args.disable_agents:
        all_agents = [
            "DataProfilerAgent", "MultiTokenizerAgent", "QualityScoringAgent",
            "SafetyPIIAgent", "DedupAgent", "CostModelAgent", "VerifierAgent"
        ]
        enabled_agents = [a for a in all_agents if a not in args.disable_agents]
    
    config = OrchestratorConfig(
        input_path=args.input_file,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        fail_fast=args.fail_fast,
        random_seed=args.seed,
        enabled_agents=enabled_agents
    )
    
    # Initialize orchestrator
    orchestrator = TokenForensicsOrchestrator(config)
    
    # Register agents
    orchestrator.register_agent(DataProfilerAgent(config))
    orchestrator.register_agent(MultiTokenizerAgent(config))
    orchestrator.register_agent(QualityScoringAgent(config))
    orchestrator.register_agent(SafetyPIIAgent(config))
    orchestrator.register_agent(DedupAgent(config))
    orchestrator.register_agent(CostModelAgent(config))
    orchestrator.register_agent(VerifierAgent(config))
    
    # Execute pipeline
    try:
        success = orchestrator.run()
        
        if success:
            print("\n" + "="*70)
            print("🎉 PIPELINE COMPLETE")
            print("="*70)
            print(f"📂 Reports: {config.output_dir.absolute()}")
            print("✅ All artifacts generated and verified")
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("❌ PIPELINE FAILED")
            print("="*70)
            print(f"📂 Check failure manifest: {config.output_dir / 'failure_manifest.json'}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
