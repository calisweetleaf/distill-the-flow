"""
SOTA++ Token Forensics Pipeline Orchestrator

Production-grade multi-agent token analysis system for training data preparation.
Part of Project Decentralize SOTA - Drop 3: Dataset Intelligence Toolkit

Author: Daeron (Christian Trey Levi Rowell)
License: Sovereign Anti-Exploitation Software License
Repository: https://github.com/calisweetleaf/distill-the-flow

Architecture:
    - 7 specialized agents running in coordinated parallel execution
    - Strict verification gates with failure manifest
    - Reproducibility-first design with cryptographic hashing
    - Machine-readable + human-readable dual outputs
    - Zero safety filtering - raw forensics only
"""

import json
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OrchestratorConfig:
    """Global configuration for token forensics pipeline."""
    
    # I/O paths
    input_path: Path
    output_dir: Path = Path("reports")
    
    # Execution
    max_workers: int = 7  # One per agent
    fail_fast: bool = False  # Stop on first agent failure
    strict_verification: bool = True
    
    # Reproducibility
    random_seed: int = 42
    include_timestamps: bool = True
    generate_repro_hash: bool = True
    
    # Agent enablement (for debugging/partial runs)
    # None = all agents enabled (default), or provide list of agent class names to enable
    enabled_agents: Optional[List[str]] = None
    
    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class AgentResult:
    """Wrapper for agent execution results."""
    
    def __init__(self, agent_name: str, success: bool, 
                 data: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None,
                 execution_time: float = 0.0):
        self.agent_name = agent_name
        self.success = success
        self.data = data or {}
        self.error = error
        self.execution_time = execution_time
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_seconds": round(self.execution_time, 3),
            "timestamp": self.timestamp
        }


class BaseAgent:
    """Abstract base for all forensic agents."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.name = self.__class__.__name__
    
    def execute(self, shared_context: Dict[str, Any]) -> AgentResult:
        """Execute agent logic with error handling."""
        start_time = time.time()
        
        try:
            result_data = self._run(shared_context)
            elapsed = time.time() - start_time
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result_data,
                execution_time=elapsed
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{self.name} failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=error_msg,
                execution_time=elapsed
            )
    
    def _run(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses."""
        raise NotImplementedError(f"{self.name}._run() not implemented")
    
    def emit_artifact(self, filename: str, content: Any, format: str = "json"):
        """Write agent output artifact."""
        output_path = self.config.output_dir / filename
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        elif format == "md":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📄 {self.name} → {output_path.name}")


class TokenForensicsOrchestrator:
    """Main orchestration engine for SOTA++ token forensics."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[BaseAgent] = []
        self.results: Dict[str, AgentResult] = {}
        self.shared_context: Dict[str, Any] = {
            "config": config,
            "start_time": datetime.utcnow().isoformat(),
            "pipeline_version": "1.0.0-sota++",
            "dataset_path": str(config.input_path)
        }
    
    def register_agent(self, agent: BaseAgent):
        """Add an agent to the execution graph."""
        # If enabled_agents is None, enable all agents
        # Otherwise, check if agent name is in the enabled list
        if self.config.enabled_agents is None or agent.name in self.config.enabled_agents:
            self.agents.append(agent)
            print(f"✅ Registered: {agent.name}")
        else:
            print(f"⏭️  Skipped: {agent.name} (not enabled)")
    
    def run(self) -> bool:
        """Execute all agents with dependency-aware parallelization."""
        
        print("\n" + "="*70)
        print("SOTA++ TOKEN FORENSICS PIPELINE")
        print("="*70)
        print(f"Dataset: {self.config.input_path}")
        print(f"Agents: {len(self.agents)}")
        print(f"Output: {self.config.output_dir}")
        print("="*70 + "\n")
        
        # Phase 0: DataProfiler must run first (serial) - populates shared_context["conversations"]
        data_profiler = next((a for a in self.agents if a.name == "DataProfilerAgent"), None)
        if data_profiler:
            print("📊 Phase 0: Data Profiling (Serial - Required First)")
            print("-" * 70)
            result = data_profiler.execute(self.shared_context)
            self.results[data_profiler.name] = result
            status = "✅" if result.success else "❌"
            print(f"{status} {data_profiler.name} ({result.execution_time:.2f}s)")
            
            if result.success:
                self.shared_context[f"{data_profiler.name}_output"] = result.data
            
            if not result.success and self.config.fail_fast:
                print("\n⚠️  Fail-fast enabled. DataProfiler failed. Aborting pipeline.")
                return False
            print()
        
        # Phase 1: Independent agents (parallel) - exclude DataProfiler, Verifier, and CostModel
        independent_agents = [
            a for a in self.agents
            if a.name not in ["VerifierAgent", "DataProfilerAgent", "CostModelAgent"]
        ]
        cost_model_agent = next((a for a in self.agents if a.name == "CostModelAgent"), None)
        
        if independent_agents:
            print("[Phase 1] Forensic Analysis (Parallel)")
            print("-" * 70)
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(agent.execute, self.shared_context): agent
                    for agent in independent_agents
                }
                
                for future in as_completed(futures):
                    agent = futures[future]
                    result = future.result()
                    self.results[agent.name] = result
                    
                    status = "PASS" if result.success else "FAIL"
                    print(f"{status} {agent.name} ({result.execution_time:.2f}s)")
                    
                    # Update shared context with results
                    if result.success:
                        self.shared_context[f"{agent.name}_output"] = result.data
                    
                    # Fail fast if configured
                    if not result.success and self.config.fail_fast:
                        print("\nFail-fast enabled. Aborting pipeline.")
                        return False

        if cost_model_agent:
            print("\n[Phase 1B] Cost Modeling (Serial dependency on tokenization)")
            print("-" * 70)
            result = cost_model_agent.execute(self.shared_context)
            self.results[cost_model_agent.name] = result

            status = "PASS" if result.success else "FAIL"
            print(f"{status} {cost_model_agent.name} ({result.execution_time:.2f}s)")

            if result.success:
                self.shared_context[f"{cost_model_agent.name}_output"] = result.data

            if not result.success and self.config.fail_fast:
                print("\nFail-fast enabled. CostModel failed. Aborting pipeline.")
                return False
        
        # Phase 2: Verifier (serial, requires all previous results)
        print("\n🔍 Phase 2: Verification")
        print("-" * 70)
        
        verifier = next((a for a in self.agents if a.name == "VerifierAgent"), None)
        if verifier:
            result = verifier.execute(self.shared_context)
            self.results["VerifierAgent"] = result
            
            status = "✅" if result.success else "❌"
            print(f"{status} VerifierAgent ({result.execution_time:.2f}s)")
        
        # Final status
        print("\n" + "="*70)
        success_count = sum(1 for r in self.results.values() if r.success)
        total_count = len(self.results)
        
        if success_count == total_count:
            print(f"✅ PIPELINE COMPLETE: {success_count}/{total_count} agents passed")
            self._generate_manifest()
            return True
        else:
            print(f"❌ PIPELINE FAILED: {success_count}/{total_count} agents passed")
            try:
                self._generate_manifest()
            except Exception as e:
                print(f"Manifest generation failed during failure path: {e}")
            self._generate_failure_manifest()
            return False
    
    def _generate_manifest(self):
        """Generate reproducibility manifest with real checksums and actual data."""
        
        # Collect all artifact files with real checksums and metadata
        files_info = {}
        for artifact_path in sorted(self.config.output_dir.glob("*")):
            if artifact_path.is_file() and artifact_path.name != "repro_manifest.json":
                file_info = {
                    "path": str(artifact_path.relative_to(self.config.output_dir)),
                    "size_bytes": artifact_path.stat().st_size,
                    "checksum": self._hash_file(artifact_path)
                }
                
                # Add row/column counts if we have dataframe info
                if artifact_path.suffix == '.parquet':
                    df = self._try_load_parquet(artifact_path)
                    if df is not None:
                        file_info["rows"] = len(df)
                        file_info["columns"] = len(df.columns)
                elif artifact_path.suffix == '.csv':
                    df = self._try_load_csv(artifact_path)
                    if df is not None:
                        file_info["rows"] = len(df)
                        file_info["columns"] = len(df.columns)
                
                files_info[artifact_path.name] = file_info
        
        # Derive dataset summary from actual processed data
        conversations = self.shared_context.get("conversations", [])
        tokenization_df = self.shared_context.get("tokenization_df")
        
        dataset_summary = {
            "total_samples": len(conversations),
            "total_tokens": 0,
            "sources": [],
            "splits": {}
        }
        
        if tokenization_df is not None:
            # Get token counts from first tokenizer column
            token_cols = [c for c in tokenization_df.columns if c.startswith("tokens_")]
            if token_cols:
                dataset_summary["total_tokens"] = int(tokenization_df[token_cols[0]].sum())
        
        manifest = {
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": self.shared_context["pipeline_version"],
            "execution_timestamp": self.shared_context["start_time"],
            "dataset_path": str(self.config.input_path),
            "dataset_hash": self._hash_file(self.config.input_path),
            "config": asdict(self.config),
            "files": files_info,
            "dataset_summary": dataset_summary,
            "agent_results": {
                name: result.to_dict() 
                for name, result in self.results.items()
            },
            "reproducibility_hash": self._compute_repro_hash(),
            "checksum_algorithm": "sha256"
        }
        
        output_path = self.config.output_dir / "repro_manifest.json"
        # Generate post-manifest artifacts first so checksums are stable.
        self._generate_token_ledger(manifest)
        self._ensure_raw_only_layout()

        # Refresh file checksums after post-manifest artifacts are written.
        refreshed_files = {}
        for artifact_path in sorted(self.config.output_dir.glob("*")):
            if artifact_path.is_file() and artifact_path.name != "repro_manifest.json":
                file_info = {
                    "path": str(artifact_path.relative_to(self.config.output_dir)),
                    "size_bytes": artifact_path.stat().st_size,
                    "checksum": self._hash_file(artifact_path),
                }

                if artifact_path.suffix == ".parquet":
                    df = self._try_load_parquet(artifact_path)
                    if df is not None:
                        file_info["rows"] = len(df)
                        file_info["columns"] = len(df.columns)
                elif artifact_path.suffix == ".csv":
                    df = self._try_load_csv(artifact_path)
                    if df is not None:
                        file_info["rows"] = len(df)
                        file_info["columns"] = len(df.columns)

                refreshed_files[artifact_path.name] = file_info

        manifest["files"] = refreshed_files
        manifest["reproducibility_hash"] = self._compute_repro_hash()

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"ðŸ“‹ Manifest â†’ {output_path.name}")
    
    def _generate_token_ledger(self, manifest: Dict[str, Any]):
        """Generate canonical token ledger with four required counters.
        
        Per MOONSHINE_DISTRIBUTED_PLAN:
        1. raw_json_tokens - o200k_base over full JSON text
        2. content_tokens_non_system - user/assistant/tool text only
        3. content_tokens_cleaned - post-policy filtering
        4. distilled_tokens_selected - final training/query subset
        """
        
        tokenization_df = self.shared_context.get("tokenization_df")
        source_hash = manifest.get("dataset_hash", "")
        
        raw_json_tokens = 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("o200k_base")
            with open(self.config.input_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            raw_json_tokens = len(enc.encode(raw_content))
        except Exception:
            pass
        
        content_tokens_non_system = 0
        if tokenization_df is not None:
            token_cols = [c for c in tokenization_df.columns if c.startswith("tokens_")]
            if token_cols:
                content_tokens_non_system = int(tokenization_df[token_cols[0]].sum())
        
        ledger = {
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_sha256": source_hash,
            "source_path": str(self.config.input_path),
            "tokenizer": "o200k_base",
            "parsing_scope": "message_content_non_system",
            "counters": {
                # Canonical aliases expected by run_validation.py
                "canonical_tokens": raw_json_tokens,
                "distilled_tokens_excluded": None,
                # Original field names (kept for backward compatibility)
                "raw_json_tokens": raw_json_tokens,
                "content_tokens_non_system": content_tokens_non_system,
                "content_tokens_cleaned": None,
                "distilled_tokens_selected": None
            },
            "notes": {
                "raw_json_tokens": "Full file tokenization including JSON structure",
                "content_tokens_non_system": "User/assistant/tool message content only, excludes system messages",
                "content_tokens_cleaned": "Post-policy filtering (computed by distillation stage)",
                "distilled_tokens_selected": "Final training/query subset (computed by distillation stage)"
            },
            "run_id": self.shared_context.get("start_time", ""),
            "pipeline_version": self.shared_context.get("pipeline_version", "1.0.0")
        }
        
        ledger_path = self.config.output_dir / "token_ledger.json"
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f, indent=2)
        
        print(f"📊 Token Ledger → {ledger_path.name}")
        
        return ledger
    
    def _ensure_raw_only_layout(self):
        """Mirror canonical parquet path expected by raw_only validator."""
        src = self.config.output_dir / "token_row_metrics.parquet"
        if not src.exists():
            return

        canonical_dir = self.config.output_dir / "canonical"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        dst = canonical_dir / "token_row_metrics.raw.parquet"

        try:
            should_copy = (not dst.exists()) or (src.stat().st_mtime > dst.stat().st_mtime)
            if should_copy:
                shutil.copy2(src, dst)
                rel = dst.relative_to(self.config.output_dir)
                print(f"[CANONICAL] Mirrored {rel}")
        except Exception as e:
            print(f"[WARN] Could not mirror canonical parquet layout: {e}")

    def _try_load_parquet(self, path: Path):
        """Try to load a parquet file, return None on failure."""
        try:
            import pandas as pd
            return pd.read_parquet(path)
        except Exception:
            return None
    
    def _try_load_csv(self, path: Path):
        """Try to load a CSV file, return None on failure."""
        try:
            import pandas as pd
            return pd.read_csv(path)
        except Exception:
            return None
    
    def _generate_failure_manifest(self):
        """Generate failure diagnostic manifest."""
        failures = {
            name: result.to_dict()
            for name, result in self.results.items()
            if not result.success
        }
        
        manifest = {
            "pipeline_version": self.shared_context["pipeline_version"],
            "execution_timestamp": self.shared_context["start_time"],
            "failed_agents": failures,
            "successful_agents": [
                name for name, result in self.results.items() if result.success
            ]
        }
        
        output_path = self.config.output_dir / "failure_manifest.json"
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"⚠️  Failure Manifest → {output_path.name}")
    
    def _hash_file(self, filepath: Path) -> str:
        """SHA256 hash of input file."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _compute_repro_hash(self) -> str:
        """Reproducibility hash over all outputs."""
        hasher = hashlib.sha256()
        
        # Hash all JSON artifacts in sorted order
        for artifact_path in sorted(self.config.output_dir.glob("*.json")):
            if artifact_path.name != "repro_manifest.json":
                hasher.update(artifact_path.read_bytes())
        
        return hasher.hexdigest()


# Export public API
__all__ = [
    "OrchestratorConfig",
    "TokenForensicsOrchestrator",
    "BaseAgent",
    "AgentResult"
]


if __name__ == "__main__":
    print(__doc__)
