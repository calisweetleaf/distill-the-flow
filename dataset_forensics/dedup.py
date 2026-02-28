"""
Deduplication engine for dataset forensics.

Supports:
- Exact deduplication using SHA256
- Near deduplication using MinHash + LSH
- Alternative: SimHash for near dedup
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Callable
from collections import defaultdict
import binascii

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """Result of deduplication operation."""
    is_duplicate: bool
    duplicate_of: Optional[str] = None  # sample_id of original
    cluster_id: Optional[str] = None
    similarity: float = 0.0
    method: str = ""


@dataclass
class DedupStats:
    """Statistics for deduplication process."""
    total_samples: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    unique_samples: int = 0
    clusters_formed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_samples': self.total_samples,
            'exact_duplicates': self.exact_duplicates,
            'near_duplicates': self.near_duplicates,
            'unique_samples': self.unique_samples,
            'clusters_formed': self.clusters_formed,
            'exact_dup_rate': self.exact_duplicates / max(self.total_samples, 1),
            'near_dup_rate': self.near_duplicates / max(self.total_samples, 1),
            'dedup_ratio': (self.total_samples - self.unique_samples) / max(self.total_samples, 1),
        }


class DedupEngine(ABC):
    """Abstract base class for deduplication engines."""
    
    @abstractmethod
    def add(self, sample_id: str, text: str) -> DedupResult:
        """Add a sample and check for duplicates."""
        pass
    
    @abstractmethod
    def get_stats(self) -> DedupStats:
        """Get deduplication statistics."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the engine state."""
        pass


class ExactDedup(DedupEngine):
    """
    Exact deduplication using SHA256 hashing.
    
    Fast and memory-efficient for detecting identical samples.
    """
    
    def __init__(self):
        self.seen_hashes: Dict[str, str] = {}  # hash -> sample_id
        self.stats = DedupStats()
        
    def add(self, sample_id: str, text: str) -> DedupResult:
        """
        Add sample and check for exact duplicates.
        
        Args:
            sample_id: Unique sample identifier
            text: Sample text
            
        Returns:
            DedupResult with duplicate status
        """
        self.stats.total_samples += 1
        
        # Compute hash
        if isinstance(text, str):
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        else:
            text_hash = hashlib.sha256(text).hexdigest()
        
        # Check for duplicate
        if text_hash in self.seen_hashes:
            self.stats.exact_duplicates += 1
            return DedupResult(
                is_duplicate=True,
                duplicate_of=self.seen_hashes[text_hash],
                method="exact_sha256"
            )
        
        # Store hash
        self.seen_hashes[text_hash] = sample_id
        self.stats.unique_samples += 1
        
        return DedupResult(is_duplicate=False, method="exact_sha256")
    
    def get_stats(self) -> DedupStats:
        return self.stats
    
    def reset(self):
        self.seen_hashes.clear()
        self.stats = DedupStats()
    
    def get_hash(self, text: str) -> str:
        """Get SHA256 hash of text."""
        if isinstance(text, str):
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        return hashlib.sha256(text).hexdigest()


class MinHashDedup(DedupEngine):
    """
    Near deduplication using MinHash + LSH (Locality Sensitive Hashing).
    
    Efficiently finds similar documents without comparing all pairs.
    
    Based on:
    - Broder, A. "On the resemblance and containment of documents"
    - MinHash for Jaccard similarity estimation
    - LSH banding technique for sublinear candidate selection
    """
    
    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        shingle_size: int = 5,
        seed: int = 42,
    ):
        """
        Initialize MinHash deduplication.
        
        Args:
            num_hashes: Number of hash functions (signature length)
            num_bands: Number of bands for LSH
            shingle_size: Character n-gram size for shingles
            seed: Random seed for reproducibility
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size
        self.seed = seed
        
        # LSH buckets
        self.buckets: List[Dict[Tuple, Set[str]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]
        
        # Store signatures and sample info
        self.signatures: Dict[str, List[int]] = {}
        self.sample_texts: Dict[str, str] = {}
        
        # Generate hash parameters
        self._init_hash_params()
        
        self.stats = DedupStats()
        
    def _init_hash_params(self):
        """Initialize hash function parameters."""
        import random
        rand = random.Random(self.seed)
        
        # For each hash function, generate (a, b) parameters for (ax + b) % p
        # Using large prime
        self.prime = (1 << 61) - 1  # Mersenne prime
        self.hash_params = [
            (rand.randint(1, self.prime - 1), rand.randint(0, self.prime - 1))
            for _ in range(self.num_hashes)
        ]
        
    def _shingle(self, text: str) -> Set[int]:
        """
        Create shingles (character n-grams) from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of shingle hashes
        """
        if len(text) < self.shingle_size:
            return {hash(text) & 0xFFFFFFFF}
        
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            shingles.add(hash(shingle) & 0xFFFFFFFF)
        return shingles
    
    def _compute_signature(self, shingles: Set[int]) -> List[int]:
        """
        Compute MinHash signature.
        
        Args:
            shingles: Set of shingle hashes
            
        Returns:
            MinHash signature (list of minimum hash values)
        """
        if not shingles:
            return [self.prime] * self.num_hashes
        
        signature = []
        for a, b in self.hash_params:
            min_hash = self.prime
            for shingle_hash in shingles:
                # (a * x + b) % p
                hash_val = ((a * shingle_hash + b) % self.prime)
                min_hash = min(min_hash, hash_val)
            signature.append(min_hash)
        
        return signature
    
    def _lsh_candidates(self, signature: List[int]) -> Set[str]:
        """
        Find candidate matches using LSH.
        
        Args:
            signature: MinHash signature
            
        Returns:
            Set of candidate sample IDs
        """
        candidates = set()
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            # Add all samples in this bucket
            bucket = self.buckets[band_idx][band_signature]
            candidates.update(bucket)
        
        return candidates
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Estimated Jaccard similarity
        """
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def add(self, sample_id: str, text: str) -> DedupResult:
        """
        Add sample and check for near duplicates.
        
        Args:
            sample_id: Unique sample identifier
            text: Sample text
            
        Returns:
            DedupResult with duplicate status and similarity
        """
        self.stats.total_samples += 1
        
        # Compute signature
        shingles = self._shingle(text)
        signature = self._compute_signature(shingles)
        
        # Find candidates
        candidates = self._lsh_candidates(signature)
        
        # Check actual similarity with candidates
        best_match = None
        best_similarity = 0.0
        
        for candidate_id in candidates:
            if candidate_id in self.signatures:
                similarity = self._jaccard_similarity(
                    signature, self.signatures[candidate_id]
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate_id
        
        # Add to LSH buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            self.buckets[band_idx][band_signature].add(sample_id)
        
        # Store signature
        self.signatures[sample_id] = signature
        self.sample_texts[sample_id] = text
        
        # Determine if duplicate
        threshold = 0.85  # Jaccard similarity threshold
        if best_similarity >= threshold:
            self.stats.near_duplicates += 1
            return DedupResult(
                is_duplicate=True,
                duplicate_of=best_match,
                similarity=best_similarity,
                method="minhash_lsh"
            )
        
        self.stats.unique_samples += 1
        return DedupResult(
            is_duplicate=False,
            similarity=best_similarity,
            method="minhash_lsh"
        )
    
    def get_stats(self) -> DedupStats:
        return self.stats
    
    def reset(self):
        self.buckets = [defaultdict(set) for _ in range(self.num_bands)]
        self.signatures.clear()
        self.sample_texts.clear()
        self.stats = DedupStats()
    
    def find_cluster(self, sample_id: str) -> List[str]:
        """Find all samples in the same cluster as given sample."""
        if sample_id not in self.signatures:
            return []
        
        sig = self.signatures[sample_id]
        cluster = []
        
        for other_id, other_sig in self.signatures.items():
            if other_id != sample_id:
                similarity = self._jaccard_similarity(sig, other_sig)
                if similarity >= 0.85:
                    cluster.append(other_id)
        
        return cluster


class SimHashDedup(DedupEngine):
    """
    Near deduplication using SimHash.
    
    SimHash is a locality-sensitive hash that produces similar hashes
    for similar documents. Hamming distance between hashes approximates
    document similarity.
    """
    
    def __init__(
        self,
        hash_bits: int = 64,
        max_hamming_distance: int = 3,
        shingle_size: int = 4,
    ):
        """
        Initialize SimHash deduplication.
        
        Args:
            hash_bits: Number of bits in hash (64 or 128)
            max_hamming_distance: Max Hamming distance for duplicates
            shingle_size: Character n-gram size
        """
        self.hash_bits = hash_bits
        self.max_hamming_distance = max_hamming_distance
        self.shingle_size = shingle_size
        
        self.hashes: Dict[str, int] = {}
        self.stats = DedupStats()
        
    def _shingle(self, text: str) -> List[str]:
        """Create character shingles."""
        if len(text) < self.shingle_size:
            return [text]
        return [
            text[i:i + self.shingle_size]
            for i in range(len(text) - self.shingle_size + 1)
        ]
    
    def _compute_simhash(self, text: str) -> int:
        """
        Compute SimHash of text.
        
        Uses feature hashing with weighted bits.
        """
        shingles = self._shingle(text)
        
        # Initialize bit vector
        vec = [0] * self.hash_bits
        
        for shingle in shingles:
            # Hash shingle to get bit pattern
            hash_val = hash(shingle)
            
            for i in range(self.hash_bits):
                bit = (hash_val >> i) & 1
                if bit:
                    vec[i] += 1
                else:
                    vec[i] -= 1
        
        # Build final hash
        simhash = 0
        for i in range(self.hash_bits):
            if vec[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _hamming_distance(self, h1: int, h2: int) -> int:
        """Calculate Hamming distance between two hashes."""
        xor = h1 ^ h2
        return bin(xor).count('1')
    
    def add(self, sample_id: str, text: str) -> DedupResult:
        """
        Add sample and check for near duplicates.
        
        Args:
            sample_id: Unique sample identifier
            text: Sample text
            
        Returns:
            DedupResult with duplicate status
        """
        self.stats.total_samples += 1
        
        # Compute SimHash
        simhash = self._compute_simhash(text)
        
        # Check against existing hashes
        best_match = None
        best_distance = self.hash_bits
        
        for other_id, other_hash in self.hashes.items():
            distance = self._hamming_distance(simhash, other_hash)
            if distance < best_distance:
                best_distance = distance
                best_match = other_id
        
        # Store hash
        self.hashes[sample_id] = simhash
        
        # Calculate similarity (1 - normalized Hamming distance)
        similarity = 1.0 - (best_distance / self.hash_bits)
        
        # Check if duplicate
        if best_distance <= self.max_hamming_distance:
            self.stats.near_duplicates += 1
            return DedupResult(
                is_duplicate=True,
                duplicate_of=best_match,
                similarity=similarity,
                method="simhash"
            )
        
        self.stats.unique_samples += 1
        return DedupResult(
            is_duplicate=False,
            similarity=similarity,
            method="simhash"
        )
    
    def get_stats(self) -> DedupStats:
        return self.stats
    
    def reset(self):
        self.hashes.clear()
        self.stats = DedupStats()


class HybridDedupEngine:
    """
    Hybrid deduplication combining exact and near dedup.
    
    First checks for exact duplicates, then near duplicates.
    """
    
    def __init__(
        self,
        enable_exact: bool = True,
        enable_near: bool = True,
        near_method: str = "minhash",  # "minhash" or "simhash"
    ):
        self.enable_exact = enable_exact
        self.enable_near = enable_near
        
        self.exact = ExactDedup() if enable_exact else None
        
        if enable_near:
            if near_method == "minhash":
                self.near = MinHashDedup()
            elif near_method == "simhash":
                self.near = SimHashDedup()
            else:
                raise ValueError(f"Unknown near method: {near_method}")
        else:
            self.near = None
        
        self.duplicate_groups: Dict[str, List[str]] = defaultdict(list)
        self.cluster_counter = 0
        
    def add(self, sample_id: str, text: str) -> DedupResult:
        """
        Add sample through both dedup engines.
        
        Args:
            sample_id: Unique sample identifier
            text: Sample text
            
        Returns:
            DedupResult with combined duplicate status
        """
        # Check exact first
        if self.exact:
            result = self.exact.add(sample_id, text)
            if result.is_duplicate:
                # Add to group
                self.duplicate_groups[result.duplicate_of].append(sample_id)
                return result
        
        # Check near
        if self.near:
            result = self.near.add(sample_id, text)
            if result.is_duplicate:
                return result
        
        return DedupResult(is_duplicate=False)
    
    def get_cluster_id(self, sample_id: str) -> Optional[str]:
        """Get cluster ID for a sample (if any)."""
        if sample_id in self.duplicate_groups:
            return f"exact_{sample_id}"
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {
            'exact': self.exact.get_stats().to_dict() if self.exact else None,
            'near': self.near.get_stats().to_dict() if self.near else None,
        }
        
        total = (
            (self.exact.stats.total_samples if self.exact else 0) or
            (self.near.stats.total_samples if self.near else 0)
        )
        
        exact_dups = self.exact.stats.exact_duplicates if self.exact else 0
        near_dups = self.near.stats.near_duplicates if self.near else 0
        
        stats['combined'] = {
            'total_samples': total,
            'total_duplicates': exact_dups + near_dups,
            'duplicate_rate': (exact_dups + near_dups) / max(total, 1),
        }
        
        return stats
    
    def reset(self):
        """Reset all engines."""
        if self.exact:
            self.exact.reset()
        if self.near:
            self.near.reset()
        self.duplicate_groups.clear()
