# Enhanced Implementation Roadmap

## Overview

This document outlines the three-phase enhancement roadmap to replace stub implementations with production-ready algorithms. Each phase builds on the existing GPU acceleration framework (v0.4.0) and introduces SIMD-optimized algorithms for HMM/MSA and advanced phylogenetic inference.

---

## Phase 1: GPU Kernel Implementation (TARGET: v0.5.0)

### Current State

The GPU acceleration framework (v0.4.0) provides:
- ✅ CUDA/HIP/Vulkan device abstraction
- ✅ Memory management infrastructure
- ✅ Dispatcher with strategy selection
- 🔄 **Stub GPU kernels** in `src/futures/gpu.rs`

### What Needs to Be Replaced

**File:** `src/futures/gpu.rs`

**Current stubs:**
```rust
pub fn execute_smith_waterman_gpu(
    device: &GpuDevice,
    seq1: &[u8],
    seq2: &[u8],
) -> Result<(Vec<i32>, usize, usize, i32), GpuError> {
    // Currently returns placeholder zero vectors
}

pub fn execute_needleman_wunsch_gpu(
    device: &GpuDevice,
    seq1: &[u8],
    seq2: &[u8],
) -> Result<Vec<i32>, GpuError> {
    // Currently returns placeholder zero vectors
}
```

### Implementation Strategy

#### Phase 1a: CUDA Runtime Kernels

**Why**: NVIDIA GPUs dominate HPC and cloud computing for bioinformatics

**Implementation**:
1. **Actual CUDA PTX Compilation**
   - Replace stub with real `cudarc` runtime compilation
   - Use `nvrtc::compile_ptx()` with optimized CUDA C++ kernel source
   - Implement shared memory optimization for scoring matrix

2. **Smith-Waterman CUDA Kernel**
   ```cuda
   // Optimizations:
   // - Shared memory for 24×24 scoring matrix (576 bytes)
   // - Block-striped DP computation
   // - Warp-level reductions for maximum score
   // - Coalesced global memory access
   // - CTA (Cooperative Thread Array) synchronization
   ```
   - **Expected Speedup**: 80-150× vs scalar on RTX 3090
   - **Support**: Sequences up to GPU memory limit (8-24GB)

3. **Needleman-Wunsch CUDA Kernel**
   ```cuda
   // Similar optimization strategy plus:
   // - Boundary condition handling via blocks
   // - Anti-diagonal DP tile computation
   // - Memory persistence across blocks via device memory
   ```

4. **Memory Optimization**
   - Use unified memory for seamless data transfer
   - Implement pinned memory for host-GPU communication
   - Add memory pooling for batch operations

**Files to Create**:
- `src/alignment/kernel/cuda_impl.rs` - Actual kernel implementation
- `src/utils/cuda_utils.rs` - Helper functions for CUDA operations

**Performance Targets**:
- Single alignment: 100-200× speedup (100K sequences)
- Batch alignment: 150-300× speedup (with memory reuse)

#### Phase 1b: HIP Runtime Kernels (AMD GPUs)

**Why**: ROCm provides open-source CUDA alternative for AMD CDNA/RDNA GPUs

**Implementation**:
1. Replace HIP stubs with real `hip-sys` FFI calls
2. Implement same kernel optimizations as CUDA
3. Add architecture-specific tuning for GFX906+ (MI100/MI200)
4. Support CDNA tensor cores for potential enhancement

**Files to Create**:
- `src/alignment/kernel/hip_impl.rs` - Real HIP kernel code
- `benches/hip_benchmarks.rs` - HIP-specific performance tests

**Performance Targets**:
- Comparable to CUDA on equivalent hardware
- 70-140× speedup on MI100

#### Phase 1c: Vulkan Compute Validation

**Why**: Validate cross-platform compute shader approach

**Implementation**:
1. Create GLSL compute shader compilation pipeline
2. Implement SPIR-V binary loading and caching
3. Add descriptor set management for large buffers
4. Implement workgroup size optimization for various GPUs

**Files to Create**:
- `src/alignment/kernel/vulkan_impl.rs` - Real Vulkan compute
- `shaders/smith_waterman.glsl` - GLSL compute shader
- `shaders/needleman_wunsch.glsl` - GLSL compute shader

**Performance Targets**:
- 60-120× speedup on modern Vulkan-capable GPUs
- Universal cross-platform support

### Testing Strategy

```rust
#[cfg(test)]
mod cuda_tests {
    // Correctness verification against scalar baseline
    #[test]
    fn test_cuda_smith_waterman_correctness() {
        // Compare CUDA result with scalar reference
    }
    
    // Performance validation
    #[test]
    fn test_cuda_alignment_speedup() {
        // Verify speedup meets target (>50x)
    }
}
```

### Completion Criteria

- ✅ All GPU kernels compile without errors
- ✅ Output matches scalar reference implementation (bit-perfect)
- ✅ Speedup meets target (>50× for medium sequences)
- ✅ Memory efficiency validated (no leaks, proper pooling)
- ✅ 20+ GPU-specific tests passing
- ✅ Cross-platform deploy tested (Linux/Windows)

### Estimated Effort

- **CUDA kernels**: 3-4 weeks
- **HIP kernels**: 2-3 weeks (reuse CUDA logic)
- **Vulkan kernels**: 2-3 weeks
- **Testing & validation**: 1-2 weeks
- **Total**: ~4 weeks (parallel development)

---

## Phase 2: HMM & MSA SIMD Optimization (TARGET: v0.6.0)

### Current State

The HMM and MSA modules have simplified implementations:

**File: `src/futures/hmm.rs`**
- Current: Placeholder probability calculations
- Stubs: `train_hmm()`, `profile_align()` use simplified DP

**File: `src/futures/msa.rs`**
- Current: Progressive alignment with scalar DP
- Stubs: PSSM construction, profile HMM creation use simple methods

### What Needs to Be Replaced

#### Part 1: HMM Module Replacement

**Current Stubs:**
```rust
pub fn train_hmm(sequences: &[Vec<AminoAcid>]) -> Result<HiddenMarkovModel> {
    // Simplified: uses naive probability counts
    // Missing: Viterbi training, EM algorithm, gap modelling
}

pub fn profile_align(
    sequence: &[AminoAcid],
    hmm: &HiddenMarkovModel,
) -> Result<AlignmentResult> {
    // Simplified: basic score calculation
    // Missing: Forward algorithm DP, proper state transitions
}
```

**Replacement Implementation**:

1. **Viterbi Training with SIMD DP**
   ```rust
   // Replace with:
   // - Forward algorithm (DP matrix computation)
   // - Backward algorithm (reverse DP pass)
   // - Viterbi decoding (max-path backtracking)
   // - SIMD parallelization of DP inner loop
   ```
   - Use AVX2: 8 parallel state transitions
   - Use NEON: 4 parallel state transitions
   - Performance: 15-20× speedup vs scalar

2. **Baum-Welch EM Algorithm**
   - Compute expected emission/transition counts
   - SIMD-accelerated probability recalculation
   - Convergence checking with vectorized operations

3. **Affine Gap Extension in HMM**
   - Model insert states properly
   - Track gap duration (open vs extend penalties)
   - Vectorize state transition scoring

**Files to Create/Modify**:
- `src/alignment/kernel/hmm_simd.rs` - SIMD HMM kernels (new)
- `src/futures/hmm.rs` - Replace simplified implementations
- `src/utils/hmm_training.rs` - Training utilities (new)

#### Part 2: MSA Module Replacement

**Current Stubs:**
```rust
pub fn construct_pssm(alignment: &MultipleAlignment) -> Result<PSSM> {
    // Simplified: just counts amino acids per position
    // Missing: Pseudocount incorporation, statistical weighting
}

pub fn create_profile_hmm(msa: &MultipleAlignment) -> Result<ProfileHMM> {
    // Simplified: converts alignment to basic probabilities
    // Missing: Proper state emission/transition estimation
}
```

**Replacement Implementation**:

1. **Position-Specific Scoring Matrix (PSSM) with Weighting**
   ```rust
   // Replace with:
   // - Sequence-weighting (Henikoff method)
   // - Pseudocount incorporation (Dirichlet priors)
   // - Information content calculation per column
   // - SIMD batch processing of alignment columns
   ```
   - Vectorize position scoring with AVX2/NEON
   - Process 512-width alignments in parallel
   - Performance: 10-15× speedup

2. **Progressive Alignment Refinement**
   - Cache intermediate DP matrices using SIMD
   - Vectorize guide-tree alignment operations
   - Profile-to-sequence alignment with vectorized DP

3. **Conservation Scoring**
   - Calculate entropy per column efficiently
   - Vectorize position scoring computation
   - Support variable gap penalties

**Files to Create/Modify**:
- `src/alignment/kernel/msa_simd.rs` - SIMD MSA kernels (new)
- `src/futures/msa.rs` - Replace simplified implementations
- `src/utils/pssm.rs` - PSSM computation utilities (new)

### Implementation Strategy

#### Algorithm 1: Forward Algorithm (DP Kernel)

```
for i in 0..sequence_len {
    for k in 0..num_states {
        // SIMD inner loop: compute 8 states in parallel (AVX2)
        score[i][k] = emission[k] * max(
            forward[i-1][k-match] + transition[k-1→k],
            forward[i-1][k-insert] + transition[insert→k],
            forward[i-1][k-delete] + transition[delete→k],
        )
    }
}
```

**SIMD Optimization**:
- Pack 8 states into `__m256i` (AVX2)
- Compute max transitions in parallel
- Reduce to find maximum per position

#### Algorithm 2: PSSM Refinement with Pseudocounts

```
for position in 0..alignment_width {
    for amino_acid in 0..20 {
        // SIMD: vectorize Dirichlet prior addition
        count = COUNT_AA_AT_POSITION(position, amino_acid)
        pseudocount = compute_dirichlet_prior(amino_acid)
        weights = weights_vec + pseudocount_vec
    }
}
```

**SIMD Targets**:
- Process 8-16 positions per iteration
- Vectorize pseudocount calculations
- Batch normalize across positions

### Testing Strategy

```rust
#[cfg(test)]
mod hmm_tests {
    #[test]
    fn test_viterbi_vs_forward_backward() {
        // Viterbi should match probabilistic path
    }
    
    #[test]
    fn test_pssm_conservation() {
        // Compare with published conservation scores
    }
    
    #[test]
    fn test_hmm_simd_accuracy() {
        // SIMD results must match scalar
    }
    
    #[test]
    fn test_msa_profile_quality() {
        // Measure improvement over simplified version
    }
}
```

### Performance Targets

| Operation | Current | Target | Speedup |
|-----------|---------|--------|---------|
| Viterbi training | Scalar | SIMD | 15-20× |
| PSSM construction | Scalar | SIMD | 10-15× |
| Profile alignment | Scalar | SIMD | 12-18× |
| MSA refinement | Scalar | SIMD | 8-12× |

### Completion Criteria

- ✅ Viterbi training produces correct consensus sequences
- ✅ Forward-backward convergence within tolerance
- ✅ PSSM conservation scores match reference implementations (HMMER)
- ✅ MSA profile quality improved measured by:
  - Benchmark suite alignment accuracy
  - Sequence homology detection rates
  - Motif discovery sensitivity
- ✅ 30+ HMM/MSA tests passing
- ✅ Speedup validated vs scalar baseline

### Estimated Effort

- **HMM Viterbi/Forward-Backward**: 2-3 weeks
- **PSSM & Pseudocount Refinement**: 1-2 weeks
- **MSA Profile Construction**: 1-2 weeks
- **SIMD Optimization**: 2-3 weeks
- **Testing & Validation**: 1-2 weeks
- **Total**: ~4-5 weeks

---

## Phase 3: Phylogenetic Inference Enhancement (TARGET: v0.7.0)

### Current State

The phylogeny module currently uses only UPGMA:

**File: `src/futures/phylogeny.rs`**
- Implemented: UPGMA algorithm
- Stubs: Maximum Parsimony, Maximum Likelihood default to UPGMA

**Current Code:**
```rust
pub fn build_tree(&mut self, method: TreeMethod) -> Result<&mut Self> {
    match method {
        TreeMethod::Upgma => upgma(distances),
        TreeMethod::MaximumParsimony => {
            // Simplified - use UPGMA for now
            upgma(distances)
        }
        TreeMethod::MaximumLikelihood => {
            // Simplified - use UPGMA for now
            upgma(distances)
        }
    }
}
```

### What Needs to Be Replaced

#### Part 1: Maximum Parsimony Heuristic Search

**Why**: Parsimony finds trees minimizing evolutionary steps (mutations)

**Current Limitation**: No parsimony search, just UPGMA fallback

**Replacement Implementation**:

1. **Fitch Algorithm Core**
   - Bottom-up DP for parsimony score calculation
   - O(n·s) complexity where n=sequences, s=sites
   - SIMD optimization for character-state scoring

2. **Heuristic Search Methods**
   - **Nearest Neighbor Interchange (NNI)**
     ```
     For each internal branch:
       1. Remove branch (split tree into 2 components)
       2. Try 2 alternative reconnections
       3. Accept if cost improves
     ```
     - Time: O(n³) for small trees
     - Implementation: Use recursive tree manipulation

   - **Subtree Pruning Regrafting (SPR)**
     ```
     For each clade:
       1. Prune subtree from tree
       2. Try reinserting at all positions
       3. Track best improvement
     ```
     - Time: O(n⁴) but finds better trees
     - Search space reduction: Early termination after no improvement

   - **Tree Bisection Reconnection (TBR)**
     ```
     More exhaustive than SPR/NNI
     For large datasets: Randomized restarts to escape local optima
     ```

3. **Bootstrap Support Calculation**
   - Sample alignment columns with replacement
   - Build parsimony tree for each bootstrap replicate
   - Count clade frequency (support percentage)
   - SIMD parallelization via Rayon batch processing

**Files to Create/Modify**:
- `src/alignment/kernel/parsimony_simd.rs` - SIMD Fitch algorithm (new)
- `src/futures/phylogeny.rs` - Replace parsimony stubs
- `src/utils/tree_search.rs` - NNI/SPR/TBR implementations (new)

**Algorithm: Accelerated Fitch Parsimony**

```rust
// SIMD vectorization of state-set operations
#[inline]
fn fitch_postorder_simd(node: &TreeNode, alignment: &[Vec<u8>]) -> (Vec<u32>, u32) {
    // Use AVX2 to score 8 character states in parallel
    // Pack state bitmasks into __m256i vectors
    // Compute downset intersections and unions in parallel
    
    // For each site (vectorized via AVX2):
    for sites in 0..alignment.len() step 8 {
        // Process 8 alignment sites in one instruction
        let state_vectors = gather_site_states(alignment, sites);
        let intersection = simd_intersect(left_subtree, right_subtree);
        cost += simd_popcount(intersection);
    }
}
```

**Performance Target**: 
- Parsimony tree search: 20-50× speedup vs scalar
- Small trees (20 sequences): <1 second
- Medium trees (100 sequences): <10 seconds
- Large trees (500+ sequences): <60 seconds

#### Part 2: Maximum Likelihood (Optional Enhancement)

**If time permits**, implement basic ML tree inference:

1. **Substitution Model Selection**
   - JC69 (Jukes-Cantor)
   - K2P (Kimura 2-parameter)
   - GTR (General Time Reversible)

2. **Likelihood Calculation**
   - Felsenstein pruning algorithm
   - Branch length estimation via Newton-Raphson
   - SIMD optimization of likelihood computation

3. **Tree Search**
   - Hill-climbing on log-likelihood
   - Use same NNI/SPR heuristics as parsimony

**Performance Target**: 
- Basic ML tree: <30 seconds for 100 sequences
- Bootstrap replicates: Parallelized via Rayon (16 cores: ~8 min)

### Implementation Strategy

#### Step 1: Implement Fitch's Algorithm (Core)

```rust
pub struct ParsimonyTree {
    tree: PhylogeneticTree,
    parsimony_scores: HashMap<NodeId, (Vec<u32>, u32)>,  // (states, total_cost)
}

impl ParsimonyTree {
    /// Calculate parsimony score for tree
    fn calculate_fitch_score(
        &mut self,
        alignment: &MultipleAlignment,
    ) -> Result<u32> {
        // Bottom-up DP: score each internal node
        // Returns minimum number of mutations
    }
    
    /// Vectorized Fitch with SIMD
    fn fitch_simd(&self, alignment: &MultipleAlignment) -> u32 {
        // Use AVX2: process 8 sites per iteration
        // Pack state bitmasks, compute in parallel
    }
}
```

#### Step 2: Implement NNI Search

```rust
pub struct TreeSearch {
    strategy: SearchStrategy,  // NNI, SPR, TBR
    max_iterations: usize,
    improvement_threshold: u32,
}

impl TreeSearch {
    fn nearest_neighbor_interchange(
        &mut self,
        tree: &mut PhylogeneticTree,
        alignment: &MultipleAlignment,
    ) -> Result<u32> {
        // Iteratively improve tree via branch swaps
        let mut best_score = calculate_parsimony(tree, alignment)?;
        let mut improved = true;
        
        while improved {
            improved = false;
            for internal_branch in tree.internal_branches() {
                let (neighbor1, neighbor2) = self.try_nni_swap(branch);
                let score1 = calculate_parsimony(&neighbor1, alignment)?;
                let score2 = calculate_parsimony(&neighbor2, alignment)?;
                
                if score1 < best_score {
                    *tree = neighbor1;
                    best_score = score1;
                    improved = true;
                }
                // ... similar for score2
            }
        }
        Ok(best_score)
    }
}
```

#### Step 3: Implement SPR Extensions

```rust
fn subtree_pruning_regrafting(
    tree: &mut PhylogeneticTree,
    alignment: &MultipleAlignment,
    max_iterations: usize,
) -> Result<u32> {
    // More exhaustive than NNI but still tractable
    // Prune each clade and try reinsertion points
    let mut best_score = calculate_parsimony(tree, alignment)?;
    
    for iteration in 0..max_iterations {
        let mut local_improvement = false;
        
        for subtree in tree.subtrees() {
            tree.remove_subtree(&subtree)?;
            
            // Try reinsertion at each branch
            for branch in tree.branches() {
                let candidate = tree.insert_subtree(&subtree, branch)?;
                let score = calculate_parsimony(&candidate, alignment)?;
                
                if score < best_score {
                    *tree = candidate;
                    best_score = score;
                    local_improvement = true;
                    break;
                }
            }
        }
        
        if !local_improvement { break; }
    }
    Ok(best_score)
}
```

### Testing Strategy

```rust
#[cfg(test)]
mod parsimony_tests {
    use super::*;
    
    #[test]
    fn test_fitch_simple() {
        // Known optimal tree should have lowest score
    }
    
    #[test]
    fn test_nni_improves_score() {
        // Random starting tree should improve
    }
    
    #[test]
    fn test_parsimony_simd_accuracy() {
        // SIMD results must match scalar
    }
    
    #[test]
    fn test_bootstrap_reproducibility() {
        // Bootstrap should be repeatable with seed
    }
    
    #[test]
    fn test_large_tree_inference() {
        // 100+ sequence trees should complete in <10s
    }
}
```

### Benchmark: Parsimony vs UPGMA

```
Benchmark: FindBest tree for 50 sequences (Benchmark dataset)
- UPGMA: O(n²) - instant
- NNI Parsimony: O(n³) - 5-10 seconds
- SPR Parsimony: O(n⁴) - 30-60 seconds
- Quality: Parsimony trees typically have 10-30% fewer mutations
```

### Performance Targets

| Operation | Current | Target | Speedup |
|-----------|---------|--------|---------|
| Fitch scoring | Scalar | SIMD | 15-20× |
| NNI search | N/A | Impl | New feature |
| SPR search | N/A | Impl | New feature |
| Bootstrap (16 cores) | N/A | Impl | Parallel |

### Completion Criteria

- ✅ Fitch algorithm produces correct parsimony scores
- ✅ NNI search finds locally optimal trees
- ✅ SPR search finds better trees than NNI
- ✅ Bootstrap support values calculated correctly
- ✅ Tree topology matches established references (e.g., Phylip output)
- ✅ 25+ parsimony/phylogeny tests passing
- ✅ Performance meets target (<10s for 100 sequences)

### Estimated Effort

- **Fitch Algorithm**: 1-2 weeks
- **NNI Search**: 1-2 weeks
- **SPR Enhancement**: 1-2 weeks
- **Bootstrap Integration**: 1 week
- **Testing & Validation**: 1-2 weeks
- **Total**: ~3-4 weeks

---

## Timeline & Milestones

```
Q2 2026:
  Week 1-4:   Phase 1 (GPU Kernels)
  Week 5-9:   Phase 2 (HMM/MSA SIMD)
  Week 10-13: Phase 3 (Phylogeny)
  
Releases:
  v0.5.0 (GPU): Real CUDA/HIP/Vulkan kernels (50-200× speedup)
  v0.6.0 (HMM): SIMD HMM & MSA (10-20× speedup)
  v0.7.0 (Phylo): Parsimony inference (new feature)
```

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| **GPU** | Speedup (100K sequences) | 50-200× |
| **GPU** | Memory efficiency | <50% GPU utilization |
| **HMM** | Speedup (Viterbi) | 15-20× |
| **HMM** | PSSM quality | ≥95% vs HMMER |
| **MSA** | Speedup (profile scoring) | 12-18× |
| **Phylo** | Parsimony search time (100 seq) | <10s |
| **Phylo** | Bootstrap (16 cores) | <10 min |
| **All** | Test coverage | >95% |

## Development Recommendations

1. **Start with Phase 1 GPU** - Proven framework, clear performance targets
2. **Parallelize GPU teams** - CUDA, HIP, Vulkan can be developed simultaneously
3. **Validate algorithms early** - Use published benchmarks (HMMER, FastTree)
4. **Profile continuously** - Use criterion.rs benchmarks after each component
5. **Community feedback** - Consider releasing alpha versions for user testing

## References

- **GPU Computing**: NVIDIA CUDA Best Practices, Vulkan Specification
- **HMM**: Durbin et al. "Biological Sequence Analysis" (Chapters 4-5)
- **MSA**: Thompson et al. "ClustalW" algorithm, HMMER documentation
- **Phylogeny**: Felsenstein "Inferring Phylogenies", PAUP manual

---

**Status**: Roadmap approved and ready for implementation sprint planning
