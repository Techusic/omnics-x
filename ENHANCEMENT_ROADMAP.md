# OMICS-X Enhancement Roadmap

**Last Updated**: March 29, 2026  
**Status**: In Progress  
**Version**: 0.8.1 → 1.0.0 (Future)

## Overview

This document outlines comprehensive enhancements to transform OMICS-X from a functional SIMD-accelerated library to a production-grade bioinformatics platform. The roadmap is organized into 5 major phases with clear deliverables and success criteria.

---

## Phase 1: Hardware-Accelerated Kernel Dispatch ✅ Planned

**Goal**: Replace mocked GPU fallback with real NVIDIA CUDA, AMD HIP, and cross-platform support.

### 1.1 Driver Integration

**Current State**: `gpu_dispatcher.rs` provides decision logic; `cuda_device_context.rs` is stubbed.

**Implementation Tasks**:

- [ ] **CUDA Bindings** (cudarc)
  - [ ] Detect CUDA runtime availability via `cudaGetDeviceCount()`
  - [ ] Query device properties: compute capability, memory, clock speeds
  - [ ] Implement `CudaDeviceContext::new()` to initialize device and streams
  - [ ] Create `DeviceHandle` wrapper for safe GPU resource management
  - [ ] Error handling for missing drivers or incompatible devices

- [ ] **HIP Bindings** (hip-sys)
  - [ ] Cross-compile HIP kernels for AMD EPYC and Radeon GPUs
  - [ ] Implement ROCm initialization and device selection
  - [ ] Support both HIP and OpenCL backends

- [ ] **Vulkan Fallback** (ash + gpu-alloc)
  - [ ] Cross-platform compute support
  - [ ] Validation layer integration for debugging
  - [ ] Synchronization primitive wrappers

### 1.2 Buffer Synchronization (H2D/D2H Transfers)

**Current State**: No real memory transfers; simulated allocation tracking.

**Implementation Tasks**:

- [ ] **Host Memory Management**
  - [ ] Pin host memory for faster H2D transfers
  - [ ] Implement `HostBuffer<T>` for page-locked allocation
  - [ ] Support DMA transfers to GPUs with IOMMU

- [ ] **Device Memory Management**
  - [ ] Implement `DeviceBuffer<T>` with reference counting
  - [ ] Async transfer scheduling with CUDA/HIP streams
  - [ ] Memory pooling to avoid repeated allocation/deallocation

- [ ] **Transfer Layers**
  - [ ] `h2d_async(host_ptr, device_ptr, size, stream)` - async transfer
  - [ ] `d2h_sync(device_ptr, host_ptr, size)` - synchronous readback
  - [ ] Overlap computation and I/O with pipelined transfers
  - [ ] Multi-GPU all-to-all communication (AllReduce for distributed)

- [ ] **Synchronization**
  - [ ] Event-based stream synchronization
  - [ ] Fence operations for command buffer barriers
  - [ ] Deadlock-free synchronization protocols

**Success Criteria**:
- Benchmark: H2D throughput ≥ 300 GB/s (PCIe 4.0)
- Benchmark: D2H throughput ≥ 200 GB/s
- No memory leaks on repeated allocation/deallocation

### 1.3 Kernel Compilation Pipeline

**Current State**: CUDA kernel code stored as strings in `gpu_kernels.rs`; no compilation/loading.

**Implementation Tasks**:

- [ ] **CUDA JIT Compilation**
  - [ ] Integrate NVRTC (NVIDIA Runtime Compilation)
  - [ ] Compile PTX at runtime for better device compatibility
  - [ ] Cache compiled fatbins to disk for faster startup
  - [ ] Support `#include` directives for modular kernels

- [ ] **AMD GPU Compilation**
  - [ ] HIP runtime compilation or offline compilation
  - [ ] hiprtc (HIP Runtime Compilation) integration
  - [ ] hipcc wrapper for compilation flags

- [ ] **Kernel Loading & Caching**
  - [ ] Load pre-compiled `.ptx` files (offline compiled)
  - [ ] Cache compiled LLVM IR between runs
  - [ ] Version check to invalidate stale caches

- [ ] **Kernel Module Organization**
  - [ ] Separate kernels: `smith_waterman.ptx`, `needleman_wunsch.ptx`, `profile_hmm.ptx`
  - [ ] Modular C++ kernel code with preprocessor configuration
  - [ ] Register allocation tuning flags

**Success Criteria**:
- First-run compilation < 2 seconds
- Cached loads < 100ms
- Kernels achieve ≥ 80% of theoretical GPU peak performance

---

## Phase 2: Mathematically Rigorous HMM Training ✅ Planned

**Goal**: Implement real parameter estimation (Baum-Welch) and PFAM compatibility.

**Current State**: Basic HMM exists in `src/futures/hmm.rs`; training is simplified.

### 2.1 Baum-Welch (EM Algorithm)

**Implementation Tasks**:

- [ ] **E-step**
  - [ ] Enhanced forward pass with initialization for all start positions
  - [ ] Backward pass computation with care for underflow
  - [ ] Posteriors: `P(state_i|sequence)` via forward/backward combination
  - [ ] Expected counts for transitions and emissions

- [ ] **M-step**
  - [ ] Re-estimate transition probabilities from expected counts
  - [ ] Re-estimate emission probabilities (with pseudocount smoothing)
  - [ ] Apply Laplace or Dirichlet-multinomial smoothing

- [ ] **Convergence Criteria**
  - [ ] Likelihood-based convergence (delta ≤ 1e-6)
  - [ ] Max iterations cap (50-100)
  - [ ] Early stopping on palindromic updates

- [ ] **Probabilistic Foundation**
  - [ ] Log-domain computations to prevent underflow
  - [ ] Numerically stable forward/backward scaling
  - [ ] Posterior probability validation (sums to 1)

**Code Location**: `src/futures/hmm.rs::ProfileHmm::train_baum_welch()`

### 2.2 PFAM Model Parser

**Implementation Tasks**:

- [ ] **HMMER3 Format Support**
  - [ ] Parse `.hmm` file headers (NAME, ACC, LENG, etc.)
  - [ ] Read transition and emission tables
  - [ ] Extract null model for E-value calibration
  - [ ] Handle composition and special insertion columns

- [ ] **Model Validation**
  - [ ] Check topologies (valid HMM structure)
  - [ ] Verify probabilities sum to 1 (with tolerance)
  - [ ] Load cutoff thresholds (GA, TC, NC for gathering/trusted/noise)

- [ ] **Database Integration**
  - [ ] Support PfamA/PfamB variants
  - [ ] Index PFAM database for fast lookups
  - [ ] Cache parser results in binary format (`.pfamcache`)

- [ ] **Streaming Parser**
  - [ ] Lazy load models without loading entire database
  - [ ] `PfamDatabase::open(path) -> Iterator<ProfileHmm>`
  - [ ] Memory-efficient handling of 20k+ PFAM entries

**Code Location**: `src/futures/hmm.rs::PfamDatabase` (new)

**Success Criteria**:
- Parse PFAM-A (97% of all profiles) with <10ms per model
- Validate against official PFAM test set
- E-values match HMMER3 within 0.1 bits

### 2.3 Advanced Scoring Features

**Implementation Tasks**:

- [ ] **Null Model E-value Calculation**
  - [ ] Implement Karlin-Altschul statistics
  - [ ] Calibrate lambda, K parameters from PFAM
  - [ ] Report E-values in addition to raw scores

- [ ] **Bit Score Conversion**
  - [ ] `bit_score = (score - log(K)) / log(2)`
  - [ ] Comparable across different training sets

---

## Phase 3: Advanced MSA & Phylogeny Heuristics ✅ Planned

**Goal**: Replace simplified algorithms with mathematically rigorous implementations.

**Current State**: MSA has basic guide tree; phylogeny has UPGMA stub.

### 3.1 Profile-to-Profile DP

**Current Implementation**: Naive "highest-scoring amino acid" matching.

**Implementation Tasks**:

- [ ] **Position-Specific Scoring Matrix (PSSM)**
  - [ ] Maintain probability distributions at each MSA column
  - [ ] Henikoff weighting to downweight duplicate-like sequences
  - [ ] Dirichlet mixture priors for pseudocount estimation
  - [ ] Shannon entropy measurement for conservation

- [ ] **Profile Alignment DP**
  - [ ] Extend DP recurrence: `max(M[i-1,j-1], I[i-1,j], D[i,j-1])` where:
    - M: profile-to-profile match cell
    - I: profile-to-sequence gap (semantic unchanged)
    - D: sequence-to-profile gap
  - [ ] Use weighted amino acid scores from profile PSSM
  - [ ] Affine gap penalty in both profiles

- [ ] **Iterative Refinement**
  - [ ] Implement MUSCLE-style iterative refinement
  - [ ] Profile consistency scoring
  - [ ] Convergence on profile stability (Frobenius norm)

**Code Location**: `src/futures/msa.rs::MsaBuilder::profile_align()`

### 3.2 Heuristic Search for Maximum Parsimony/ML

**Current Implementation**: MP/ML fall back to UPGMA.

**Implementation Tasks**:

- [ ] **Nearest Neighbor Interchange (NNI)**
  - [ ] Local perturbation of tree topology
  - [ ] Swap subtrees around central edge
  - [ ] Score with parsimony score or likelihood
  - [ ] Greedy local search with random restarts

- [ ] **Subtree Pruning and Regrafting (SPR)**
  - [ ] More aggressive heuristic than NNI
  - [ ] Detach subtree and reattach elsewhere
  - [ ] Broader neighborhood exploration

- [ ] **Integration with MP/ML**
  - [ ] Initialize with NJ tree
  - [ ] Refine via NNI/SPR hill climbing
  - [ ] Report final tree and optimization history

**Code Location**: `src/futures/phylogeny.rs::TreeBuilder::refine_nni()`

### 3.3 Ancestral Sequence Reconstruction

**Current Implementation**: Returns `"INFERRED"` placeholder string.

**Implementation Tasks**:

- [ ] **Sankoff's Algorithm**
  - [ ] Dynamic programming on tree for joint state reconstruction
  - [ ] Compute optimal amino acids at each internal node
  - [ ] Parsimony cost minimization at each node

- [ ] **Fitch's Algorithm** (simpler version for binary parsimony)
  - [ ] Upward and downward passes on tree
  - [ ] Reconstruct most parsimonious sequences

- [ ] **Ambiguity Handling**
  - [ ] Report multiple equally likely states prob with probabilities
  - [ ] Confidence scores based on tree topology

**Code Location**: `src/futures/phylogeny.rs::PhylogeneticTree::reconstruct_ancestors()`

**Success Criteria**:
- MSA scoring matches T-Coffee/MUSCLE within 5%
- MP tree score matches TNT within 1%
- Ancestor reconstruction validates against simulated evolution

---

## Phase 4: SIMD Extensions for Secondary Modules ✅ Planned

**Goal**: Vectorize HMM and MSA bottlenecks (currently scalar).

**Current State**: Core alignment has AVX2/NEON; HMM/MSA are pure scalar.

### 4.1 Vectorized Viterbi Algorithm

**Challenge**: Viterbi has strict state dependency (i → i+1).

**Implementation Tasks**:

- [ ] **Batch Viterbi**
  - [ ] Process multiple sequences against same HMM in parallel
  - [ ] SIMD width: 8 sequences (AVX2-i32) or 4 (NEON)
  - [ ] Shared HMM, separate DP tables per sequence

- [ ] **Horizontal Operations**
  - [ ] Max reduction across SIMD lanes
  - [ ] Conditional state tracking (bitpacked)

- [ ] **Kernel Implementation**
  - [ ] `src/alignment/kernel/hmm_viterbi_avx2.rs`
  - [ ] `src/alignment/kernel/hmm_viterbi_neon.rs`
  - [ ] 4-8x speedup expected over scalar

**Code Location**: `src/alignment/kernel/hmm_viterbi.rs` (new)

### 4.2 MSA Profile Vectorization

**Current Bottleneck**: Position-by-position amino acid counting (O(n²m)).

**Implementation Tasks**:

- [ ] **Vectorized Histogram**
  - [ ] SIMD histogram for 24 amino acids
  - [ ] Lane-wise accumulation then horizontal add
  - [ ] 8x faster than scalar loop

- [ ] **Profile Score Lookup**
  - [ ] Vectorized PSSM table lookups
  - [ ] Gather loads for non-contiguous accesses
  - [ ] Prefetch PSSM data

- [ ] **Kernel Implementation**
  - [ ] `src/alignment/kernel/msa_profile_avx2.rs`
  - [ ] `src/alignment/kernel/msa_profile_neon.rs`

**Code Location**: `src/alignment/kernel/msa_profile.rs` (new)

### 4.3 Forward/Backward Algorithm Vectorization

**Implementation Tasks**:

- [ ] **Batch Forward Pass**
  - [ ] Multiple sequences, same HMM
  - [ ] Vectorized log-space addition

- [ ] **Numerically Stable Log-Sum-Exp**
  - [ ] SIMD implementation of `log(exp(a) + exp(b))`
  - [ ] Underflow-safe floating point

---

## Phase 5: Production Tooling & CLI ✅ Planned

**Goal**: Complete standalone tool with benchmarking and CLI interface.

### 5.1 Real-Time Benchmarking Suite

**Current State**: Criterion benchmarks in `benches/alignment_benchmarks.rs`; GPU not included.

**Implementation Tasks**:

- [ ] **Throughput Metrics**
  - [ ] Genomic Cell Updates Per Second (GCUPS)
  - [ ] Formula: `(len1 * len2 * num_queries) / (time_seconds * 1e9)`
  - [ ] Report: CPU GCUPS vs GPU GCUPS vs speedup factor

- [ ] **GPU vs SIMD Comparison**
  - [ ] Benchmark latency (single alignment)
  - [ ] Benchmark throughput (batch alignments)
  - [ ] Memory bandwidth utilization
  - [ ] PCIe transfer overhead characterization

- [ ] **Criterion Integration**
  - [ ] `benches/gpu_throughput.rs`
  - [ ] `benches/simd_vs_gpu.rs`
  - [ ] `benches/memory_bandwidth.rs`
  - [ ] Generate comparative HTML reports

- [ ] **Profile Real Datasets**
  - [ ] SwissProt database (500k proteins)
  - [ ] NCBI nr database (100M sequences)
  - [ ] Genome alignments (30M bp × 30M bp)

**Code Location**: `benches/gpu_throughput.rs` (new)

### 5.2 CLI Interface

**Current State**: No command-line tool; library-only usage.

**Implementation Tasks**:

- [ ] **Argument Parser** (clap)
  - [ ] Subcommands: `align`, `msa`, `hmm-search`, `phylogeny`
  - [ ] Input formats: FASTA, GenBank, BAM
  - [ ] Output formats: CIGAR, SAM, BAM, JSON, XML

- [ ] **align Subcommand**
  - [ ] `omics-x align -q queries.fasta -s subject.fasta -o results.bam`
  - [ ] Algorithm selection: `--algorithm [sw|nw|banded]`
  - [ ] GPU/CPU selection: `--device [auto|cpu|gpu|gpu:0]`
  - [ ] Output format: `--output-format [bam|sam|json]`

- [ ] **msa Subcommand**
  - [ ] `omics-x msa -i seqs.fasta -o alignment.fasta`
  - [ ] Guide tree: `--guide-tree [upgma|nj]`
  - [ ] Iteration: `--iterations 2`

- [ ] **hmm-search Subcommand**
  - [ ] `omics-x hmm-search --hmm pfam.hmm query.fasta`
  - [ ] Domain detection with E-value thresholds
  - [ ] Batch processing

- [ ] **phylogeny Subcommand**
  - [ ] `omics-x phylogeny -i msa.fasta --method [upgma|nj|mp|ml]`
  - [ ] Bootstrap: `--bootstrap 100`
  - [ ] Output: Newick, JSON with confidence scores

- [ ] **Performance Profiling**
  - [ ] `--benchmark` flag for throughput reporting
  - [ ] `--profile` for CPU/GPU profiling integration

**Code Location**: `src/bin/omics-x.rs` (new)

### 5.3 BAM File Processing Pipeline

**Current State**: BAM serialization exists; no streaming pipeline.

**Implementation Tasks**:

- [ ] **Streaming BAM Reader**
  - [ ] Memory-efficient line-by-line processing
  - [ ] Index support (`.bai`) for random access
  - [ ] Compressed vs uncompressed I/O paths

- [ ] **Batch Alignment from BAM**
  - [ ] `load_batch_from_bam(reader, batch_size) -> AlignmentBatch`
  - [ ] Automatic batching for GPU efficiency
  - [ ] Zero-copy views where possible

- [ ] **BAM Output Writer**
  - [ ] Streaming SAM header and record output
  - [ ] BGZF compression
  - [ ] SAM/BAM format validation

**Code Location**: `src/alignment/bam.rs` (extend existing)

---

## Implementation Schedule

### Sprint 1 (Week 1-2): GPU Kernel Dispatch
- CUDA driver integration (cudarc)
- HIP bindings (optional, can be Phase 2)
- Device memory management and H2D/D2H transfers
- CUDA kernel compilation via NVRTC

**Deliverable**: GPU-accelerated Smith-Waterman working with 10x speedup over CPU

### Sprint 2 (Week 3-4): HMM Training & PFAM
- Baum-Welch EM implementation
- PFAM parser and model loading
- E-value calculation (Karlin-Altschul)

**Deliverable**: Load Pfam-A, score sequences with real E-values

### Sprint 3 (Week 5-6): MSA & Phylogeny
- Profile-to-profile DP alignment
- NNI/SPR tree refinement heuristics
- Ancestral sequence reconstruction (Sankoff)

**Deliverable**: Multiple sequence alignment with iterative refinement; MP/ML trees

### Sprint 4 (Week 7): SIMD Extensions
- Vectorized Viterbi (batch processing)
- Vectorized MSA profile scoring

**Deliverable**: 4-8x speedup for HMM and MSA on SIMD hardware

### Sprint 5 (Week 8): CLI & Benchmarking
- Full CLI tool with subcommands
- Comprehensive GPU vs SIMD benchmarks
- BAM pipeline integration

**Deliverable**: Standalone `omics-x` binary ready for production use

---

## Success Criteria

### Per-Phase Validation

1. **GPU Dispatch**: Benchmark ≥ 10x speedup over SIMD for alignments; PCIe overhead < 5%
2. **HMM Training**: E-values match HMMER3 within 0.1 bits; Baum-Welch converges in < 50 iterations
3. **MSA/Phylogeny**: Scoring within 5% of T-Coffee; MP trees match TNT within 1%
4. **SIMD Extensions**: 4-8x speedup for Viterbi and profile scoring
5. **CLI/Benchmarking**: Tool processes 1M alignments/sec on GPU; zero runtime errors on 1k query scripts

### Overall Metrics

- **Code Quality**: Zero compiler warnings; 95%+ test coverage
- **Documentation**: Complete API docs, 5+ end-to-end examples
- **Performance**: 100-200 GCUPS on GPU across diverse workloads
- **Compatibility**: Works on Linux, macOS, Windows; both x86-64 and aarch64

---

## Known Blockers & Mitigations

| Blocker | Impact | Mitigation |
|---------|--------|-----------|
| CUDA availability in CI/CD | Difficult GPU testing | Docker with NVIDIA CUDA base image |
| PFAM file format changes | Parser brittleness | Version-gated parser with fallback |
| GPU memory limits (<12GB) | Large MSA processing | Tiled DP algorithm with streaming I/O |
| Numerical instability in log-domain | ML likelihood underflow | Numerically-aware library (e.g., `rug`) |

---

## Future Work (Post-1.0)

- [ ] Multi-GPU support (data parallelism across GPUs)
- [ ] Sparse DP for highly divergent sequences
- [ ] GPU-accelerated variant calling
- [ ] Integration with machine learning frameworks (PyTorch/TensorFlow)
- [ ] Cloud deployment (AWS Batch, Google Life Sciences)

---

## References

- **HMM Training**: Rabiner (1989) "A tutorial on hidden Markov models"
- **MSA**: Edgar (2004) "MUSCLE: Multiple sequence alignment with high accuracy and high throughput"
- **Phylogeny**: Felsenstein (2004) "Inferring Phylogenies"
- **GPU Algorithms**: Harris et al. (2007) "Scalable data structures for GPU computing"
- **PFAM**: Mistry et al. (2021) "Pfam: The protein families database in 2021"

---

**Current Progress**: Phase 1 in progress  
**Target Completion**: Q2 2026  
**Contributors**: @techusic, community
