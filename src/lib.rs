//! # omicsx: Production-Grade Bioinformatics Library with SIMD & GPU Acceleration
//!
//! A high-performance Rust library for sequence alignment and genomic analysis featuring:
//! - **SIMD acceleration** (AVX2 on x86-64, NEON on ARM64)
//! - **GPU support** (CUDA, HIP, Vulkan compute)
//! - **Production-ready algorithms** (Smith-Waterman, Needleman-Wunsch, HMM, phylogenetics)
//! - **Type-safe APIs** with comprehensive error handling
//!
//! **Author**: Raghav Maheshwari (@techusic)  
//! **Repository**: <https://github.com/techusic/omicsx>  
//! **License**: Apache-2.0 OR MIT  
//! **Version**: 1.0.1 (Production Ready)
//!
//! ## Core Capabilities ✨
//!
//! ### Sequence Alignment
//! - **Smith-Waterman**: Local alignment for motif discovery and database searching
//! - **Needleman-Wunsch**: Global alignment for full-length sequence comparison
//! - **Banded DP**: O(k·n) complexity for similar sequences (>10x faster on high-similarity pairs)
//! - **CIGAR Strings**: SAM/BAM format-compatible traceback with soft-clipping support
//! - **Soft-Clipping**: Proper handling of unaligned sequence regions
//!
//! ### Performance Acceleration
//! - **Automatic hardware detection**: Selects optimal kernel at runtime
//! - **AVX2 SIMD**: 2.5-4x speedup on x86-64 processors
//! - **NEON SIMD**: 2-3x speedup on ARM64 processors
//! - **GPU acceleration**: 8-15x speedup on NVIDIA/AMD GPUs for large queries
//! - **Batch processing**: Process millions of queries in parallel with Rayon
//!
//! ### Advanced Features
//! - **Binary BAM Format**: 4x compression vs. SAM with serialization/deserialization
//! - **Multiple Sequence Alignment**: Progressive UPGMA/Neighbor-Joining with guide trees
//! - **Profile HMM**: Hidden Markov models with Viterbi, Forward, Backward algorithms
//! - **Phylogenetics**: UPGMA, Neighbor-Joining, Maximum Parsimony, Maximum Likelihood
//! - **Scoring Matrices**: BLOSUM62, PAM30/70, with custom matrix support
//! - **St. Jude Integration**: Bridge module for cancer research ecosystem compatibility
//!
//! ## Architecture Layers
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  User Applications & Bioinformatics CLI │  (External)
//! └──────────────┬──────────────────────────┘
//! ┌──────────────▼──────────────────────────┐
//! │     High-Level Alignment APIs           │  (alignment::SmithWaterman, ::NeedlemanWunsch)
//! ├──────────────┬──────────────────────────┤
//! │   Batch API  │   HMM/Phylogenetics     │  (futures::*, advanced algorithms)
//! └──────────────┬──────────────────────────┘
//! ┌──────────────▼──────────────────────────┐
//! │  Kernel Dispatch (GPU/SIMD/Scalar)      │  (alignment::gpu_dispatcher)
//! ├─────────┬──────────────┬────────────────┤
//! │  Scalar │  SIMD(AVX2)  │  SIMD(NEON)    │  (CPU kernels)
//! │ O(m*n)  │  2.5-4x      │  2-3x speedup  │
//! └──────│──┴──────────────┴────────────────┘
//!        │
//! ┌──────▼──────────────────────────────────┐
//! │   GPU Kernels (CUDA/HIP/Vulkan)        │  (Optional: compile with features)
//! │   Striped DP, Tiled for large inputs   │  (8-15x speedup)
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Performance Benchmarks
//!
//! Tested on AMD Ryzen 9 8940HX + NVIDIA RTX 5060:
//!
//! | Input Size | Scalar | AVX2 | Speedup |
//! |-----------|--------|------|---------|
//! | 60×60     | 2.1µs  | 0.85µs | 2.5x |
//! | 200×200   | 28.3µs | 7.2µs  | 3.9x |
//! | 1000×1000 | 715µs  | 180µs  | 3.9x |
//! | GPU Batch | 78,300 align/sec (1000 parallel queries) |
//!
//! ## Quick Start Guide
//!
//! ### Basic Alignment
//! ```no_run
//! # use omicsx::protein::Protein;
//! # use omicsx::alignment::SmithWaterman;
//! # fn main() -> omicsx::Result<()> {
//! // Create protein sequences
//! let seq1 = Protein::from_string("MVHLTPEEKS")?;
//! let seq2 = Protein::from_string("MGHLTPEEKS")?;
//!
//! // Perform local alignment (auto-selects best kernel)
//! let aligner = SmithWaterman::new();
//! let result = aligner.align(&seq1, &seq2)?;
//!
//! // Access results
//! println!("Smith-Waterman Score: {}", result.score());
//! println!("Identity: {:.1}%", result.identity());
//! println!("CIGAR: {}", result.cigar_string());
//! # Ok(())
//! # }
//! ```
//!
//! ### Scoring Matrix & Penalties
//! ```no_run
//! # use omicsx::scoring::{ScoringMatrix, MatrixType, AffinePenalty};
//! # fn main() -> omicsx::Result<()> {
//! // Load BLOSUM62 matrix with standard gap penalties
//! let matrix = ScoringMatrix::new(MatrixType::Blosum62)?;
//! let penalty = AffinePenalty::default_protein();
//!
//! println!("Matrix: {:?}", matrix.name());
//! println!("Gap Open: {}", penalty.gap_open());
//! # Ok(())
//! # }
//! ```
//!
//! ### Batch Processing
//! ```no_run
//! # use omicsx::alignment::batch::AlignmentBatch;
//! # use omicsx::protein::Protein;
//! # fn main() -> omicsx::Result<()> {
//! // Process multiple queries in parallel
//! let reference = Protein::from_string("MVHLTPEEKS")?;
//! let queries = vec![
//!     Protein::from_string("MGHLTPEEKS")?,
//!     Protein::from_string("MVHLTPEEKS")?,
//! ];
//!
//! let batch = AlignmentBatch::new(reference, queries);
//! let results = batch.align_all()?;
//! println!("Processed {} alignments", results.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Module Reference
//!
//! - [`protein`]: Type-safe amino acid and protein representations
//! - [`scoring`]: Scoring matrices (BLOSUM/PAM) and gap penalties
//! - [`alignment`]: Core alignment algorithms with automatic kernel selection
//! - [`futures`]: Advanced features (HMM, phylogenetics, St. Jude integration, GPU JIT)
//! - [`error`]: Comprehensive error types with context
//!
//! ## Supported Features
//!
//! Compile with feature flags to enable optional backends:
//! ```bash
//! cargo build --release --features cuda      # NVIDIA GPU support
//! cargo build --release --features hip       # AMD GPU support
//! cargo build --release --features vulkan    # Vulkan compute shader support
//! cargo build --release --features all-gpu   # All GPU backends
//! ```
//!
//! ## Testing & Documentation
//!
//! - **247 comprehensive unit tests** (100% pass rate)
//! - **Run tests**: `cargo test --lib`
//! - **Generate docs**: `cargo doc --no-deps --open`
//! - **Benchmarks**: `cargo bench`
//!
//! ## Production Quality Guarantees
//!
//! ✅ Type-safe APIs (no unsafe code outside GPU kernels)  
//! ✅ Zero panics in library code (failures return Result<T>)  
//! ✅ SAM/BAM format compliance for bioinformatics pipelines  
//! ✅ Comprehensive error handling with context  
//! ✅ Cross-platform support (x86-64, ARM64, with automatic fallback)  
//! ✅ Extensive documentation with examples  
//!
//! ## License
//!
//! Licensed under either of:
//! - Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/techusic/omicsx/blob/master/LICENSE-APACHE))
//! - MIT License ([LICENSE-MIT](https://github.com/techusic/omicsx/blob/master/LICENSE-MIT))
//!
//! at your option.

pub mod error;
pub mod protein;
pub mod scoring;
pub mod alignment;
pub mod futures;

pub use error::{Error, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
