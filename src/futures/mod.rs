//! 🔮 Future Enhancement Modules
//!
//! This module contains scaffolding for planned features to extend OMICS-SIMD's capabilities.
//! Each submodule represents a distinct enhancement area with a clear interface for future implementation.
//!
//! # Planned Enhancements
//!
//! - **Scoring Matrices**: Data integration framework for additional matrices (PAM40/70/120, GONNET, HOXD)
//! - **Formats**: BLAST-compatible output formats (XML, JSON, tabular)
//! - **GPU Acceleration**: CUDA/HIP/Vulkan compute interface
//! - **MSA**: Multiple Sequence Alignment algorithms
//! - **HMM**: Profile Hidden Markov Models for protein families
//! - **Phylogeny**: Phylogenetic tree construction (neighbor-joining, UPGMA)
//!
//! # Architecture
//!
//! Each module is designed to:
//! - Maintain separation of concerns
//! - Integrate cleanly with existing Phase 1-3 infrastructure
//! - Support incremental implementation without breaking changes
//! - Enable feature-gated compilation for optional capabilities

pub mod formats;
pub mod gpu;
pub mod hmm;
pub mod matrices;
pub mod msa;
pub mod phylogeny;
pub mod pfam;
pub mod tree_refinement;
pub mod hmmer3_full_parser;
pub mod msa_profile_alignment;
pub mod phylogeny_parsimony;
pub mod gpu_jit_compiler;
pub mod cli_file_io;

// Import and re-export specific items to avoid glob conflicts
pub use formats::{BlastJson, BlastTabular, BlastXml, FormatError, Gff3Record};
pub use gpu::{DeviceProperties, GpuBackend, GpuDevice, GpuError, GpuMemory};
pub use hmm::{Domain, HmmError, HmmState, ProfileHmm, StateType, ViterbiPath};
pub use matrices::{MatrixError, MatrixValidation};
pub use msa::{DistanceMatrix, MsaBuilder, MsaError, MultipleSequenceAlignment, Profile};
pub use pfam::{EValueStats, PfamDatabase, PfamProfile};
pub use phylogeny::{
    PhylogeneticTree, PhylogenyError, TreeBuilder, TreeMethod as PhylogeneticTreeMethod,
    TreeNode, TreeStats,
};
pub use tree_refinement::{RefinableTree, TreeOptimizer, TreeNode as RefinedTreeNode};

// New advanced modules (v0.8.1+)
pub use hmmer3_full_parser::{Hmmer3Database, Hmmer3Model};
pub use msa_profile_alignment::{ProfileAlignment, ProfileAlignmentState};
pub use phylogeny_parsimony::{CharState, ParsimonytreeBuilder, ParsimonyStateSet};
pub use gpu_jit_compiler::{CompiledKernel, GpuJitCompiler, JitOptions, KernelTemplates};
pub use cli_file_io::{BatchProcessor, FileFormat, SeqFileReader, SeqFileWriter, SeqRecord};

