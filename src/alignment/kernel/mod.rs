//! # SIMD and GPU Kernel Module
//!
//! Architecture-specific SIMD-accelerated alignment kernels.
//! Provides compile-time selection between scalar, SIMD, and GPU implementations.
//! Includes specialized kernels for HMM training and MSA/phylogenetic inference.

pub mod scalar;
pub mod banded;
pub mod hmm_simd;
pub mod msa_simd;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "hip")]
pub mod hip;

#[cfg(feature = "vulkan")]
pub mod vulkan;
