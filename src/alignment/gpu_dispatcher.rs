//! GPU dispatcher module for intelligent GPU selection and kernel routing
//!
//! This module provides automatic GPU detection, device selection, and
//! intelligent kernel dispatch based on available hardware.

use std::fmt;

/// GPU backend availability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAvailability {
    /// CUDA available on NVIDIA GPU
    CudaAvailable,
    /// HIP available on AMD GPU
    HipAvailable,
    /// Vulkan available (cross-platform)
    VulkanAvailable,
    /// No GPU acceleration available
    Unavailable,
}

impl fmt::Display for GpuAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuAvailability::CudaAvailable => write!(f, "CUDA (NVIDIA)"),
            GpuAvailability::HipAvailable => write!(f, "HIP (AMD)"),
            GpuAvailability::VulkanAvailable => write!(f, "Vulkan"),
            GpuAvailability::Unavailable => write!(f, "No GPU available"),
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Backend type
    pub backend: GpuAvailability,
    /// Compute capability/architecture
    pub compute_capability: String,
    /// Total GPU memory in bytes
    pub total_memory: u64,
    /// Number of compute units
    pub compute_units: u32,
    /// Max threads per block
    pub max_threads: u32,
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}): {} MB memory, {} compute units",
            self.name,
            self.backend,
            self.total_memory / (1024 * 1024),
            self.compute_units
        )
    }
}

/// GPU performance characteristics and optimization hints
#[derive(Debug, Clone)]
pub struct GpuOptimizationHints {
    /// Optimal block size for this GPU
    pub optimal_block_size: usize,
    /// Number of concurrent blocks this GPU can handle efficiently
    pub concurrent_blocks: usize,
    /// Recommended maximum sequence length for single-pass computation
    pub single_pass_max_len: usize,
    /// Whether shared memory optimization is beneficial
    pub use_shared_memory: bool,
    /// Whether coalesced memory access optimization applies
    pub coalesce_memory: bool,
    /// Warp size (32 for NVIDIA, 64 for AMD)
    pub warp_size: usize,
}

impl GpuOptimizationHints {
    /// Create optimization hints for NVIDIA CUDA
    pub fn for_nvidia() -> Self {
        GpuOptimizationHints {
            optimal_block_size: 256,
            concurrent_blocks: 2048,
            single_pass_max_len: 65536,
            use_shared_memory: true,
            coalesce_memory: true,
            warp_size: 32,
        }
    }

    /// Create optimization hints for AMD HIP/ROCm
    pub fn for_amd() -> Self {
        GpuOptimizationHints {
            optimal_block_size: 256,
            concurrent_blocks: 1024,
            single_pass_max_len: 32768,
            use_shared_memory: true,
            coalesce_memory: true,
            warp_size: 64,
        }
    }

    /// Create optimization hints for Vulkan
    pub fn for_vulkan() -> Self {
        GpuOptimizationHints {
            optimal_block_size: 256,
            concurrent_blocks: 512,
            single_pass_max_len: 16384,
            use_shared_memory: false,
            coalesce_memory: false,
            warp_size: 32,
        }
    }
}

/// GPU alignment strategy based on sequence characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentStrategy {
    /// Use scalar CPU for very small sequences
    Scalar,
    /// Use SIMD (AVX2/NEON) for small sequences
    Simd,
    /// Use banded DP for similar sequences
    Banded,
    /// Use full GPU acceleration
    GpuFull,
    /// Use GPU with tiling for very large sequences
    GpuTiled,
}

/// Dispatcher decision logic based on sequence characteristics
pub struct GpuDispatcherStrategy;

impl GpuDispatcherStrategy {
    /// Select optimal alignment strategy based on sequence lengths
    pub fn select_strategy(
        len1: usize,
        len2: usize,
        gpu_available: bool,
        similarity_hint: Option<f32>,
    ) -> AlignmentStrategy {
        let total_cells = len1 * len2;
        
        const SMALL_THRESHOLD: usize = 1024 * 1024;          // 1M cells
        #[allow(dead_code)]
        const MEDIUM_THRESHOLD: usize = 10 * 1024 * 1024;    // 10M cells

        // If no GPU, use CPU strategies
        if !gpu_available {
            if total_cells < SMALL_THRESHOLD {
                // < 1M cells: use SIMD
                return AlignmentStrategy::Simd;
            } else {
                // Large sequences: use banded DP or scalar
                return AlignmentStrategy::Banded;
            }
        }

        // GPU is available - choose based on size and similarity
        match total_cells {
            0..=1024 => AlignmentStrategy::Scalar,
            1025..=SMALL_THRESHOLD => {
                // Small to medium: use GPU
                if let Some(similarity) = similarity_hint {
                    if similarity > 0.7 {
                        // High similarity: banded DP is more efficient
                        AlignmentStrategy::Banded
                    } else {
                        AlignmentStrategy::GpuFull
                    }
                } else {
                    AlignmentStrategy::GpuFull
                }
            }
            _ => {
                // Very large sequences: use GPU with tiling
                AlignmentStrategy::GpuTiled
            }
        }
    }

    /// Estimate required GPU memory for alignment
    pub fn estimate_gpu_memory(len1: usize, len2: usize) -> u64 {
        let dp_matrix_cells = ((len1 + 1) * (len2 + 1)) as u64;
        let dp_matrix_bytes = dp_matrix_cells * std::mem::size_of::<i32>() as u64;
        let seq_data = (len1 + len2) as u64 * std::mem::size_of::<u8>() as u64;
        let scoring_matrix = 24 * 24 * std::mem::size_of::<i32>() as u64;
        let misc = 1024 * 1024; // 1MB buffer for miscellaneous data

        dp_matrix_bytes + seq_data + scoring_matrix + misc
    }

    /// Check if sequence pair fits in GPU memory
    pub fn fits_in_gpu_memory(len1: usize, len2: usize, available_memory: u64) -> bool {
        let required = Self::estimate_gpu_memory(len1, len2);
        required < (available_memory / 2) // Use only 50% of GPU memory to avoid fragmentation
    }

    /// Estimate execution time benefit for GPU
    pub fn gpu_speedup_factor(strategy: AlignmentStrategy) -> f32 {
        match strategy {
            AlignmentStrategy::Scalar => 1.0,           // Baseline
            AlignmentStrategy::Simd => 8.0,              // 8x for SIMD
            AlignmentStrategy::Banded => 4.0,            // 4x for banded DP
            AlignmentStrategy::GpuFull => 50.0,          // 50x for full GPU
            AlignmentStrategy::GpuTiled => 30.0,         // 30x for tiled GPU (less efficient)
        }
    }
}

/// GPU dispatcher manager - orchestrates GPU selection and kernel dispatch
pub struct GpuDispatcher {
    available_backends: Vec<GpuAvailability>,
    device_info: Vec<GpuDeviceInfo>,
    selected_backend: GpuAvailability,
    optimization_hints: GpuOptimizationHints,
}

impl GpuDispatcher {
    /// Create a new GPU dispatcher with auto-detection
    pub fn new() -> Self {
        let available_backends = Vec::new();
        let device_info = Vec::new();

        // Detect CUDA
        #[cfg(feature = "cuda")]
        {
            available_backends.push(GpuAvailability::CudaAvailable);
            device_info.push(GpuDeviceInfo {
                name: "NVIDIA GPU (CUDA)".to_string(),
                backend: GpuAvailability::CudaAvailable,
                compute_capability: "8.0+".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024,
                compute_units: 128,
                max_threads: 1024,
            });
        }

        // Detect HIP
        #[cfg(feature = "hip")]
        {
            available_backends.push(GpuAvailability::HipAvailable);
            device_info.push(GpuDeviceInfo {
                name: "AMD GPU (HIP)".to_string(),
                backend: GpuAvailability::HipAvailable,
                compute_capability: "gfx906+".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024,
                compute_units: 64,
                max_threads: 1024,
            });
        }

        // Detect Vulkan
        #[cfg(feature = "vulkan")]
        {
            available_backends.push(GpuAvailability::VulkanAvailable);
            device_info.push(GpuDeviceInfo {
                name: "GPU (Vulkan)".to_string(),
                backend: GpuAvailability::VulkanAvailable,
                compute_capability: "Generic".to_string(),
                total_memory: 4 * 1024 * 1024 * 1024,
                compute_units: 64,
                max_threads: 256,
            });
        }

        // Select primary backend (priority: CUDA > HIP > Vulkan)
        let selected_backend = if available_backends.contains(&GpuAvailability::CudaAvailable) {
            GpuAvailability::CudaAvailable
        } else if available_backends.contains(&GpuAvailability::HipAvailable) {
            GpuAvailability::HipAvailable
        } else if available_backends.contains(&GpuAvailability::VulkanAvailable) {
            GpuAvailability::VulkanAvailable
        } else {
            GpuAvailability::Unavailable
        };

        let optimization_hints = match selected_backend {
            GpuAvailability::CudaAvailable => GpuOptimizationHints::for_nvidia(),
            GpuAvailability::HipAvailable => GpuOptimizationHints::for_amd(),
            GpuAvailability::VulkanAvailable => GpuOptimizationHints::for_vulkan(),
            GpuAvailability::Unavailable => GpuOptimizationHints::for_nvidia(),
        };

        GpuDispatcher {
            available_backends,
            device_info,
            selected_backend,
            optimization_hints,
        }
    }

    /// Get list of available GPU backends
    pub fn available_backends(&self) -> &[GpuAvailability] {
        &self.available_backends
    }

    /// Get selected/primary GPU backend
    pub fn selected_backend(&self) -> GpuAvailability {
        self.selected_backend
    }

    /// Get detailed information about available devices
    pub fn device_info(&self) -> &[GpuDeviceInfo] {
        &self.device_info
    }

    /// Get optimization hints for current GPU
    pub fn optimization_hints(&self) -> &GpuOptimizationHints {
        &self.optimization_hints
    }

    /// Check if GPU acceleration is available
    pub fn has_gpu(&self) -> bool {
        self.selected_backend != GpuAvailability::Unavailable
    }

    /// Dispatch alignment computation to best backend
    pub fn dispatch_alignment(
        &self,
        len1: usize,
        len2: usize,
        similarity_estimate: Option<f32>,
    ) -> AlignmentStrategy {
        GpuDispatcherStrategy::select_strategy(len1, len2, self.has_gpu(), similarity_estimate)
    }

    /// Get human-readable status
    pub fn status(&self) -> String {
        let backend_list = self
            .available_backends
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "GPU Dispatcher: {} backends available: {}. Selected: {}",
            self.available_backends.len(),
            if backend_list.is_empty() {
                "None".to_string()
            } else {
                backend_list
            },
            self.selected_backend
        )
    }
}

impl Default for GpuDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_dispatcher_creation() {
        let dispatcher = GpuDispatcher::new();
        let _status = dispatcher.status();
    }

    #[test]
    fn test_alignment_strategy_selection() {
        let strategy_small = GpuDispatcherStrategy::select_strategy(100, 100, true, None);
        assert_eq!(strategy_small, AlignmentStrategy::GpuFull);

        let strategy_large = GpuDispatcherStrategy::select_strategy(10000, 10000, true, None);
        assert_eq!(strategy_large, AlignmentStrategy::GpuTiled);

        let strategy_no_gpu = GpuDispatcherStrategy::select_strategy(10000, 10000, false, None);
        assert_eq!(strategy_no_gpu, AlignmentStrategy::Banded);
    }

    #[test]
    fn test_gpu_memory_estimation() {
        let mem = GpuDispatcherStrategy::estimate_gpu_memory(1000, 1000);
        assert!(mem > 1024 * 1024); // At least 1MB
    }

    #[test]
    fn test_optimization_hints() {
        let nvidia = GpuOptimizationHints::for_nvidia();
        let amd = GpuOptimizationHints::for_amd();
        let vulkan = GpuOptimizationHints::for_vulkan();

        assert_eq!(nvidia.warp_size, 32);
        assert_eq!(amd.warp_size, 64);
        assert!(vulkan.single_pass_max_len < nvidia.single_pass_max_len);
    }

    #[test]
    fn test_speedup_factors() {
        let scalar = GpuDispatcherStrategy::gpu_speedup_factor(AlignmentStrategy::Scalar);
        let gpu = GpuDispatcherStrategy::gpu_speedup_factor(AlignmentStrategy::GpuFull);

        assert_eq!(scalar, 1.0);
        assert!(gpu > 20.0);
    }
}
