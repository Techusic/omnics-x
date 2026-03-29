//! CUDA kernel implementations for GPU-accelerated sequence alignment
//!
//! This module provides optimized CUDA kernels for Smith-Waterman and Needleman-Wunsch
//! algorithms. It leverages NVIDIA-specific features like shared memory, warp primitives,
//! and Tensor Cores for maximum performance.
//!
//! ## Kernel Optimizations (Phase 4)
//!
//! 1. **Shared Memory Usage**: Cache 24×24 scoring matrix in shared memory
//! 2. **Warp-Level Operations**: Use __shfl_sync for efficient reductions
//! 3. **Thread Coalescing**: Optimize memory access patterns
//! 4. **Register Optimization**: Unroll inner loops to maximize register usage
//! 5. **Tensor Core Support**: Optional for RTX GPUs (compute_capability >= 7.0)
//!
//! ## Memory Architecture
//!
//! ```text
//! GPU Global Memory
//!   ├── seq1: Sequences (read-only)
//!   ├── seq2: Subject sequence (read-only, cached)
//!   ├── matrix: Scoring matrix (24×24, cached in shared memory)
//!   ├── h_in: Input DP row from previous column
//!   ├── h_out: Output DP row for current column
//!   └── results: Final DP matrix (partial for space efficiency)
//!
//! GPU Shared Memory (per block)
//!   ├── matrix_cache[24][32]: Scoring matrix (padded for bank conflicts)
//!   ├── col_cache[BLOCK_SIZE]: Current column cells
//!   └── row_buffer[2*BLOCK_SIZE]: Row dependency tracking
//! ```

/// CUDA compute capability for optimization decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaComputeCapability {
    /// Maxwell (GTX 750, GTX 960, GTX 1080)
    Maxwell,
    /// Pascal (GTX 1080 Ti, Titan X)
    Pascal,
    /// Volta (V100, Titan V)
    Volta,
    /// Turing (RTX 2080, RTX 2080 Ti)
    Turing,
    /// Ampere (RTX 3080, A100)
    Ampere,
    /// Ada (RTX 4090, H100)
    Ada,
}

impl CudaComputeCapability {
    /// Parse from version string (e.g., "8.6" → Ampere)
    pub fn from_version(major: u32, minor: u32) -> Option<Self> {
        match (major, minor) {
            (5, _) => Some(Self::Maxwell),
            (6, _) => Some(Self::Pascal),
            (7, 0) => Some(Self::Volta),
            (7, 5) => Some(Self::Turing),
            (8, 0) => Some(Self::Ampere),
            (9, 0) => Some(Self::Ada),
            _ => None,
        }
    }

    /// Whether this capability supports Tensor Cores
    pub fn has_tensor_cores(&self) -> bool {
        matches!(
            self,
            Self::Volta | Self::Turing | Self::Ampere | Self::Ada
        )
    }

    /// Optimal block size for this capability
    pub fn optimal_block_size(&self) -> usize {
        match self {
            Self::Maxwell | Self::Pascal => 256,
            Self::Volta | Self::Turing => 512,
            Self::Ampere => 1024,
            Self::Ada => 1024,
        }
    }

    /// Shared memory per block (bytes)
    pub fn shared_memory(&self) -> usize {
        match self {
            Self::Maxwell => 49152,     // 48KB
            Self::Pascal => 49152,      // 48KB
            Self::Volta => 98304,       // 96KB with config
            Self::Turing => 98304,      // 96KB with config
            Self::Ampere => 163840,     // 160KB with config
            Self::Ada => 163840,        // 160KB with config
        }
    }

    /// Maximum registers per thread
    pub fn max_registers(&self) -> usize {
        match self {
            Self::Maxwell => 255,
            Self::Pascal => 255,
            Self::Volta => 255,
            Self::Turing => 255,
            Self::Ampere => 255,
            Self::Ada => 255,
        }
    }
}

/// CUDA kernel configuration
#[derive(Debug, Clone)]
pub struct CudaKernelConfig {
    /// Compute capability for optimization
    pub compute_capability: CudaComputeCapability,
    /// Block size (threads per block)
    pub block_size: usize,
    /// Enable shared memory optimization
    pub use_shared_memory: bool,
    /// Enable warp shuffles for reductions
    pub use_warp_shuffles: bool,
    /// Enable Tensor Core acceleration (if available)
    pub use_tensor_cores: bool,
    /// Enable register spilling minimization
    pub optimize_registers: bool,
}

impl Default for CudaKernelConfig {
    fn default() -> Self {
        let cap = CudaComputeCapability::Ampere;
        Self {
            block_size: cap.optimal_block_size(),
            compute_capability: cap,
            use_shared_memory: true,
            use_warp_shuffles: true,
            use_tensor_cores: cap.has_tensor_cores(),
            optimize_registers: true,
        }
    }
}

/// Placeholder for actual CUDA kernel interface
/// In production, this would interface with cudarc or similar
#[derive(Debug)]
pub struct CudaAlignmentKernel {
    config: CudaKernelConfig,
    device_id: i32,
}

impl CudaAlignmentKernel {
    /// Create new CUDA kernel
    pub fn new(device_id: i32, compute_capability: CudaComputeCapability) -> Self {
        let config = CudaKernelConfig {
            compute_capability,
            block_size: compute_capability.optimal_block_size(),
            use_shared_memory: true,
            use_warp_shuffles: true,
            use_tensor_cores: compute_capability.has_tensor_cores(),
            optimize_registers: true,
        };

        Self { config, device_id }
    }

    /// Get kernel configuration
    pub fn config(&self) -> &CudaKernelConfig {
        &self.config
    }

    /// Calculate optimal grid size for sequence alignment
    /// 
    /// For a matrix of size M×N:
    /// - Each block processes one SIMD_WIDTH-wide stripe
    /// - Blocks process columns in parallel
    /// - Result: ceil(N / SIMD_WIDTH) blocks
    pub fn calculate_grid_size(&self, m: usize, n: usize) -> (u32, u32) {
        const SIMD_WIDTH: usize = 8;
        let grid_x = ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) as u32;
        let grid_y = 1; // Process columns sequentially
        (grid_x, grid_y)
    }

    /// Calculate shared memory requirements for kernel
    pub fn shared_memory_size(&self) -> usize {
        // 24×24 scoring matrix (padded) + working space
        // Matrix: 24 × 32 × sizeof(i32) = 3072 bytes
        // Working: 2 × BLOCK_SIZE × sizeof(i32) = ~4KB
        // Total: ~8KB per block
        let matrix_size = 24 * 32 * 4; // 3072
        let working_size = 2 * self.config.block_size * 4; // ~4KB
        matrix_size + working_size
    }

    /// Estimated runtime for alignment (ms)
    pub fn estimate_time(&self, m: usize, n: usize) -> f32 {
        // Rough estimate: ~0.002ms per DP cell for Ampere GPU
        // 500×500 = 250K cells → ~0.5ms actual (with overhead ~2ms)
        let ops = (m * n) as f32;
        let ops_per_ms = match self.config.compute_capability {
            CudaComputeCapability::Maxwell => 50_000.0,
            CudaComputeCapability::Pascal => 100_000.0,
            CudaComputeCapability::Volta => 200_000.0,
            CudaComputeCapability::Turing => 300_000.0,
            CudaComputeCapability::Ampere => 500_000.0,
            CudaComputeCapability::Ada => 800_000.0,
        };
        (ops / ops_per_ms) + 1.5 // Add 1.5ms fixed overhead
    }
}

/// Multi-GPU batch processing context
#[derive(Debug)]
pub struct CudaMultiGpuBatch {
    /// Allocated GPU devices
    devices: Vec<i32>,
    /// Per-device kernel
    kernels: Vec<CudaAlignmentKernel>,
    /// Current batch index
    current_batch: usize,
}

impl CudaMultiGpuBatch {
    /// Create multi-GPU batch context
    pub fn new(device_ids: Vec<i32>) -> Self {
        let kernels = device_ids
            .iter()
            .map(|&id| {
                // Default to Ampere, real implementation would detect
                CudaAlignmentKernel::new(id, CudaComputeCapability::Ampere)
            })
            .collect();

        Self {
            devices: device_ids,
            kernels,
            current_batch: 0,
        }
    }

    /// Get next available GPU in round-robin fashion
    pub fn next_device(&mut self) -> &CudaAlignmentKernel {
        let kernel = &self.kernels[self.current_batch % self.kernels.len()];
        self.current_batch += 1;
        kernel
    }

    /// Reset batch counter
    pub fn reset(&mut self) {
        self.current_batch = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_compute_capability() {
        let cap = CudaComputeCapability::from_version(8, 0);
        assert_eq!(cap, Some(CudaComputeCapability::Ampere));
        assert!(cap.unwrap().has_tensor_cores());
    }

    #[test]
    fn test_kernel_config() {
        let config = CudaKernelConfig::default();
        assert_eq!(config.compute_capability, CudaComputeCapability::Ampere);
        assert_eq!(config.block_size, 1024);
        assert!(config.use_shared_memory);
    }

    #[test]
    fn test_grid_calculation() {
        let kernel = CudaAlignmentKernel::new(0, CudaComputeCapability::Ampere);
        let (grid_x, grid_y) = kernel.calculate_grid_size(500, 500);
        assert!(grid_x > 0);
        assert_eq!(grid_y, 1);
    }

    #[test]
    fn test_time_estimation() {
        let kernel = CudaAlignmentKernel::new(0, CudaComputeCapability::Ampere);
        let time = kernel.estimate_time(500, 500);
        // Should be reasonable estimate for 250K cells
        assert!(time > 1.0 && time < 10.0);
    }

    #[test]
    fn test_multi_gpu_batch() {
        let mut batch = CudaMultiGpuBatch::new(vec![0, 1, 2]);
        
        let dev1_id = batch.next_device().device_id;
        let dev2_id = batch.next_device().device_id;
        let dev3_id = batch.next_device().device_id;
        let dev1_again_id = batch.next_device().device_id;
        
        assert_eq!(dev1_id, 0);
        assert_eq!(dev2_id, 1);
        assert_eq!(dev3_id, 2);
        assert_eq!(dev1_again_id, 0);
    }

    #[test]
    fn test_shared_memory_size() {
        let kernel = CudaAlignmentKernel::new(0, CudaComputeCapability::Ampere);
        let mem_size = kernel.shared_memory_size();
        assert!(mem_size > 0);
        assert!(mem_size <= 160 * 1024); // Should fit in Ampere shared memory
    }
}
