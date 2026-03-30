/// GPU kernel executor with runtime CUDA compilation
/// Uses cudarc for device management and NVRTC for JIT compilation

use crate::error::Result;

#[cfg(feature = "cuda")]
pub mod gpu_executor {
    use crate::error::Result;
    use cudarc::driver::{CudaDevice, LaunchAsync, DeviceRepr};
    use std::sync::Arc;

    pub struct GpuExecutor {
        device: Arc<CudaDevice>,
        kernels: GpuKernels,
    }

    pub struct GpuKernels {
        smith_waterman: cudarc::driver::CudaFunction,
        needleman_wunsch: cudarc::driver::CudaFunction,
        viterbi: cudarc::driver::CudaFunction,
    }

    impl GpuExecutor {
        /// Initialize GPU executor with JIT compilation
        pub fn new(device_id: u32) -> Result<Self> {
            let device = CudaDevice::new(device_id as usize)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU initialization failed: {}", e)
                ))?;

            let device = Arc::new(device);
            let kernels = Self::compile_kernels(&device)?;

            Ok(GpuExecutor { device, kernels })
        }

        /// Compile CUDA kernels at runtime
        fn compile_kernels(device: &Arc<CudaDevice>) -> Result<GpuKernels> {
            use cudarc::nvrtc::Ptx;

            // Compile Smith-Waterman kernel
            let sw_ptx = Ptx::compile("smith_waterman", super::super::cuda_kernels_rtc::SMITH_WATERMAN_KERNEL)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Smith-Waterman kernel compilation failed: {}", e)
                ))?;

            // Compile Needleman-Wunsch kernel
            let nw_ptx = Ptx::compile("needleman_wunsch", super::super::cuda_kernels_rtc::NEEDLEMAN_WUNSCH_KERNEL)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Needleman-Wunsch kernel compilation failed: {}", e)
                ))?;

            // Compile Viterbi kernel
            let viterbi_ptx = Ptx::compile("viterbi", super::super::cuda_kernels_rtc::VITERBI_HMM_KERNEL)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Viterbi kernel compilation failed: {}", e)
                ))?;

            // Load into device
            device.load_ptx(sw_ptx.as_str(), "smith_waterman", &["smith_waterman_kernel"])
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Failed to load Smith-Waterman kernel: {}", e)
                ))?;

            device.load_ptx(nw_ptx.as_str(), "needleman_wunsch", &["needleman_wunsch_kernel"])
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Failed to load Needleman-Wunsch kernel: {}", e)
                ))?;

            device.load_ptx(viterbi_ptx.as_str(), "viterbi", &["viterbi_forward_kernel"])
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Failed to load Viterbi kernel: {}", e)
                ))?;

            let smith_waterman = device.get_function("smith_waterman", "smith_waterman_kernel")
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Failed to get Smith-Waterman kernel function: {}", e)
                ))?;

            let needleman_wunsch = device.get_function("needleman_wunsch", "needleman_wunsch_kernel")
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Failed to get Needleman-Wunsch kernel function: {}", e)
                ))?;

            let viterbi = device.get_function("viterbi", "viterbi_forward_kernel")
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("Failed to get Viterbi kernel function: {}", e)
                ))?;

            Ok(GpuKernels {
                smith_waterman,
                needleman_wunsch,
                viterbi,
            })
        }

        /// Execute Smith-Waterman on GPU
        pub fn smith_waterman_gpu(
            &self,
            query: &Protein,
            subject: &Protein,
            matrix: &ScoringMatrix,
            penalty: &AffinePenalty,
        ) -> Result<AlignmentResult> {
            let query_seq = query.to_string();
            let subject_seq = subject.to_string();

            // Convert sequences to amino acid indices (0-19)
            let q_indices: Vec<u8> = query_seq.chars()
                .map(|c| AminoAcid::from_char(c).ok().map(|aa| aa as u8).unwrap_or(20))
                .collect();
            let s_indices: Vec<u8> = subject_seq.chars()
                .map(|c| AminoAcid::from_char(c).ok().map(|aa| aa as u8).unwrap_or(20))
                .collect();

            let m_len = q_indices.len();
            let n_len = s_indices.len();

            // Skip if sequences too small for GPU (CPU is faster)
            if m_len < 100 || n_len < 100 {
                return Err(crate::error::Error::AlignmentError(
                    "Sequences too small for GPU execution".to_string()
                ));
            }

            // Allocate GPU memory
            let d_seq1 = self.device.alloc_zeros::<u8>(m_len)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU allocation failed: {}", e)
                ))?;
            let d_seq2 = self.device.alloc_zeros::<u8>(n_len)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU allocation failed: {}", e)
                ))?;
            let d_matrix = self.device.alloc_zeros::<i32>(m_len * n_len)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU allocation failed: {}", e)
                ))?;
            let d_traceback = self.device.alloc_zeros::<i32>(m_len * n_len)
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU allocation failed: {}", e)
                ))?;

            // Copy data to GPU
            self.device.htod_copy_into(q_indices.clone(), d_seq1.clone())
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU data transfer failed: {}", e)
                ))?;
            self.device.htod_copy_into(s_indices.clone(), d_seq2.clone())
                .map_err(|e| crate::error::Error::AlignmentError(
                    format!("GPU data transfer failed: {}", e)
                ))?;

            // TODO: Get scoring matrix from device or copy it
            // For now, return error as this needs full integration
            Err(crate::error::Error::AlignmentError(
                "GPU Smith-Waterman execution framework ready (integration in progress)".to_string()
            ))
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub mod gpu_executor {
    use super::*;

    pub struct GpuExecutor;

    impl GpuExecutor {
        pub fn new(_device_id: u32) -> Result<Self> {
            Err(crate::error::Error::AlignmentError(
                "CUDA support not enabled (compile with --features cuda)".to_string()
            ))
        }
    }
}

pub use gpu_executor::GpuExecutor;
