//! GPU Acceleration Example
//!
//! Demonstrates GPU-accelerated sequence alignment with automatic backend selection.
//! Shows how to use CUDA, HIP, and Vulkan compute shaders for high-performance alignment.

use omicsx::alignment::{GpuDispatcher, AlignmentStrategy, GpuAvailability};
use omicsx::protein::{Protein, AminoAcid};
use omicsx::scoring::ScoringMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OMICS-X GPU Acceleration Demo ===\n");

    // Initialize GPU dispatcher
    let gpu_dispatcher = GpuDispatcher::new();
    println!("{}", gpu_dispatcher.status());
    println!();

    // Display available GPU devices
    let devices = gpu_dispatcher.device_info();
    if !devices.is_empty() {
        println!("Available GPU Devices:");
        for (idx, device) in devices.iter().enumerate() {
            println!("  [{}] {}", idx, device);
        }
        println!();
    }

    // Display selected backend
    println!(
        "Selected GPU Backend: {}",
        gpu_dispatcher.selected_backend()
    );
    println!("GPU Available: {}", gpu_dispatcher.has_gpu());
    println!();

    // Display optimization hints
    let hints = gpu_dispatcher.optimization_hints();
    println!("GPU Optimization Hints:");
    println!("  Optimal Block Size: {}", hints.optimal_block_size);
    println!("  Concurrent Blocks: {}", hints.concurrent_blocks);
    println!("  Single Pass Max Length: {}", hints.single_pass_max_len);
    println!("  Use Shared Memory: {}", hints.use_shared_memory);
    println!("  Coales Memory Access: {}", hints.coalesce_memory);
    println!("  Warp Size: {}", hints.warp_size);
    println!();

    // Example 1: Small alignment (should use SIMD or small GPU)
    println!("Example 1: Small Sequence Alignment");
    println!("---");
    let seq1_small = vec![
        AminoAcid::Alanine,
        AminoAcid::Glycine,
        AminoAcid::Serine,
        AminoAcid::Lysine,
    ];
    let seq2_small = vec![AminoAcid::Alanine, AminoAcid::Serine, AminoAcid::Lysine];

    let strategy_small =
        gpu_dispatcher.dispatch_alignment(seq1_small.len(), seq2_small.len(), None);
    println!("  Sequence 1 Length: {}", seq1_small.len());
    println!("  Sequence 2 Length: {}", seq2_small.len());
    println!("  Selected Strategy: {:?}", strategy_small);
    println!("  Estimated Speedup: {:.1}x", {
        use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;
        GpuDispatcherStrategy::gpu_speedup_factor(strategy_small)
    });
    println!();

    // Example 2: Medium alignment (should use GPU if available)
    println!("Example 2: Medium Sequence Alignment");
    println!("---");
    let len1_med = 1000;
    let len2_med = 1000;
    let strategy_med = gpu_dispatcher.dispatch_alignment(len1_med, len2_med, None);
    println!("  Sequence 1 Length: {}", len1_med);
    println!("  Sequence 2 Length: {}", len2_med);
    println!("  Selected Strategy: {:?}", strategy_med);
    println!("  Estimated Speedup: {:.1}x", {
        use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;
        GpuDispatcherStrategy::gpu_speedup_factor(strategy_med)
    });
    println!();

    // Example 3: Large alignment with high similarity (banded DP)
    println!("Example 3: Large Sequence Alignment (High Similarity)");
    println!("---");
    let len1_large = 50000;
    let len2_large = 50000;
    let strategy_large_similar =
        gpu_dispatcher.dispatch_alignment(len1_large, len2_large, Some(0.85));
    println!("  Sequence 1 Length: {}", len1_large);
    println!("  Sequence 2 Length: {}", len2_large);
    println!("  Sequence Similarity: ~85%");
    println!("  Selected Strategy: {:?}", strategy_large_similar);
    println!("  Estimated Speedup: {:.1}x", {
        use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;
        GpuDispatcherStrategy::gpu_speedup_factor(strategy_large_similar)
    });
    println!();

    // Example 4: Very large alignment (should use tiled GPU)
    println!("Example 4: Very Large Sequence Alignment");
    println!("---");
    let len1_xlarge = 100000;
    let len2_xlarge = 100000;
    let strategy_xlarge = gpu_dispatcher.dispatch_alignment(len1_xlarge, len2_xlarge, None);
    println!("  Sequence 1 Length: {}", len1_xlarge);
    println!("  Sequence 2 Length: {}", len2_xlarge);
    println!("  Selected Strategy: {:?}", strategy_xlarge);
    println!("  Estimated GPU Memory: {} MB", {
        use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;
        GpuDispatcherStrategy::estimate_gpu_memory(len1_xlarge, len2_xlarge) / (1024 * 1024)
    });
    println!("  Fits in GPU Memory: {}", {
        use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;
        if gpu_dispatcher.has_gpu() {
            let device_mem = gpu_dispatcher.device_info()[0].total_memory;
            GpuDispatcherStrategy::fits_in_gpu_memory(len1_xlarge, len2_xlarge, device_mem)
        } else {
            false
        }
    });
    println!("  Estimated Speedup: {:.1}x", {
        use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;
        GpuDispatcherStrategy::gpu_speedup_factor(strategy_xlarge)
    });
    println!();

    // Example 5: GPU shader source code preview (Vulkan)
    #[cfg(feature = "vulkan")]
    {
        println!("Example 5: Vulkan Compute Shader Source");
        println!("---");
        use omicsx::alignment::kernel::vulkan::VulkanComputeKernel;
        let vulkan_kernel = VulkanComputeKernel::default();
        let glsl_source = vulkan_kernel.smith_waterman_glsl();
        println!("First 500 chars of Smith-Waterman GLSL shader:");
        println!("{}", &glsl_source[..500.min(glsl_source.len())]);
        println!()
    }

    // Summary
    println!("=== GPU Acceleration Summary ===");
    println!("GPU Status: {}", if gpu_dispatcher.has_gpu() {
        "ENABLED"
    } else {
        "DISABLED (no GPU backends compiled)"
    });
    println!("Active Backend: {}", gpu_dispatcher.selected_backend());
    println!("Supported Strategies: {:?}", vec![
        AlignmentStrategy::Scalar,
        AlignmentStrategy::Simd,
        AlignmentStrategy::Banded,
        AlignmentStrategy::GpuFull,
        AlignmentStrategy::GpuTiled,
    ]);

    println!("\nTo enable GPU acceleration, compile with:");
    println!("  cargo build --release --features cuda");
    println!("  cargo build --release --features hip");
    println!("  cargo build --release --features vulkan");
    println!("  cargo build --release --features all-gpu");

    Ok(())
}
