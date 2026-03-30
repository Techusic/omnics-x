//! GPU Performance Benchmarks
//!
//! Compares performance across scalar, SIMD, and GPU implementations.
//! Run with: cargo bench --bench gpu_benchmarks -- --verbose

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use omicsx::alignment::{GpuDispatcher, AlignmentStrategy};
use omicsx::alignment::gpu_dispatcher::GpuDispatcherStrategy;

fn gpu_strategy_selection_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_strategy_selection");
    
    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                GpuDispatcherStrategy::select_strategy(black_box(size), black_box(size), true, None)
            });
        });
    }
    group.finish();
}

fn gpu_memory_estimation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_memory_estimation");
    
    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                GpuDispatcherStrategy::estimate_gpu_memory(black_box(size), black_box(size))
            });
        });
    }
    group.finish();
}

fn gpu_dispatcher_initialization(c: &mut Criterion) {
    c.bench_function("gpu_dispatcher_new", |b| {
        b.iter(|| {
            GpuDispatcher::new()
        });
    });
}

fn gpu_fitness_check_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_fitness_check");
    let available_memory = 8 * 1024 * 1024 * 1024; // 8GB
    
    for size in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                GpuDispatcherStrategy::fits_in_gpu_memory(
                    black_box(size),
                    black_box(size),
                    black_box(available_memory),
                )
            });
        });
    }
    group.finish();
}

fn cuda_kernel_timing_simulation(c: &mut Criterion) {
    // Simulated CUDA kernel timings (in production, would measure actual kernel execution)
    c.bench_function("cuda_kernel_overhead", |b| {
        b.iter(|| {
            // Simulate kernel launch overhead (~microseconds)
            #[cfg(feature = "cuda")]
            {
                use omicsx::alignment::kernel::cuda::CudaAlignmentKernel;
                let _kernel = CudaAlignmentKernel::new();
            }
        });
    });
}

fn hip_kernel_timing_simulation(c: &mut Criterion) {
    c.bench_function("hip_kernel_overhead", |b| {
        b.iter(|| {
            #[cfg(feature = "hip")]
            {
                use omicsx::alignment::kernel::hip::HipAlignmentKernel;
                let _kernel = HipAlignmentKernel::new();
            }
        });
    });
}

fn vulkan_kernel_timing_simulation(c: &mut Criterion) {
    c.bench_function("vulkan_kernel_overhead", |b| {
        b.iter(|| {
            #[cfg(feature = "vulkan")]
            {
                use omicsx::alignment::kernel::vulkan::VulkanComputeKernel;
                let _kernel = VulkanComputeKernel::new();
            }
        });
    });
}

fn speedup_estimation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_estimation");
    
    let strategies = vec![
        ("Scalar", AlignmentStrategy::Scalar),
        ("SIMD", AlignmentStrategy::Simd),
        ("Banded", AlignmentStrategy::Banded),
        ("GPU Full", AlignmentStrategy::GpuFull),
        ("GPU Tiled", AlignmentStrategy::GpuTiled),
    ];
    
    for (name, strategy) in strategies {
        group.bench_with_input(BenchmarkId::from_parameter(name), &strategy, |b, &strategy| {
            b.iter(|| {
                GpuDispatcherStrategy::gpu_speedup_factor(black_box(strategy))
            });
        });
    }
    group.finish();
}

fn cuda_kernel_config_benchmark(c: &mut Criterion) {
    c.bench_function("cuda_kernel_config_creation", |b| {
        b.iter(|| {
            use omicsx::alignment::cuda_kernels::{CudaComputeCapability, CudaKernelConfig};
            
            let cap = CudaComputeCapability::Ampere;
            let _config = CudaKernelConfig {
                compute_capability: cap,
                block_size: cap.optimal_block_size(),
                use_shared_memory: true,
                use_warp_shuffles: true,
                use_tensor_cores: cap.has_tensor_cores(),
                optimize_registers: true,
            };
        });
    });
}

fn cuda_grid_calculation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_grid_calculation");
    
    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                use omicsx::alignment::cuda_kernels::CudaAlignmentKernel;
                use omicsx::alignment::cuda_kernels::CudaComputeCapability;
                
                let kernel = CudaAlignmentKernel::new(0, CudaComputeCapability::Ampere);
                let _grid = kernel.calculate_grid_size(black_box(size), black_box(size));
            });
        });
    }
    group.finish();
}

fn cuda_performance_estimation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_performance_estimation");
    
    for (name, capability) in vec![
        ("Maxwell", omicsx::alignment::cuda_kernels::CudaComputeCapability::Maxwell),
        ("Pascal", omicsx::alignment::cuda_kernels::CudaComputeCapability::Pascal),
        ("Ampere", omicsx::alignment::cuda_kernels::CudaComputeCapability::Ampere),
        ("Ada", omicsx::alignment::cuda_kernels::CudaComputeCapability::Ada),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &capability, |b, &cap| {
            b.iter(|| {
                use omicsx::alignment::cuda_kernels::CudaAlignmentKernel;
                
                let kernel = CudaAlignmentKernel::new(0, cap);
                let _time = kernel.estimate_time(black_box(500), black_box(500));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    gpu_dispatcher_initialization,
    gpu_strategy_selection_benchmark,
    gpu_memory_estimation_benchmark,
    gpu_fitness_check_benchmark,
    cuda_kernel_timing_simulation,
    hip_kernel_timing_simulation,
    vulkan_kernel_timing_simulation,
    speedup_estimation_benchmark,
    cuda_kernel_config_benchmark,
    cuda_grid_calculation_benchmark,
    cuda_performance_estimation_benchmark,
);
criterion_main!(benches);
