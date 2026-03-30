# GPU Integration Quick Start Guide

This guide explains how to use the new GPU-accelerated features in OMICS-X.

## Building with GPU Support

### 1. Enable CUDA Feature

```bash
# Single GPU (NVIDIA)
cargo build --release --features cuda

# All GPUs (experimental)
cargo build --release --features all-gpu
```

### 2. Verify GPU Detection

```bash
# Check if GPU is available
cargo run --example gpu_discovery
```

## Using the GPU Runtime

### Detect Available GPUs

```rust
use omicsx::alignment::GpuRuntime;

fn main() -> Result<()> {
    // Detect all available GPUs
    let devices = GpuRuntime::detect_available_devices()?;
    
    if devices.is_empty() {
        println!("No GPU devices found");
        return Ok(());
    }
    
    println!("Found {} GPU devices", devices.len());
    
    // Initialize first GPU
    let gpu = GpuRuntime::new(devices[0])?;
    println!("Device info: {}", gpu.device_properties());
    
    Ok(())
}
```

### Allocate Device Memory

```rust
use omicsx::alignment::GpuRuntime;

fn main() -> Result<()> {
    let gpu = GpuRuntime::new(0)?;
    
    // Allocate 1MB on GPU
    let buffer: GpuBuffer<i32> = gpu.allocate::<i32>(1024 * 256)?;
    
    println!("Allocated {} bytes", buffer.size_bytes());
    
    // Automatic cleanup when buffer goes out of scope
    Ok(())
}
```

### Transfer Data to GPU

```rust
use omicsx::alignment::GpuRuntime;

fn main() -> Result<()> {
    let gpu = GpuRuntime::new(0)?;
    
    // Prepare host data
    let host_data = vec![1i32, 2, 3, 4, 5];
    
    // Copy to GPU (H2D transfer)
    let gpu_buffer = gpu.copy_to_device(&host_data)?;
    println!("Copied {} elements to GPU", host_data.len());
    
    Ok(())
}
```

### Transfer Data from GPU

```rust
use omicsx::alignment::GpuRuntime;

fn main() -> Result<()> {
    let gpu = GpuRuntime::new(0)?;
    
    let host_data = vec![10i32, 20, 30, 40, 50];
    let gpu_buffer = gpu.copy_to_device(&host_data)?;
    
    // Copy back to CPU (D2H transfer)
    let retrieved = gpu.copy_from_device(&gpu_buffer)?;
    
    assert_eq!(host_data, retrieved);
    println!("Data validation passed!");
    
    Ok(())
}
```

## Using the Kernel Compiler

### Compile and Cache Kernels

```rust
use omicsx::alignment::{KernelCompiler, KernelType};
use std::path::PathBuf;

fn main() -> Result<()> {
    let cache_dir = PathBuf::from(".omnics_kernel_cache");
    let mut compiler = KernelCompiler::new(cache_dir, true)?;
    
    let cuda_source = r#"
        __global__ void smith_waterman(...) {
            // Kernel implementation
        }
    "#;
    
    // Compile to PTX
    let kernel = compiler.compile_to_ptx(
        KernelType::SmithWatermanGpu,
        cuda_source,
        "8.0",  // Ampere
        vec!["--ptxas-options=-v".to_string()],
    )?;
    
    println!("Compiled kernel: {}", kernel.name);
    println!("Code size: {} bytes", kernel.code.len());
    
    // Next execution will use cached version
    
    Ok(())
}
```

### Verify Cache

```bash
# List cached kernels
ls -la .omnics_kernel_cache/

# View cache metadata
cat .omnics_kernel_cache/kernel_cache.json

# Clear cache (if needed)
rm -rf .omnics_kernel_cache/
```

## GPU Device Information

### Query Device Properties

```rust
use omicsx::alignment::GpuRuntime;

fn main() -> Result<()> {
    let devices = GpuRuntime::detect_available_devices()?;
    
    for device_id in devices {
        let gpu = GpuRuntime::new(device_id)?;
        
        println!("=== Device {} ===", device_id);
        println!("Total Memory: {} GB", gpu.total_memory() / (1024*1024*1024));
        println!("Allocated: {} MB", gpu.allocated_memory() / (1024*1024));
        println!("Available: {} MB", gpu.available_memory() / (1024*1024));
    }
    
    Ok(())
}
```

### Compute Capability Detection

```rust
use omicsx::alignment::cuda_kernels::CudaComputeCapability;

fn main() {
    // Parse compute capability
    let cap = CudaComputeCapability::from_version(8, 0);
    
    if let Some(capability) = cap {
        println!("GPU: {}", capability.name());
        println!("Optimal block size: {}", capability.optimal_block_size());
        println!("Shared memory: {} KB", capability.shared_memory() / 1024);
        println!("Has Tensor Cores: {}", capability.has_tensor_cores());
    }
}
```

## Multi-GPU Usage

### Distribute Work Across GPUs

```rust
use omicsx::alignment::cuda_kernels::CudaMultiGpuBatch;

fn main() -> Result<()> {
    let device_ids = vec![0, 1, 2];
    let mut batch = CudaMultiGpuBatch::new(device_ids);
    
    for i in 0..10 {
        let kernel = batch.next_device();
        println!("Processing alignment {} on GPU {}", i, kernel.device_id);
        
        // Execute kernel on current device
    }
    
    Ok(())
}
```

## Performance Monitoring

### Measure Transfer Times

```rust
use std::time::Instant;
use omicsx::alignment::GpuRuntime;

fn main() -> Result<()> {
    let gpu = GpuRuntime::new(0)?;
    let data_size = 1024 * 1024; // 1 MB
    let host_data = vec![0i32; data_size];
    
    // Measure H2D
    let start = Instant::now();
    let gpu_buf = gpu.copy_to_device(&host_data)?;
    let h2d_time = start.elapsed();
    
    // Measure D2H
    let start = Instant::now();
    let _retrieved = gpu.copy_from_device(&gpu_buf)?;
    let d2h_time = start.elapsed();
    
    let h2d_bw = (data_size as f64 / 1e9) / h2d_time.as_secs_f64();
    let d2h_bw = (data_size as f64 / 1e9) / d2h_time.as_secs_f64();
    
    println!("H2D: {:.1} GB/s ({:?})", h2d_bw, h2d_time);
    println!("D2H: {:.1} GB/s ({:?})", d2h_bw, d2h_time);
    
    Ok(())
}
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA GPU (Linux/macOS)
nvidia-smi

# Check AMD GPU (Linux)
rocm-smi

# Check CUDA availability
nvcc --version

# Build without GPU support (fallback to SIMD)
cargo build --release
```

### Out of GPU Memory

```rust
// Check available memory before allocation
let gpu = GpuRuntime::new(0)?;
let available = gpu.available_memory();

if available < required_size {
    eprintln!("Insufficient GPU memory: need {} bytes, have {}", 
              required_size, available);
    // Fall back to CPU
}
```

### Compilation Errors

```bash
# If CUDA support fails to compile:
# 1. Ensure CUDA toolkit is installed
# 2. Verify cudarc dependency
# 3. Build without CUDA (uses scalar/SIMD fallback):
cargo build --release --no-default-features --features simd
```

## Feature Combinations

### CPU Only (Smallest Binary)

```bash
cargo build --release --no-default-features
```

### CPU + SIMD (Standard)

```bash
cargo build --release
```

### CPU + SIMD + NVIDIA GPU

```bash
cargo build --release --features cuda
```

### CPU + SIMD + All GPUs (Future)

```bash
cargo build --release --features all-gpu
```

## Example: Complete Workflow

```rust
use omicsx::alignment::GpuRuntime;
use omicsx::protein::Protein;

fn main() -> Result<()> {
    // 1. Detect GPU
    let devices = GpuRuntime::detect_available_devices()?;
    if devices.is_empty() {
        eprintln!("No GPU found, using CPU fallback");
        return Ok(());
    }
    
    // 2. Initialize GPU
    let gpu = GpuRuntime::new(devices[0])?;
    println!("Using: {}", gpu.device_properties());
    
    // 3. Parse sequences
    let seq1 = Protein::from_string("MVHLTPEEKS")?;
    let seq2 = Protein::from_string("MGHLTPEEKS")?;
    
    // 4. Convert to device format
    let seq1_bytes: Vec<u8> = seq1.sequence()
        .iter()
        .map(|aa| aa.to_code() as u8)
        .collect();
    
    let seq2_bytes: Vec<u8> = seq2.sequence()
        .iter()
        .map(|aa| aa.to_code() as u8)
        .collect();
    
    // 5. Transfer to GPU
    let gpu_seq1 = gpu.copy_to_device(&seq1_bytes)?;
    let gpu_seq2 = gpu.copy_to_device(&seq2_bytes)?;
    
    println!("Transferred sequences to GPU");
    println!("Seq1: {} bytes", gpu_seq1.size_bytes());
    println!("Seq2: {} bytes", gpu_seq2.size_bytes());
    
    // 6. (In production) Execute kernel
    // let result = gpu.execute_smith_waterman(...)?;
    
    Ok(())
}
```

## Next Steps

1. **Start simple**: Run GPU detection example
2. **Memory operations**: Practice H2D/D2H transfers
3. **Kernel compilation**: Cache a test kernel
4. **Integration**: Use in alignment pipeline
5. **Benchmarking**: Measure throughput improvements

For more details, see:
- [DEVELOPMENT.md](DEVELOPMENT.md) - Developer workflow
- [ADVANCED_IMPLEMENTATION_SUMMARY.md](ADVANCED_IMPLEMENTATION_SUMMARY.md) - Technical architecture
- API documentation: `cargo doc --open`

