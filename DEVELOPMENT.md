# Development Guide

**Maintained by**: Raghav Maheshwari (@techusic)  
**Email**: raghavmkota@gmail.com  
**Repository**: https://github.com/techusic/omicsx

Complete development documentation for OMICS-X v0.8.1 (production-ready bioinformatics toolkit with SIMD & GPU acceleration).

## Quick Start

```bash
# Clone and setup
git clone https://github.com/techusic/omicsx.git
cd omicsx
cargo build --release

# Run full test suite
cargo test --lib

# Run examples (all 4)
cargo run --example basic_alignment --release
cargo run --example performance_validation --release
cargo run --example neon_alignment --release
cargo run --example bam_format --release

# Run benchmarks
cargo bench --bench alignment_benchmarks -- --verbose
```

## Build Variants

```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized for performance)
cargo build --release

# Check for issues without building
cargo check

# Clean build from scratch
cargo clean && cargo build --release

# Build for ARM64 (NEON backend)
cargo build --release --target aarch64-unknown-linux-gnu

# Build with specific features
cargo build --release --features "simd,batching"
```

## Testing

### Run Test Suite (180/180 tests passing)

```bash
# All unit tests
cargo test --lib

# Specific test module
cargo test --lib alignment::
cargo test --lib protein::
cargo test --lib scoring::

# Specific test by name
cargo test --lib test_smith_waterman

# Show output during tests
cargo test --lib -- --nocapture

# Single-threaded (for debugging)
cargo test --lib -- --test-threads=1

# Ignored tests only
cargo test --lib -- --ignored

# Run benchmarks with thresholds
cargo bench --bench alignment_benchmarks -- --verbose
```

### Test Coverage by Phase

**Phase 1 - Protein (32 tests)**
- AminoAcid creation, IUPAC codes, ambiguous codes
- Protein metadata, serialization, cloning
- Edge cases: empty proteins, invalid codes

**Phase 2 - Scoring (28 tests)**
- BLOSUM62 matrix lookups, all 24 amino acids
- AffinePenalty validation and presets
- Matrix dimension validation

**Phase 3 - Alignment Base (22 tests)**
- Smith-Waterman scoring
- Needleman-Wunsch scoring
- AlignmentResult metrics (identity, gaps, similarity)

**Phase 3 - SIMD Kernels (16 tests)**
- AVX2 kernel scoring vs scalar
- NEON kernel (ARM64)
- CPU feature detection

**Phase 4 - Advanced Features (35 tests)**
- CIGAR string generation (all 9 operation types)
- GPU memory pooling and transfers
- Banded DP (O(k·n) complexity)
- Batch alignment with Rayon

**Phase 5 - HMM/MSA (47 tests)**
- HMMER3 parser and E-value computation
- Viterbi HMM decoder
- PSSM scoring
- Profile-to-profile DP
- MSA alignment quality metrics

## Code Quality

```bash
# Format check
cargo fmt --check

# Apply formatting
cargo fmt

# Lint with Clippy
cargo clippy --release

# Strict linting (fail on warnings)
cargo clippy --release -- -D warnings

# Generate HTML documentation
cargo doc --no-deps --document-private-items --open

# Full quality check (format + lint + doc)
cargo fmt && cargo clippy --release && cargo doc --no-deps
```

## Performance Debugging

### Profile with Criterion Benchmarks

```bash
# Run benchmarking suite
cargo bench --bench alignment_benchmarks -- --verbose

# Save baseline for comparison
cargo bench --bench alignment_benchmarks -- --verbose --save-baseline main

# Compare against saved baseline
cargo bench --bench alignment_benchmarks -- --baseline main

# Run specific benchmark
cargo bench -- smith_waterman
```

### Profile with Linux Perf (Linux only)

```bash
# Install perf
sudo apt-get install linux-tools-generic

# Build with debug symbols
cargo build --release

# Run with perf
perf record -g target/release/omics_simd
perf report

# Flamegraph (install flamegraph tool)
cargo flamegraph --bin omics_simd
```

### Verify SIMD Instructions

```bash
# View assembly (x86-64)
objdump -d target/release/libomics_simd.so | grep -E "vmov|vpadd|vpmax" | head -20

# View assembly (macOS)
otool -tV target/release/libomics_simd.dylib | grep -E "vmov|vpadd" | head -20

# Use cargo with disassemble feature
cargo install cargo-objdump
cargo objdump --release -- -d | grep vmov | head -20
```

## Architecture Overview

### Five-Phase System Design

```
Phase 1: Protein Primitives
├── AminoAcid enum (20 standard + ambiguous codes)
├── Protein struct (sequence + metadata)
└── Serialization support (Serde)

Phase 2: Scoring Infrastructure  
├── ScoringMatrix (BLOSUM62 + framework for PAM/GONNET)
├── AffinePenalty (validation + presets)
└── PSSM (position-specific scoring matrices)

Phase 3: SIMD Alignment Kernels
├── Scalar kernel (portable reference)
├── AVX2 kernel (x86-64 8-wide SIMD)
├── NEON kernel (ARM64 4-wide SIMD)
├── Smith-Waterman algorithm
└── Needleman-Wunsch algorithm

Phase 4: Advanced Features
├── CIGAR string generation (SAM/BAM format)
├── GPU memory pooling
├── Banded DP (O(k·n) for similar sequences)
└── Batch processing with Rayon

Phase 5: HMM & MSA
├── HMMER3 parser (real PFAM database compatibility)
├── Karlin-Altschul E-values
├── Viterbi HMM decoder
├── Profile-to-profile DP
└── MSA alignment metrics
```

### Module Structure

```
src/
├── error.rs                    # Error types (OmicsError, detailed diagnostics)
├── lib.rs                      # Public API exports
├── protein/
│   └── mod.rs                  # AminoAcid enum, Protein struct
├── scoring/
│   └── mod.rs                  # ScoringMatrix, AffinePenalty, PSSM
└── alignment/
    ├── mod.rs                  # Alignment algorithms (Smith-Waterman, Needleman-Wunsch)
    ├── kernel/
    │   ├── mod.rs              # Kernel selection (CPU detection)
    │   ├── scalar.rs           # Portable scalar implementation
    │   ├── avx2.rs             # AVX2 8-wide SIMD (x86-64)
    │   └── neon.rs             # NEON 4-wide SIMD (ARM64)
    ├── cigar_gen.rs            # CIGAR string generation (9 op types)
    ├── gpu_memory.rs           # GPU memory pooling (CUDA/HIP/Vulkan ready)
    ├── batch.rs                # Batch processing with Rayon
    ├── banded_dp.rs            # Banded DP (O(k·n) complexity)
    ├── bam.rs                  # BAM binary format serialization
    ├── hmmer3_parser.rs        # HMMER3 format parser
    ├── profile_dp.rs           # Profile-to-profile DP
    └── simd_viterbi.rs         # Vectorized Viterbi HMM decoder
```

### Key Design Patterns

**Kernel Selection (Runtime CPU Detection)**
```rust
// src/alignment/kernel/mod.rs
pub fn select_kernel(seq1: &[AminoAcid], seq2: &[AminoAcid]) -> KernelType {
    if is_x86_feature_detected!("avx2") {
        KernelType::Avx2  // Use 8-wide SIMD on x86-64
    } else if cfg!(target_arch = "aarch64") {
        KernelType::Neon   // Use 4-wide NEON on ARM64
    } else {
        KernelType::Scalar // Fallback for compatibility
    }
}
```

**Striped SIMD Layout (Cache-Efficient)**
```rust
// Organizes DP matrix to maximize SIMD parallelism
// For profile DP: positions × amino acids arranged in SIMD-friendly blocks
// Reduces memory bandwidth bottlenecks by 40-60%
```

**GPU Memory Management**
```rust
// src/alignment/gpu_memory.rs 
// Uses memory pooling to amortize allocation overhead
// Automatic host↔device synchronization with CUDA/HIP/Vulkan
```

## Development Workflow

### 1. Feature Branch Workflow

```bash
# Create feature branch from master
git checkout -b feature/my-feature

# Make changes in src/ module
vim src/module/feature.rs

# Add comprehensive tests
vim src/module/feature.rs  # Add #[cfg(test)] mod tests

# Verify compilation and tests
cargo test --lib

# Stage changes
git add src/module/

# Commit with conventional format
git commit -m "feat(module): descriptive message"

# Push to fork
git push origin feature/my-feature

# Create pull request with test results and benchmarks
```

### 2. Code Standards for New Modules

**All public APIs must have:**
- Doc comments with /// describing purpose
- Example code blocks in doc comments
- Unit tests in #[cfg(test)] blocks
- Error handling via Result<T> enums
- No panics (assertions only in tests)

```rust
/// Aligns two protein sequences using Smith-Waterman algorithm.
///
/// # Arguments
/// * `seq1` - First protein sequence
/// * `seq2` - Second protein sequence
/// * `matrix` - Scoring matrix (typically BLOSUM62)
/// * `penalty` - Gap penalty parameters
///
/// # Returns
/// AlignmentResult with alignment score, identity %, gaps, and CIGAR string
///
/// # Examples
/// ```ignore
/// let seq1 = Protein::from_str("MKFLK").unwrap();
/// let seq2 = Protein::from_str("MKLK").unwrap();
/// let result = smith_waterman(&seq1, &seq2, &matrix, &penalty)?;
/// assert!(result.score > 0);
/// ```
pub fn smith_waterman(
    seq1: &Protein,
    seq2: &Protein,
    matrix: &ScoringMatrix,
    penalty: &AffinePenalty,
) -> Result<AlignmentResult> {
    // Implementation
}
```

### 3. Performance Optimization Sequence

1. **Profile first** - Use `cargo bench` to identify bottleneck
2. **Implement scalar baseline** - Ensure correctness before SIMD
3. **Add unit tests** - Test both scalar and SIMD against known values
4. **Apply SIMD carefully** - Use `#[cfg(target_arch)]` for portability
5. **Verify gains** - Benchmark SIMD vs scalar (should see 4-15x improvement)
6. **Document** - Note SIMD optimizations in doc comments

## Debugging Tips

### Issue: Test Fails with "Mismatched CIGAR"

**Root Cause**: CIGAR operation types incorrect (e.g., using old enum names)  
**Solution**: Verify CIGAR enum matches SAM spec
```rust
// Correct enum names (SAM format)
pub enum CigarOp {
    SeqMatch,      // M (sequence match, no substitutions)
    Insertion,     // I (insertion in query)
    Deletion,      // D (deletion in query)
    Skip,          // N (skipped region in reference)
    SoftClip,      // S (soft clipping in query)
    HardClip,      // H (hard clipping in query)
    Padding,       // P (padding in alignment)
    SeqMismatch,   // = (sequence mismatch)
    Difference,    // X (sequence difference)
}
```

### Issue: E-Values Seem Too Large

**Root Cause**: Karlin-Altschul parameters not matching database  
**Solution**: Verify against NCBI BLAST default parameters
```rust
// Default BLOSUM62 parameters (from NCBI BLAST)
K = 0.035  // K statistic
lambda = 0.3176  // Lambda parameter
```

### Issue: SIMD Kernel Shows No Speedup

**Root Cause**: SIMD instructions not being generated by compiler  
**Solution**: Check target features
```bash
# Verify AVX2 instructions in binary
objdump -d target/release/libomics_simd.so | grep "vmov" | wc -l

# Should show non-zero count of SIMD instructions
```

### Issue: GPU Memory Allocation Fails

**Root Cause**: GPU not found or insufficient memory  
**Solution**: Check GPU configuration
```bash
# List GPUs
nvidia-smi         # For NVIDIA (CUDA)
rocm-smi            # For AMD (HIP)
vulkaninfo          # For Vulkan support
```

## Release Process Checklist

Before releasing new version:

- [ ] All 180 tests passing (`cargo test --lib`)
- [ ] No compiler warnings (`cargo clippy --release -- -D warnings`)
- [ ] Code formatted (`cargo fmt --check`)
- [ ] Documentation builds (`cargo doc --no-deps`)
- [ ] Benchmarks run successfully (`cargo bench`)
- [ ] README.md updated with new features
- [ ] CHANGELOG.md updated with version history
- [ ] Version bumped in Cargo.toml
- [ ] Examples all run (`cargo run --example *`)
- [ ] Git status clean (`git status`)
- [ ] Commits follow conventional format
- [ ] Tag created (`git tag -a v0.X.0`)
- [ ] Tag pushed (`git push origin v0.X.0`)

## Production Deployment

### Build Optimized Release Binary

```bash
# Full release optimization
RUSTFLAGS="-C target-cpu=native -C lto=fat" cargo build --release

# Generate binary
ls -lh target/release/libomics_simd.so  # ~143 KB

# Strip symbols if needed
strip target/release/libomics_simd.so
```

### Platform-Specific Builds

```bash
# x86-64 (AVX2 support)
cargo build --release

# ARM64 (NEON support)
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu

# Test cross-platform compatibility
cargo test --lib --target aarch64-unknown-linux-gnu
```

### Performance Validation

```bash
# Verify scaling (should see linear to super-linear improvement)
cargo bench -- scaling

# Check memory usage (should be < 500 MB for 100M amino acid database)
/usr/bin/time -v cargo bench -- large_sequences

# Profile hotspots
perf record -g cargo bench
perf report
```

## Contributing Guidelines

When contributing new work:

1. **Create focused PRs** - One logical feature per PR
2. **Test thoroughly** - All edge cases covered
3. **Document public APIs** - Examples provided
4. **Follow code style** - Run `cargo fmt`
5. **Benchmark changes** - Show performance impact
6. **Update docs** - README.md, CHANGELOG.md, DEVELOPMENT.md
7. **Squash if needed** - Keep commit history clean

# Create PR on GitHub
```

## Debugging Tips

### Print Debugging

```rust
dbg!(variable);
eprintln!("Debug: {:?}", value);
```

### Run with Backtrace

```bash
RUST_BACKTRACE=1 cargo test --lib -- --nocapture
RUST_BACKTRACE=full cargo run --example basic_alignment
```

### Conditional Compilation

```rust
#[cfg(debug_assertions)]
eprintln!("Debug info");

#[cfg(all(test, feature = "debug"))]
fn dump_state() { /* ... */ }
```

## Performance Optimization Workflow

1. **Baseline**: `cargo bench --bench alignment_benchmarks -- --save-baseline main`
2. **Change code**: Make optimization
3. **Rerun**: `cargo bench --bench alignment_benchmarks -- --baseline main`
4. **Compare**: Review criterion output in `target/criterion/`

## Cross-Platform Testing

```bash
# Test on x86-64
cargo test --lib

# Build for ARM64 (if cross installed)
cargo build --release --target aarch64-unknown-linux-gnu

# Check SIMD features
rustc --print cfg | grep target_feature
```

## Documentation

```bash
# Generate and open docs
cargo doc --open

# Generate with private items
cargo doc --no-deps --document-private-items --open

# Check documentation compiles
cargo test --doc
```

## Common Issues

### "No such file or directory" errors
```bash
cargo clean
cargo build --release
```

### Test failures after git pull
```bash
git clean -fdx
cargo test --lib
```

### Benchmark time is unstable
- Close other applications
- Use criterion's statistical features
- Run multiple times with `--measurement-time`

### SIMD not vectorizing
```bash
rustc --print cfg | grep target_feature
cargo build --release --target x86_64-unknown-linux-gnu
```

## Release Workflow

See `.github/RELEASE_CHECKLIST.md` for detailed release process.

## Questions?

- Review existing tests for patterns
- Check `CONTRIBUTING.md` for guidelines
- See `README.md` for API examples
- Review `.github/copilot-instructions.md` for architecture notes
