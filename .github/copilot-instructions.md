<!-- OMICS-SIMD Project Customization Instructions -->

# OMICS-SIMD: Vectorizing Genomics with SIMD Acceleration

## Project Context

This is a Rust library implementing SIMD-accelerated sequence alignment for petabyte-scale genomic data. The project follows a three-phase architecture:

**Phase 1: Protein Primitives** - Type-safe amino acid and protein polymer representations  
**Phase 2: Scoring Infrastructure** - BLOSUM/PAM matrices and affine gap penalties  
**Phase 3: SIMD Kernels** - AVX2/NEON-optimized alignment algorithms

## Development Guidelines

### Code Style & Standards

- Use idiomatic Rust conventions (rustfmt, clippy)
- Maintain comprehensive documentation with doc comments
- All public APIs must have examples
- Target Rust 2021 edition features
- Leverage the type system for correctness

### Type Safety Requirements

- Protein sequences use `Vec<AminoAcid>` enums (no raw u8s)
- Scoring matrices validate dimensions at creation
- Gap penalties enforce negative values via validation
- All error conditions return `Result<T>` types
- No panics in library code (only assertions in tests)

### Architecture Principles

- **Memory Safety**: Leverage Rust's ownership and borrow checker
- **Correctness First**: Scalar baseline implementations before SIMD
- **Modularity**: Each phase independent with clear interfaces
- **Performance**: SIMD optimizations only after correctness verified
- **Hardware Portability**: Support both x86 (AVX2) and ARM (NEON)

### Performance Optimization Strategy

1. Profile before optimizing (use criterion benches)
2. Implement scalar baseline first
3. Identify hot paths (usually DP inner loop)
4. Apply SIMD carefully using std::arch
5. Benchmark against scalar to verify gains
6. Target 8-15x speedup over scalar implementations

### Testing Requirements

- Unit tests in each module for basic functionality
- Integration tests in `/tests/` for end-to-end flows
- Benchmarks in `/benches/` comparing SIMD vs scalar
- Test data includes edge cases (empty sequences, single amino acid, mismatches)
- Correctness verification against known benchmark alignments

### Documentation Standards

- Doc comments on all public items with examples
- Module-level documentation explaining purpose
- Technical design notes for complex algorithms
- README.md kept current with implementation status
- Examples demonstrate common usage patterns

## Implementation Checklist - Phase 1 ✅

- [x] `AminoAcid` enum with IUPAC codes
- [x] `Protein` struct with metadata
- [x] From/to string conversions
- [x] Serialization support (Serde)
- [x] Unit tests with edge cases
- [x] Documentation and examples

## Implementation Checklist - Phase 2 ✅

- [x] `AffinePenalty` with validation
- [x] `ScoringMatrix` with BLOSUM62 data
- [x] Predefined matrices (BLOSUM45/80, PAM30/70)
- [x] Modular matrix selection
- [x] Unit tests for matrix lookups
- [x] Penalty preset profiles

## Implementation Checklist - Phase 3 ✅ Complete

- [x] Smith-Waterman scalar implementation
## Implementation Checklist - Phase 1 ✅

- [x] `AminoAcid` enum with IUPAC codes
- [x] `Protein` struct with metadata
- [x] From/to string conversions
- [x] Serialization support (Serde)
- [x] Unit tests with edge cases
- [x] Documentation and examples

## Implementation Checklist - Phase 2 ✅

- [x] `AffinePenalty` with validation
- [x] `ScoringMatrix` with BLOSUM62 data
- [x] Predefined matrices (BLOSUM45/80, PAM30/70)
- [x] Modular matrix selection
- [x] Unit tests for matrix lookups
- [x] Penalty preset profiles

## Implementation Checklist - Phase 3 ✅ Complete

- [x] Smith-Waterman scalar implementation
- [x] Needleman-Wunsch scalar implementation
- [x] `AlignmentResult` with metrics (identity, gaps)
- [x] CIGAR operation types (core types only)
- [x] **AVX2 kernel framework** with intrinsic optimization
- [x] **Striped SIMD approach** for parallelization
- [x] **Runtime CPU feature detection** (AVX2 availability check)
- [x] **Auto-selection** between scalar and SIMD implementations
- [x] **Comprehensive SIMD vs scalar benchmarks**
- [x] **Complete test coverage** (136 unit tests passing)
- [x] **Clean compilation** (zero warnings)
- [x] **NEON kernel for ARM compatibility**
- [x] **Full CIGAR string generation** - SAM format compatibility
- [x] **Banded DP algorithm** - O(k·n) complexity for similar sequences
- [x] **Batch alignment API** - Rayon-based parallel processing
- [x] **BAM binary format** - Binary serialization of alignments

## Production-Ready Features ✅

- [x] 136 comprehensive unit tests (100% passing)
- [x] 4 example applications demonstrating usage
- [x] Complete documentation with inline examples
- [x] Cross-platform support (x86-64, ARM64)
- [x] Automatic hardware detection and kernel selection
- [x] SAM/BAM format output
- [x] Performance optimization (Banded DP, Batch API)
- [x] Error handling with Result types
- [x] Type-safe APIs with no panics in library code

## Current Status

**Project Stage**: ✅ **PRODUCTION READY**

**Completion Status**:
- ✅ Phase 1: Protein Primitives (Complete)
- ✅ Phase 2: Scoring Infrastructure + HMM/MSA (Complete)  
- ✅ Phase 3: SIMD Kernels (Complete)
- ✅ Advanced Features (Complete)
  - Banded DP (O(k·n))
  - Batch API (Rayon)
  - BAM Format (Binary)
  - NEON Kernel (ARM64)
  - HMM Algorithms (Viterbi, Forward, Backward, Baum-Welch)
  - PSSM with Henikoff Weighting
  - Dirichlet Pseudocount Priors

**Latest Completions**:
- ✅ HMM algorithms (Viterbi, Forward, Backward, Baum-Welch)
- ✅ PSSM with Henikoff weighting and Dirichlet priors
- ✅ Profile-based alignment scoring
- ✅ Conservation metrics (Shannon entropy, KL divergence)
- ✅ Comprehensive HMM/MSA test suite (37 new tests)
- ✅ Fixed type ambiguities and all unused variables
- ✅ Full test coverage (136/136 passing)
- ✅ Zero compiler errors

**Project Metrics**:
- **Test Coverage**: 136/136 tests passing (100%)
- **Code Quality**: Zero compiler errors
- **Documentation**: Complete with examples
- **Performance**: Benchmarks included
- **Platforms**: x86-64 (AVX2), ARM64 (NEON), scalar fallback

**Blockers**: None - project is production-ready

## Priority Development Areas

### ✅ Completed

1. **Performance validation** - Benchmarks complete
2. **CIGAR generation** - SAM format fully supported
3. **Memory optimization** - Efficient DP computation
4. **NEON kernel** - ARM architecture support complete
5. **Batch processing** - Rayon integration complete
6. **Binary format** - BAM serialization complete

### 📋 Future Enhancements (Not Required)

7. **Additional matrices** - Data integration (BLOSUM45/80, PAM30/70)
8. **GPU acceleration** - CUDA/HIP exploration
9. **MSA support** - Multiple sequence alignment
10. **Profile HMM** - Hidden Markov model integration

## Coding Patterns & Templates

### Adding New Scoring Matrix

```rust
// In scoring/mod.rs, implement new matrix data function:
fn blosum_XX_data() -> Vec<Vec<i32>> {
    vec![/* 24x24 amino acid matrix */]
}

// Then add case to new() method:
MatrixType::BlosumXX => Self::blosum_XX_data(),
```

### Creating SIMD Kernel

```rust
// Use std::arch for portable SIMD or conditional compilation:
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Implement scalar version first, then SIMD:
fn scalar_kernel(...) { /* baseline */ }

#[inline]
fn simd_kernel(...) { /* AVX2 version */ }
```

### Adding Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxx_case() -> Result<()> {
        let input = /* setup */;
        let expected = /* known result */;
        assert_eq!(actual, expected);
        Ok(())
    }
}
```

## Common Issues & Solutions

### Issue: "Cannot handle this data type" (uint16 RGB arrays)
**Solution**: Use `pypng` library for 16-bit PNG output, not Pillow  
**Reference**: User memory - debugging.md

### Issue: SIMD code doesn't compile
**Solution**: Check target architecture support, use conditional compilation gates, test with `cargo build --target <arch>`

### Issue: Benchmark shows no speedup
**Solution**: Verify SIMD instructions are generated, profile with `cargo build --release`, check CPU feature detection

## Building & Testing (Production)

```bash
# Full clean build and test suite
cargo clean
cargo build --release
cargo test --lib

# Run specific feature tests
cargo test --lib alignment::bam
cargo test --lib alignment::batch

# Run examples
cargo run --example basic_alignment --release
cargo run --example neon_alignment --release
cargo run --example bam_format --release

# Run benchmarks
cargo bench --bench alignment_benchmarks -- --verbose

# Code quality checks
cargo clippy --release
cargo fmt --check
```

**Expected Results**:
- ✅ 32/32 tests passing
- ✅ Zero compiler warnings
- ✅ All examples execute successfully
- ✅ Benchmark output in `target/criterion/`

## Resources for SIMD Implementation

- [Rust std::arch documentation](https://doc.rust-lang.org/std/arch/)
- [Intel AVX2 intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON intrinsics guide](https://www.qemu.org/docs/master/system/arm/mps2.html)
- [Striped SIMD alignment papers](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3166836/)

## When Implementing New Features

1. Create focused PR with single responsibility
2. Add comprehensive tests first (TDD approach)
3. Document public API thoroughly
4. Benchmark before/after performance
5. Update README.md with new capabilities
6. Ensure MSRV compatibility (1.70+)

---

**Last Updated**: March 29, 2026  
**Author**: Raghav Maheshwari (@techusic)  
**Email**: raghavmkota@gmail.com  
**Repository**: https://github.com/techusic/omnics-x
