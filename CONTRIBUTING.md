# Contributing to omicsx

**Maintained by**: Raghav Maheshwari (@techusic)  
**Email**: raghavmkota@gmail.com  
**Repository**: https://github.com/techusic/omicsx

Thank you for your interest in contributing to omicsx! This document provides guidelines and procedures for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. Fork the repository: https://github.com/techusic/omicsx/fork
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/omicsx.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Set up your environment: `cargo build --release`

## Development Guidelines

### Rust Standards

- **Edition**: Rust 2021
- **MSRV**: 1.70+
- **Format**: Run `cargo fmt` before committing
- **Lint**: Pass `cargo clippy --release`
- **Tests**: All tests must pass: `cargo test --lib`

### Code Quality

- All public APIs must have documentation with examples
- Unit tests required for new functionality
- Integration tests for end-to-end workflows
- No `unsafe` code except in SIMD kernel modules with clear comments
- No panics in library code (only in tests/examples)

### Commit Messages

Use conventional commits format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Example:
```
feat(alignment): add banded DP optimization

Implements O(k·n) complexity algorithm for similar sequences
using diagonal band restriction.

Closes #42
```

## Making Changes

### Phase 1: Protein Primitives
- File: `src/protein/mod.rs`
- Add tests in `mod.rs` tests section
- Update README.md if adding public APIs

### Phase 2: Scoring Infrastructure
- File: `src/scoring/mod.rs`
- Update matrix data in dedicated functions
- Test with known reference values

### Phase 3: SIMD Kernels
- Files: `src/alignment/kernel/*.rs`
- Implement scalar version first
- Profile before optimizing with SIMD
- Benchmark with Criterion.rs

### Future Enhancements
- Files: `src/futures/*.rs`
- Each module is self-contained
- Replace `todo!()` macros with implementations
- Update `#[ignore]` tests to be active
- Maintain error type hierarchy

## Testing

```bash
# Run all tests
cargo test --lib

# Run specific test
cargo test --lib path::to::test

# Run with output
cargo test --lib -- --nocapture

# Benchmark
cargo bench --bench alignment_benchmarks -- --verbose

# Check with Clippy
cargo clippy --release

# Format check
cargo fmt --check
```

## Performance Considerations

1. Profile before optimizing
2. SIMD optimization only after scalar correctness
3. Benchmark against baseline implementations
4. Document performance characteristics
5. Test on multiple architectures (x86-64, ARM64)

## Documentation

- Add doc comments to all public items
- Include examples in public API docs
- Update README.md for major features
- Add inline comments for complex algorithms
- Use /// for documentation, // for implementation notes

## Pull Request Process

1. Update tests and documentation
2. Ensure all tests pass: `cargo test --lib`
3. Ensure no warnings: `cargo clippy --release`
4. Ensure formatted: `cargo fmt`
5. Squash related commits
6. Write clear PR description

## Architecture

```
src/
├── protein/          # Phase 1: Types and validation
├── scoring/          # Phase 2: Matrices and penalties
├── alignment/        # Phase 3: SIMD kernels
│   ├── kernel/      # Scalar, AVX2, NEON implementations
│   ├── batch.rs     # Parallel processing
│   ├── bam.rs       # Binary format
│   └── mod.rs       # Integration
└── futures/         # Future enhancements
    ├── matrices.rs  # Additional scoring matrices
    ├── formats.rs   # BLAST/GFF3 export
    ├── gpu.rs       # GPU acceleration
    ├── msa.rs       # Multiple sequence alignment
    ├── hmm.rs       # Profile HMM
    └── phylogeny.rs # Phylogenetic trees
```

## Architecture Decisions

- **Type Safety**: Use enums instead of flags/raw values
- **Error Handling**: `Result<T>` for all fallible operations
- **Memory Safety**: Leverage Rust's ownership model
- **Performance**: SIMD where measurable benefit exists
- **Portability**: Support both x86-64 and ARM64

## Issues and Features

- Bug reports: Use GitHub Issues with reproducible example
- Feature requests: Discuss design before implementation
- Documentation: PRs for typos and clarifications welcome

## Licensing

By contributing to omicsx, you agree that your contributions will be made available under:
- **MIT License** for non-commercial use
- **Dual commercial license** as part of the project's commercial licensing model

Your contributions may be used in both open-source and commercial contexts.

## Questions?

- Read the README.md for usage examples
- Check existing tests for patterns
- Review the copilot-instructions.md for architecture notes
- Open a discussion issue for architectural questions

Thank you for contributing! 🧬
