# OMICS-X: MASTER BUG REGISTRY
**Date**: April 1, 2026  
**Last Updated**: Complete audit cycle  
**Total Bugs Found**: 21  
**Status**: 3 FIXED ✅ | 18 UNFIXED ⏳

---

## Executive Summary

### Critical State Assessment
🔴 **4 CRITICAL** bugs found including:
- HMM transition logic corrupts all algorithms
- GPU memory leak causes device failure  
- BAM integer overflow causes data loss
- CUDA shared memory limits (already fixed ✅)

The codebase has **systemic correctness issues** requiring immediate high-priority fixes across core algorithms and I/O.

---

## Master Bug Registry - All 21 Bugs

### ✅ FIXED (3)

| # | Bug | Severity | File | Status | Date |
|---|-----|----------|------|--------|------|
| 6 | CUDA Dynamic Shared Memory | CRITICAL | cuda_kernels_rtc.rs | ✅ FIXED | 2026-04-01 |
| 7 | Hardcoded Karlin-Altschul | HIGH | hmm.rs | ✅ FIXED | 2026-04-01 |
| 1 | NEON Excessive Allocations | MEDIUM | simd_viterbi.rs | ✅ FIXED | 2026-04-01 |

### ⏳ TODO (18)

| # | Bug | Severity | File | Category | Est. LOC |
|---|-----|----------|------|----------|---------|
| **10** | **HMM Transition Logic Error** | **CRITICAL** | **hmm.rs** | **Logic** | **40-50** |
| **11** | **BAM Integer Overflow** | **CRITICAL** | **bam.rs** | **Type** | **30-40** |
| **12** | **GPU Memory Leak** | **CRITICAL** | **gpu_executor.rs** | **Resource** | **60-80** |
| 2 | GPU Kernel Launch (Viterbi) | HIGH | simd_viterbi.rs | GPU | 40-50 |
| 3 | CUDA smith/needleman CPU | HIGH | kernel/cuda.rs | GPU | 80-100 |
| 5 | Incomplete HMM Transitions | HIGH | hmmer3_parser.rs | Parser | 60-80 |
| 9 | Unsafe Transition Indexing | HIGH | simd_viterbi.rs | Logic | 70-90 |
| 13 | Mutex Poisoning Cascade | HIGH | distributed.rs | Concurrency | 70-90 |
| 14 | Type Underflow | HIGH | mod.rs | Type | 40-50 |
| 15 | String Parsing DoS | HIGH | hmm_multiformat.rs | Parser | 50-70 |
| 16 | Uninitialized DP Vectors | HIGH | hmm.rs | Logic | 30-40 |
| 17 | Missing Bounds Checks | HIGH | simd_viterbi.rs | Bounds | 50-70 |
| 4 | HMMER3 Parsing Fragility | MEDIUM | hmmer3_full_parser.rs | Parser | 50-70 |
| 8 | .gz Compression Support | MEDIUM | cli_file_io.rs | I/O | 40-50 |
| 18 | CIGAR Off-by-One | MEDIUM | mod.rs | Logic | 1 |
| 19 | AA Code Wraparound | MEDIUM | hmm.rs | Logic | 40-50 |
| 20 | expect() Panic on CIGAR | MEDIUM | mod.rs | Error | 20-30 |
| 21 | Race Condition in Future | MEDIUM | hmm.rs | Concurrency | 30-40 |

---

## Severity Breakdown

### 🔴 CRITICAL (4 bugs)
**Define**: System-level data corruption, device failure, complete algorithm breakdown
- Bug #6 ✅ FIXED - CUDA shared memory overflow
- Bug #10 ⏳ TODO - HMM transition logic (all algorithms invalid)
- Bug #11 ⏳ TODO - BAM overflow (data loss)
- Bug #12 ⏳ TODO - GPU memory leak (device exhaustion)

### 🟠 HIGH (10 bugs)
**Define**: Algorithm incorrectness, crashes, dropped results
- Bugs #2, #3, #5, #9 - GPU acceleration + algorithm defects (original audit)
- Bugs #13-17 - Concurrency, bounds, uninitialized memory (new audit)

### 🟡 MEDIUM (7 bugs)
**Define**: Edge cases, corner cases, robustness
- Bugs #1, #4, #8 - Performance + robustness (original audit)
- Bugs #18-21 - Off-by-one, parsing, error handling (new audit)

---

## Implementation Roadmap - Prioritized by Impact

### PHASE 0: CRITICAL FIX (IMMEDIATE - < 30 min)
**Impact**: Fix algorithms, prevent data corruption

- [ ] **Bug #10** - HMM Transition Indices (40-50 LOC) ⚠️ **HIGHEST PRIORITY**
  - Affects: Forward, Viterbi, Backward, Baum-Welch algorithms
  - Impact: All HMM results currently invalid
  - Estimated: 30 min
  
- [ ] **Bug #18** - CIGAR Off-by-One (1 LOC) ✓ Quick win
  - Affects: Alignment output format
  - Estimated: 2 min

### PHASE 1: CRITICAL FIXES (1-2 hours)

- [ ] **Bug #11** - BAM Integer Overflow (30-40 LOC)
  - Use u32 instead of i32, validate sizes
  - Prevents data corruption
  
- [ ] **Bug #12** - GPU Memory Leak (60-80 LOC)
  - Add RAII scope guard for GPU buffers
  - Prevents device exhaustion
  
- [ ] **Bug #14** - Type Underflow (40-50 LOC)
  - Use saturating_sub, valid checks
  - Prevents score inversion

- [ ] **Bug #13** - Mutex Poisoning (70-90 LOC)
  - Handle poisoned mutexes gracefully
  - Prevents cascade failures

### PHASE 2: HIGH PRIORITY FIXES (2-3 hours)

- [ ] **Bug #15** - String Parsing DoS (50-70 LOC)
  - Defensive parsing with proper error handling
  - Prevents crashes on malformed input

- [ ] **Bug #16** - Uninitialized Vectors (30-40 LOC)
  - Explicit initialization before use
  - Prevents undefined behavior

- [ ] **Bug #17** - Bounds Checking (50-70 LOC)
  - Add validation before array access
  - Prevents panics

- [ ] **Bug #2, #3** - GPU Kernel Launches (120-150 LOC)
  - Implement actual GPU computation
  - Enables acceleration

- [ ] **Bug #5, #9** - HMM Transitions Parsing (130-170 LOC)
  - Complete 7-transition parsing
  - Safe transition indexing

### PHASE 3: MEDIUM PRIORITY (1-2 hours)

- [ ] **Bug #4** - HMMER3 Robust Parsing (50-70 LOC)
  - State machine parser
  - Robustness improvement

- [ ] **Bug #8** - .gz Compression (40-50 LOC)
  - Auto-detect .gz format
  - UX improvement

- [ ] **Bug #19** - AA Code Validation (40-50 LOC)
  - Enum-based validation
  - Prevent silent corruption

- [ ] **Bug #20** - expect() Error Handling (20-30 LOC)
  - Replace panics with proper errors
  - Better debugging

- [ ] **Bug #21** - Race Condition (30-40 LOC)
  - Arc<Mutex<>> wrapping
  - Deterministic results

---

## Dependency Graph

```
Bug #10 (HMM Logic) ──┐
                      ├─→ Bug #9 (Transition Indexing)
                      │
Bug #5 (Transitions) ─┘

Bug #12 (GPU Leak) ──┐
                     ├─→ Bug #2 (Viterbi GPU)
Bug #3 (CUDA SW/NW)─┘

Bug #14 (Type) ──┐
                 └─→ GAP cost scoring
Bug #11 (BAM) ──┐
                └─→ Large genome support

Independent bugs: #1-4, #8, #13, #15-21
```

---

## Risk Assessment Matrix

| Bug | Impact | Likelihood | Risk | Action |
|-----|--------|-----------|------|--------|
| #10 | Very High | Very High | 🔴 CRITICAL | Fix immediately |
| #12 | Very High | Medium | 🔴 CRITICAL | Fix immediately |
| #11 | High | Medium | 🟠 HIGH | Fix ASAP |
| #2,#3 | High | Low | 🟠 HIGH | Fix soon |
| #14 | High | Low | 🟠 HIGH | Fix soon |
| #15 | Medium | Low | 🟡 MEDIUM | Fix eventually |
| Others | Low-Med | Low | 🟡 MEDIUM | Fix as time permits |

---

## Testing Guide

### For Each Bug Fix, Add:

1. **Unit Test** - Direct test of fixed function
2. **Regression Test** - Ensure fix doesn't break other code
3. **Integration Test** - End-to-end workflow with fix
4. **Stress Test** - Large inputs, edge cases

Example for Bug #10 (HMM Transitions):
```rust
#[test]
fn test_hm_transitions_indices_correct() {
    // Each transition must use distinct indices
    // Forward/Viterbi/Backward produce same results on known input
    // Baum-Welch converges to reference parameters
    let hmm = load_reference_hmm("PFAM_reference.hmm");
    let sequence = b"MKVLVIGDRPG...";
    
    let forward_score = hmm.forward_score(sequence);
    let viterbi_path = hmm.viterbi(sequence);
    
    // Validate against known reference results
    assert!((forward_score - REFERENCE_SCORE).abs() < 0.01);
    assert_eq!(viterbi_path.cigar, REFERENCE_CIGAR);
}
```

---

## Metrics

### Before Fixes (Current State)
- ✅ Compiles: Yes
- ✅ Tests Pass: 275/275 (100%)
- ❌ Correctness: ⚠️ Algorithm results invalid (Bug #10)
- ❌ Stability: ⚠️ GPU crash risk, memory leak (Bug #12)
- ❌ Data Integrity: ⚠️ Overflow, wraparound (Bugs #11, #19)

### After All Fixes (Target State)
- ✅ Compiles: Yes (no breaking changes)
- ✅ Tests Pass: 300+/300+ (100%+, new tests added)
- ✅ Correctness: All algorithms validated
- ✅ Stability: No panics, proper error handling
- ✅ Data Integrity: Safe type conversions, bounds checked

---

## Estimated Effort Summary

| Phase | Bugs | Total LOC | Estimated Time |
|-------|------|-----------|-----------------|
| Phase 0 | 2 | 41-51 | 0.5 hours |
| Phase 1 | 5 | 220-280 | 1.5 hours |
| Phase 2 | 5 | 380-520 | 2.5 hours |
| Phase 3 | 6 | 270-380 | 2 hours |
| **TOTAL** | **18** | **911-1231** | **6.5 hours** |

Combined with already-fixed bugs: **12-16 hours total project work**

---

## Recommendations

### 1. IMMEDIATE ACTION (Next 30 minutes)
Fix Bug #10 and #18 - these have highest impact/effort ratio

### 2. TRIAGE (Next 2 hours)
Fix Critical and High-severity bugs in Phase 0-2

### 3. SUSTAINED (Next 4-6 hours)
Complete all Medium-severity bugs in Phase 3

### 4. VALIDATION (Ongoing)
- Run full test suite after each phase
- Add regression tests for each fix
- Performance benchmark before/after

### 5. DOCUMENTATION
- Document why each bug occurred
- Add safeguards to prevent recurrence
- Update developer guidelines

---

## Version Planning

- **v1.1.0-critical-fixes** - Current (Bugs #1, #6, #7)
- **v1.1.1-correctness** - Phase 0-1 (Bugs #10-14, #18)
- **v1.1.2-stability** - Phase 2 (Bugs #2, #3, #5, #9, #15-17)
- **v1.2.0** - Phase 3 complete (All bugs fixed + new features)

---

**Project Status**: ⚠️ **CRITICAL** - Immediate action required

**Recommendation**: Halt feature development, prioritize bug fixes for correctness and stability.

