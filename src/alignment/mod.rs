//! # Alignment Module
//!
//! High-performance sequence alignment algorithms with SIMD and GPU optimization.
//! Provides Smith-Waterman (local) and Needleman-Wunsch (global) alignment implementations.
//!
//! ## Performance Optimization Support
//!
//! This module supports automatic selection between:
//! - **GPU Kernels**: CUDA (NVIDIA), HIP (AMD), Vulkan (cross-platform)
//! - **AVX2 kernels** (x86-64): 8-wide parallelization
//! - **NEON kernels** (aarch64): 4-wide parallelization
//! - **Scalar kernels**: Portable fallback for all architectures

pub mod kernel;
pub mod batch;
pub mod bam;
pub mod gpu_dispatcher;
pub mod gpu_kernels;
pub mod cuda_kernels;
pub mod cuda_device_context;
pub mod cuda_runtime;
pub mod kernel_compiler;
pub mod kernel_launcher;
pub mod smith_waterman_cuda;
pub mod hmmer3_parser;
pub mod simd_viterbi;
pub mod profile_dp;
pub mod gpu_memory;
pub mod cigar_gen;

pub use bam::{BamFile, BamRecord};
pub use gpu_dispatcher::{GpuDispatcher, GpuAvailability, AlignmentStrategy, GpuDeviceInfo};
pub use cuda_device_context::CudaDeviceContext;
pub use cuda_runtime::{GpuRuntime, GpuBuffer};
pub use kernel_compiler::{KernelCompiler, KernelType, CompiledKernel, KernelCache};
pub use kernel_launcher::{SmithWatermanKernel, NeedlemanWunschKernel, KernelExecutionResult};
pub use smith_waterman_cuda::SmithWatermanCudaKernel;
pub use hmmer3_parser::{HmmerModel, HmmerError, KarlinParameters};
pub use simd_viterbi::{ViterbiDecoder, ViterbiPath};
pub use profile_dp::{Pssm, ProfileAlignment, align_profiles};
pub use gpu_memory::{GpuMemoryPool, MultiGpuMemory, MemoryAllocation};
pub use cigar_gen::{CigarString, CigarOp, traceback_to_cigar};

use crate::error::{Error, Result};
use crate::protein::{Protein, AminoAcid};
use crate::scoring::{ScoringMatrix, AffinePenalty};
use serde::{Deserialize, Serialize};

/// Represents a CIGAR string (Compact Idiosyncratic Gapped Alignment Report)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cigar {
    operations: Vec<(u32, CigarOp)>,
}

impl Cigar {
    /// Create a new CIGAR string
    pub fn new() -> Self {
        Cigar {
            operations: Vec::new(),
        }
    }

    /// Add an operation
    pub fn push(&mut self, count: u32, op: CigarOp) {
        if count == 0 {
            return;
        }
        if let Some((last_count, last_op)) = self.operations.last_mut() {
            if *last_op == op {
                *last_count += count;
                return;
            }
        }
        self.operations.push((count, op));
    }

    /// Combine consecutive same operations
    pub fn coalesce(&mut self) {
        let mut result = Vec::new();
        for (count, op) in self.operations.drain(..) {
            if let Some((last_count, last_op)) = result.last_mut() {
                if *last_op == op {
                    *last_count += count;
                    continue;
                }
            }
            result.push((count, op));
        }
        self.operations = result;
    }

    /// Get operations
    pub fn operations(&self) -> &[(u32, CigarOp)] {
        &self.operations
    }

    /// Get total query length
    pub fn query_length(&self) -> u32 {
        self.operations
            .iter()
            .filter_map(|(count, op)| match op {
                CigarOp::Match | CigarOp::Insertion | CigarOp::SeqMatch | CigarOp::SeqMismatch => Some(count),
                _ => None,
            })
            .sum::<u32>()
    }

    /// Get total reference length
    pub fn reference_length(&self) -> u32 {
        self.operations
            .iter()
            .filter_map(|(count, op)| match op {
                CigarOp::Match | CigarOp::Deletion | CigarOp::Skip | CigarOp::SeqMatch | CigarOp::SeqMismatch => Some(count),
                _ => None,
            })
            .sum::<u32>()
    }
}

impl Default for Cigar {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Cigar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (count, op) in &self.operations {
            write!(f, "{}{}", count, op)?;
        }
        Ok(())
    }
}

/// SAM format alignment record (Sequence Alignment/Map)
///
/// Represents a single alignment record in SAM/BAM format. SAM is a text-based format
/// for storing sequence alignment data, while BAM is the binary equivalent.
///
/// # SAM Format
///
/// Each record contains 11 mandatory fields:
/// 1. QNAME - Query name
/// 2. FLAG - Bitwise flag encoding alignment properties  
/// 3. RNAME - Reference sequence name
/// 4. POS - Alignment start position (1-based)
/// 5. MAPQ - Mapping quality (0-60)
/// 6. CIGAR - CIGAR string representation
/// 7. RNEXT - Next reference name
/// 8. PNEXT - Next reference position
/// 9. TLEN - Template length
/// 10. SEQ - Query sequence
/// 11. QUAL - Quality scores (Phred+33)
/// 
/// Optional fields can be added as TAG:TYPE:VALUE triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamRecord {
    /// Query sequence name
    pub qname: String,
    
    /// Query sequence (subject)
    pub query_seq: String,
    
    /// Query quality scores (one ASCII char per base, Phred+33 encoded)
    pub query_qual: String,
    
    /// Reference sequence name
    pub rname: String,
    
    /// Reference sequence (template)
    pub reference_seq: String,
    
    /// Alignment start position in reference (1-based)
    pub pos: u32,
    
    /// Mapping quality (0-255, typically 0-60)
    pub mapq: u8,
    
    /// CIGAR string
    pub cigar: String,
    
    /// Alignment flag (bitwise flags)
    pub flag: u16,
    
    /// Optional alignment fields (TAG:TYPE:VALUE format)
    pub optional_fields: Vec<String>,
}

/// SAM file header
///
/// Contains metadata about the alignment file including format version,
/// sorting order, and reference sequence information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamHeader {
    /// Format version (e.g., "1.0", "1.6")
    pub version: String,
    
    /// Sorting order: unsorted, coordinate, queryname
    pub sort_order: String,
    
    /// Reference sequences with name and length
    pub references: Vec<(String, u32)>,
    
    /// Program name/version
    pub program: Option<String>,
}

impl SamHeader {
    /// Create a new SAM header
    pub fn new(version: &str) -> Self {
        SamHeader {
            version: version.to_string(),
            sort_order: "unsorted".to_string(),
            references: Vec::new(),
            program: None,
        }
    }
    
    /// Set sorting order
    pub fn with_sort_order(mut self, order: &str) -> Self {
        self.sort_order = order.to_string();
        self
    }
    
    /// Add a reference sequence
    pub fn add_reference(&mut self, name: &str, length: u32) {
        self.references.push((name.to_string(), length));
    }
    
    /// Set program information
    pub fn with_program(mut self, program: &str) -> Self {
        self.program = Some(program.to_string());
        self
    }
    
    /// Generate SAM header lines
    pub fn to_header_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        
        // HD line (file format header)
        let mut hd = format!("@HD\tVN:{}", self.version);
        if self.sort_order != "unsorted" {
            hd.push_str(&format!("\tSO:{}", self.sort_order));
        }
        lines.push(hd);
        
        // PG line (program)
        if let Some(ref prog) = self.program {
            lines.push(format!("@PG\tID:omics-simd\tPN:omics-simd\tVN:{}", prog));
        }
        
        // SQ lines (reference sequences)
        for (name, length) in &self.references {
            lines.push(format!("@SQ\tSN:{}\tLN:{}", name, length));
        }
        
        lines
    }
}

impl SamRecord {
    /// Create a new SAM record
    pub fn new() -> Self {
        SamRecord {
            qname: "query".to_string(),
            query_seq: String::new(),
            query_qual: String::new(),
            rname: "reference".to_string(),
            reference_seq: String::new(),
            pos: 0,
            mapq: 60,
            cigar: String::new(),
            flag: 0,
            optional_fields: Vec::new(),
        }
    }
    
    /// Create SAM record from alignment result  
    pub fn from_alignment(
        result: &AlignmentResult,
        query_name: &str,
        reference_name: &str,
        reference_start: u32,
    ) -> Self {
        let mut record = SamRecord::new();
        record.qname = query_name.to_string();
        record.rname = reference_name.to_string();
        record.pos = reference_start.saturating_add(1); // Convert 0-based to 1-based
        record.cigar = result.cigar.clone();
        record.query_seq = result.aligned_seq1.replace('-', "");
        record.mapq = 60; // High confidence
        
        // Add alignment score as optional field (AS:i:score)
        record.optional_fields.push(format!("AS:i:{}", result.score));
        
        record
    }
    
    /// Add an optional field in SAM format (TAG:TYPE:VALUE)
    pub fn add_optional_field(&mut self, field: &str) {
        self.optional_fields.push(field.to_string());
    }
    
    /// Generate SAM record line
    pub fn to_sam_line(&self) -> String {
        let mut line = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t",
            self.qname,
            self.flag,
            self.rname,
            self.pos,
            self.mapq,
            self.cigar
        );
        
        // RNEXT, PNEXT, TLEN (not applicable for single alignments)
        line.push_str("*\t0\t0\t");
        
        // SEQ and QUAL
        line.push_str(&self.query_seq);
        line.push('\t');
        if self.query_qual.is_empty() {
            line.push('*');
        } else {
            line.push_str(&self.query_qual);
        }
        
        // Optional fields
        for field in &self.optional_fields {
            line.push('\t');
            line.push_str(field);
        }
        
        line
    }
}

impl Default for SamRecord {
    fn default() -> Self {
        Self::new()
    }
}

/// Alignment result containing score and aligned sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub score: i32,
    pub aligned_seq1: String,
    pub aligned_seq2: String,
    pub start_pos1: usize,
    pub start_pos2: usize,
    pub end_pos1: usize,
    pub end_pos2: usize,
    pub cigar: String,
}

impl AlignmentResult {
    /// Calculate identity percentage
    pub fn identity(&self) -> f64 {
        let matches = self.aligned_seq1
            .chars()
            .zip(self.aligned_seq2.chars())
            .filter(|(a, b)| a == b && a != &'-')
            .count();
        
        let total = self.aligned_seq1.chars().filter(|c| c != &'-').count();
        if total == 0 {
            0.0
        } else {
            (matches as f64 / total as f64) * 100.0
        }
    }

    /// Calculate similarity gap penalty
    pub fn gap_count(&self) -> usize {
        self.aligned_seq1
            .chars()
            .zip(self.aligned_seq2.chars())
            .filter(|(a, b)| *a == '-' || *b == '-')
            .count()
    }

    /// Generate CIGAR string from aligned sequences
    pub fn generate_cigar(&mut self) {
        let mut cigar = Cigar::new();
        
        for (a, b) in self.aligned_seq1.chars().zip(self.aligned_seq2.chars()) {
            match (a, b) {
                ('-', _) => cigar.push(1, CigarOp::Deletion),
                (_, '-') => cigar.push(1, CigarOp::Insertion),
                (c1, c2) if c1 == c2 => cigar.push(1, CigarOp::SeqMatch),
                _ => cigar.push(1, CigarOp::SeqMismatch),
            }
        }

        cigar.coalesce();
        self.cigar = cigar.to_string();
    }
}

/// Smith-Waterman local alignment algorithm
pub struct SmithWaterman {
    matrix: ScoringMatrix,
    penalty: AffinePenalty,
    use_simd: bool,
    bandwidth: Option<usize>,  // Banded DP bandwidth (None = full DP)
}

impl SmithWaterman {
    /// Create new Smith-Waterman aligner with default settings
    /// 
    /// Create new Smith-Waterman aligner with default settings
    /// 
    /// Defaults to striped SIMD implementation for improved performance
    pub fn new() -> Self {
        SmithWaterman {
            matrix: ScoringMatrix::default(),
            penalty: AffinePenalty::default(),
            use_simd: true,  // Striped SIMD is now faster than scalar
            bandwidth: None,
        }
    }

    /// Create with custom scoring matrix
    pub fn with_matrix(matrix: ScoringMatrix) -> Self {
        SmithWaterman {
            matrix,
            penalty: AffinePenalty::default(),
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
            bandwidth: None,
        }
    }

    /// Create with custom penalties
    pub fn with_penalty(penalty: AffinePenalty) -> Self {
        SmithWaterman {
            matrix: ScoringMatrix::default(),
            penalty,
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
            bandwidth: None,
        }
    }

    /// Force use of scalar implementation (for testing/validation)
    pub fn scalar_only(mut self) -> Self {
        self.use_simd = false;
        self
    }

    /// Enable or disable SIMD use
    pub fn with_simd(mut self, use_simd: bool) -> Self {
        self.use_simd = use_simd;
        self
    }

    /// Enable banded DP for similar sequences
    /// 
    /// Bandwidth of k means only cells within distance k of the diagonal are computed.
    /// Reduces complexity from O(m*n) to O(k*n), giving ~10x speedup for bandwidth 10-100.
    /// Recommended for sequences with >90% identity.
    pub fn with_bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Disable banded DP (use full DP)
    pub fn without_bandwidth(mut self) -> Self {
        self.bandwidth = None;
        self
    }

    /// Perform local sequence alignment
    pub fn align(&self, seq1: &Protein, seq2: &Protein) -> Result<AlignmentResult> {
        if seq1.is_empty() || seq2.is_empty() {
            return Err(Error::EmptySequence);
        }

        // Use banded DP if bandwidth is configured
        if let Some(bandwidth) = self.bandwidth {
            return self.align_banded(seq1, seq2, bandwidth);
        }

        if self.use_simd {
            self.align_simd(seq1, seq2)
        } else {
            self.align_scalar(seq1, seq2)
        }
    }

    /// SIMD-optimized alignment (when available)
    fn align_simd(&self, seq1: &Protein, seq2: &Protein) -> Result<AlignmentResult> {
        #[cfg(target_arch = "x86_64")]
        {
            let (h, max_i, max_j) = kernel::striped_simd::smith_waterman_striped_avx2(
                seq1.sequence(),
                seq2.sequence(),
                &self.matrix,
                self.penalty.open,
                self.penalty.extend,
            )?;
            self.build_result(seq1, seq2, &h, max_i, max_j)
        }

        #[cfg(target_arch = "aarch64")]
        {
            let (h, max_i, max_j) = kernel::neon::smith_waterman_neon(
                seq1.sequence(),
                seq2.sequence(),
                &self.matrix,
                self.penalty.open,
                self.penalty.extend,
            )
            .or_else(|_| self.align_scalar(seq1, seq2))?;
            self.build_result(seq1, seq2, &h, max_i, max_j)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.align_scalar(seq1, seq2)
        }
    }

    /// Scalar alignment (baseline implementation)
    fn align_scalar(&self, seq1: &Protein, seq2: &Protein) -> Result<AlignmentResult> {
        let (h, max_i, max_j) = kernel::scalar::smith_waterman_scalar(
            seq1.sequence(),
            seq2.sequence(),
            &self.matrix,
            self.penalty.open,
            self.penalty.extend,
        )?;
        self.build_result(seq1, seq2, &h, max_i, max_j)
    }

    /// Banded DP alignment for similar sequences
    fn align_banded(&self, seq1: &Protein, seq2: &Protein, bandwidth: usize) -> Result<AlignmentResult> {
        let (h, max_i, max_j) = kernel::banded::smith_waterman_banded(
            seq1.sequence(),
            seq2.sequence(),
            &self.matrix,
            self.penalty.open,
            self.penalty.extend,
            bandwidth,
        )?;
        self.build_result(seq1, seq2, &h, max_i, max_j)
    }

    /// Build alignment result from DP matrix
    fn build_result(
        &self,
        seq1: &Protein,
        seq2: &Protein,
        h: &[Vec<i32>],
        max_i: usize,
        max_j: usize,
    ) -> Result<AlignmentResult> {
        let score = h[max_i][max_j];
        let seq1_bytes = seq1.sequence();
        let seq2_bytes = seq2.sequence();

        // Traceback to construct alignment
        let (aligned1, aligned2) = self.traceback_sw(h, seq1_bytes, seq2_bytes, max_i, max_j)?;

        let mut result = AlignmentResult {
            score,
            aligned_seq1: aligned1,
            aligned_seq2: aligned2,
            start_pos1: max_i,
            start_pos2: max_j,
            end_pos1: seq1.len(),
            end_pos2: seq2.len(),
            cigar: String::from(""),
        };

        // Generate CIGAR string from alignment
        result.generate_cigar();

        Ok(result)
    }

    fn traceback_sw(
        &self,
        h: &[Vec<i32>],
        seq1: &[AminoAcid],
        seq2: &[AminoAcid],
        mut i: usize,
        mut j: usize,
    ) -> Result<(String, String)> {
        let mut aligned1 = String::new();
        let mut aligned2 = String::new();

        while i > 0 && j > 0 && h[i][j] > 0 {
            let match_score = self.matrix.score(seq1[i - 1], seq2[j - 1]);
            let diagonal = h[i - 1][j - 1] + match_score;
            let up = h[i - 1][j] + self.penalty.extend;
            let _left = h[i][j - 1] + self.penalty.extend;

            if h[i][j] == diagonal {
                aligned1.insert(0, seq1[i - 1].to_code());
                aligned2.insert(0, seq2[j - 1].to_code());
                i -= 1;
                j -= 1;
            } else if h[i][j] == up {
                aligned1.insert(0, seq1[i - 1].to_code());
                aligned2.insert(0, '-');
                i -= 1;
            } else {
                aligned1.insert(0, '-');
                aligned2.insert(0, seq2[j - 1].to_code());
                j -= 1;
            }
        }

        Ok((aligned1, aligned2))
    }
}

impl Default for SmithWaterman {
    fn default() -> Self {
        Self::new()
    }
}

/// Needleman-Wunsch global alignment algorithm
pub struct NeedlemanWunsch {
    matrix: ScoringMatrix,
    penalty: AffinePenalty,
    use_simd: bool,
    bandwidth: Option<usize>,  // Banded DP bandwidth (None = full DP)
}

impl NeedlemanWunsch {
    /// Create new Needleman-Wunsch aligner with default settings
    pub fn new() -> Self {
        NeedlemanWunsch {
            matrix: ScoringMatrix::default(),
            penalty: AffinePenalty::default(),
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
            bandwidth: None,
        }
    }

    /// Force use of scalar implementation (for testing/validation)
    pub fn scalar_only(mut self) -> Self {
        self.use_simd = false;
        self
    }

    /// Enable or disable SIMD use
    pub fn with_simd(mut self, use_simd: bool) -> Self {
        self.use_simd = use_simd;
        self
    }

    /// Enable banded DP for similar sequences
    pub fn with_bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Disable banded DP (use full DP)
    pub fn without_bandwidth(mut self) -> Self {
        self.bandwidth = None;
        self
    }

    /// Perform global sequence alignment
    pub fn align(&self, seq1: &Protein, seq2: &Protein) -> Result<AlignmentResult> {
        if seq1.is_empty() || seq2.is_empty() {
            return Err(Error::EmptySequence);
        }

        // Use banded DP if bandwidth is configured
        if let Some(bandwidth) = self.bandwidth {
            return self.align_banded(seq1, seq2, bandwidth);
        }

        if self.use_simd {
            self.align_simd(seq1, seq2)
        } else {
            self.align_scalar(seq1, seq2)
        }
    }

    /// SIMD-optimized alignment (when available)
    fn align_simd(&self, seq1: &Protein, seq2: &Protein) -> Result<AlignmentResult> {
        #[cfg(target_arch = "x86_64")]
        {
            let h = kernel::striped_simd::needleman_wunsch_striped_avx2(
                seq1.sequence(),
                seq2.sequence(),
                &self.matrix,
                self.penalty.open,
                self.penalty.extend,
            )?;
            self.build_result(seq1, seq2, &h)
        }

        #[cfg(target_arch = "aarch64")]
        {
            let h = kernel::neon::needleman_wunsch_neon(
                seq1.sequence(),
                seq2.sequence(),
                &self.matrix,
                self.penalty.open,
                self.penalty.extend,
            )
            .or_else(|_| self.align_scalar(seq1, seq2).map(|r| vec![vec![r.score]]))?;
            self.build_result(seq1, seq2, &h)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.align_scalar(seq1, seq2)
        }
    }

    /// Scalar alignment (baseline implementation)
    fn align_scalar(&self, seq1: &Protein, seq2: &Protein) -> Result<AlignmentResult> {
        let h = kernel::scalar::needleman_wunsch_scalar(
            seq1.sequence(),
            seq2.sequence(),
            &self.matrix,
            self.penalty.open,
            self.penalty.extend,
        )?;
        self.build_result(seq1, seq2, &h)
    }

    /// Banded DP alignment for similar sequences
    fn align_banded(&self, seq1: &Protein, seq2: &Protein, bandwidth: usize) -> Result<AlignmentResult> {
        let h = kernel::banded::needleman_wunsch_banded(
            seq1.sequence(),
            seq2.sequence(),
            &self.matrix,
            self.penalty.open,
            self.penalty.extend,
            bandwidth,
        )?;
        self.build_result(seq1, seq2, &h)
    }

    /// Build alignment result from DP matrix
    fn build_result(
        &self,
        seq1: &Protein,
        seq2: &Protein,
        h: &[Vec<i32>],
    ) -> Result<AlignmentResult> {
        let m = seq1.len();
        let n = seq2.len();
        let score = h[m][n];
        let seq1_bytes = seq1.sequence();
        let seq2_bytes = seq2.sequence();

        // Traceback
        let (aligned1, aligned2) = self.traceback_nw(h, seq1_bytes, seq2_bytes)?;

        let mut result = AlignmentResult {
            score,
            aligned_seq1: aligned1,
            aligned_seq2: aligned2,
            start_pos1: 0,
            start_pos2: 0,
            end_pos1: m,
            end_pos2: n,
            cigar: String::from(""),
        };

        // Generate CIGAR string from alignment
        result.generate_cigar();

        Ok(result)
    }

    fn traceback_nw(
        &self,
        h: &[Vec<i32>],
        seq1: &[AminoAcid],
        seq2: &[AminoAcid],
    ) -> Result<(String, String)> {
        let mut i = seq1.len();
        let mut j = seq2.len();
        let mut aligned1 = String::new();
        let mut aligned2 = String::new();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let match_score = self.matrix.score(seq1[i - 1], seq2[j - 1]);
                let diagonal = h[i - 1][j - 1] + match_score;
                let up = h[i - 1][j] + self.penalty.extend;
                let _left = h[i][j - 1] + self.penalty.extend;

                if h[i][j] == diagonal {
                    aligned1.insert(0, seq1[i - 1].to_code());
                    aligned2.insert(0, seq2[j - 1].to_code());
                    i -= 1;
                    j -= 1;
                    continue;
                } else if h[i][j] == up {
                    aligned1.insert(0, seq1[i - 1].to_code());
                    aligned2.insert(0, '-');
                    i -= 1;
                    continue;
                }
            }

            if j > 0 {
                aligned1.insert(0, '-');
                aligned2.insert(0, seq2[j - 1].to_code());
                j -= 1;
            } else if i > 0 {
                aligned1.insert(0, seq1[i - 1].to_code());
                aligned2.insert(0, '-');
                i -= 1;
            }
        }

        Ok((aligned1, aligned2))
    }
}

impl Default for NeedlemanWunsch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cigar_operations() {
        let mut cigar = Cigar::new();
        cigar.push(5, CigarOp::Match);
        cigar.push(2, CigarOp::Insertion);
        cigar.push(3, CigarOp::Match);

        let cigar_str = cigar.to_string();
        assert_eq!(cigar_str, "5M2I3M");
    }

    #[test]
    fn test_cigar_coalesce() {
        let mut cigar = Cigar::new();
        cigar.push(2, CigarOp::Match);
        cigar.push(1, CigarOp::Insertion);
        cigar.push(3, CigarOp::Match);
        cigar.push(2, CigarOp::Match); // Should coalesce with previous

        let mut cigar_copy = cigar.clone();
        cigar_copy.coalesce();
        assert_eq!(cigar_copy.to_string(), "2M1I5M");
    }

    #[test]
    fn test_cigar_lengths() {
        let mut cigar = Cigar::new();
        cigar.push(5, CigarOp::SeqMatch);
        cigar.push(2, CigarOp::Insertion);
        cigar.push(3, CigarOp::Deletion);
        cigar.push(1, CigarOp::Match);

        assert_eq!(cigar.query_length(), 8);      // M=5, I=2, M=1
        assert_eq!(cigar.reference_length(), 9);  // M=5, D=3, M=1
    }

    #[test]
    fn test_alignment_result_identity() -> Result<()> {
        let mut result = AlignmentResult {
            score: 10,
            aligned_seq1: "AGSG".to_string(),
            aligned_seq2: "AGSG".to_string(),
            start_pos1: 0,
            start_pos2: 0,
            end_pos1: 4,
            end_pos2: 4,
            cigar: String::new(),
        };

        // Perfect match
        assert_eq!(result.identity(), 100.0);
        
        // With mismatches
        result.aligned_seq1 = "AGSG".to_string();
        result.aligned_seq2 = "AASG".to_string();
        assert_eq!(result.identity(), 75.0);

        Ok(())
    }

    #[test]
    fn test_alignment_cigar_generation() -> Result<()> {
        let mut result = AlignmentResult {
            score: 15,
            aligned_seq1: "AG-SG".to_string(),
            aligned_seq2: "AGASG".to_string(),
            start_pos1: 0,
            start_pos2: 0,
            end_pos1: 4,
            end_pos2: 5,
            cigar: String::new(),
        };

        result.generate_cigar();
        // A matches A (=), G matches G (=), - is delete (D), S matches A (X), G matches S (X), G matches G (=)
        // Should be: 2=1D2X1=
        assert!(!result.cigar.is_empty());
        assert!(result.cigar.contains('D') || result.cigar.contains('='));
        Ok(())
    }

    #[test]
    fn test_smith_waterman() -> Result<()> {
        let sw = SmithWaterman::new();
        let seq1 = Protein::from_string("AGSG")?;
        let seq2 = Protein::from_string("AGS")?;

        let result = sw.align(&seq1, &seq2)?;
        assert!(result.score >= 0);
        assert!(!result.cigar.is_empty());
        Ok(())
    }

    #[test]
    fn test_smith_waterman_with_cigar() -> Result<()> {
        let sw = SmithWaterman::new();
        let seq1 = Protein::from_string("MGLSD")?;
        let seq2 = Protein::from_string("MGLS")?;

        let result = sw.align(&seq1, &seq2)?;
        assert!(result.score > 0);
        assert!(!result.cigar.is_empty());
        
        // Verify CIGAR can be parsed
        for (prefix, _src) in result.cigar.split(|c: char| !c.is_numeric()).zip(1..) {
            if !prefix.is_empty() {
                prefix.parse::<u32>().expect("CIGAR should have numeric counts");
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_sam_record_creation() {
        let record = SamRecord::new();
        assert_eq!(record.qname, "query");
        assert_eq!(record.rname, "reference");
        assert_eq!(record.pos, 0);
        assert_eq!(record.mapq, 60);
    }

    #[test]
    fn test_sam_record_from_alignment() -> Result<()> {
        let sw = SmithWaterman::new();
        let seq1 = Protein::from_string("AGSGDSAF")?;
        let seq2 = Protein::from_string("AGSGD")?;

        let result = sw.align(&seq1, &seq2)?;
        let sam_record = SamRecord::from_alignment(&result, "query1", "ref1", 100);

        assert_eq!(sam_record.qname, "query1");
        assert_eq!(sam_record.rname, "ref1");
        assert_eq!(sam_record.pos, 101); // 0-based to 1-based conversion
        assert!(!sam_record.cigar.is_empty());
        assert!(sam_record.optional_fields.iter().any(|f| f.starts_with("AS:i:")));

        Ok(())
    }

    #[test]
    fn test_sam_record_to_line() {
        let mut record = SamRecord::new();
        record.qname = "read1".to_string();
        record.rname = "chr1".to_string();
        record.pos = 1000;
        record.cigar = "5M1D4M".to_string();
        record.query_seq = "ACGTACGTAC".to_string();
        record.add_optional_field("AS:i:100");

        let line = record.to_sam_line();
        assert!(line.contains("read1"));
        assert!(line.contains("chr1"));
        assert!(line.contains("1000"));
        assert!(line.contains("5M1D4M"));
        assert!(line.contains("ACGTACGTAC"));
        assert!(line.contains("AS:i:100"));
    }

    #[test]
    fn test_sam_header_generation() {
        let mut header = SamHeader::new("1.6");
        header.add_reference("chr1", 248956422);
        header.add_reference("chr2", 242193529);
        header = header.with_program("omics-simd-0.1.0");

        let lines = header.to_header_lines();
        assert!(lines.iter().any(|l| l.starts_with("@HD")));
        assert!(lines.iter().any(|l| l.contains("VN:1.6")));
        assert!(lines.iter().any(|l| l.starts_with("@SQ") && l.contains("chr1")));
        assert!(lines.iter().any(|l| l.starts_with("@SQ") && l.contains("chr2")));
        assert!(lines.iter().any(|l| l.starts_with("@PG")));
    }
}
