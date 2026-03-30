//! # Scoring Infrastructure Module
//!
//! Provides scoring matrices (BLOSUM, PAM) and affine gap penalty models
//! for biologically accurate sequence alignment.

use serde::{Deserialize, Serialize};
use crate::error::{Error, Result};
use crate::protein::AminoAcid;

/// Affine gap penalty model
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffinePenalty {
    /// Gap opening penalty (typically negative)
    pub open: i32,
    /// Gap extension penalty (typically negative, less severe than open)
    pub extend: i32,
}

impl AffinePenalty {
    /// Create new affine penalty with validation
    pub fn new(open: i32, extend: i32) -> Result<Self> {
        if open > 0 || extend > 0 {
            return Err(Error::InvalidGapPenalty);
        }
        Ok(AffinePenalty { open, extend })
    }

    /// Default penalties suitable for protein alignment
    pub fn default_protein() -> Self {
        AffinePenalty {
            open: -11,
            extend: -1,
        }
    }

    /// Strict penalties for high-confidence alignment
    pub fn strict() -> Self {
        AffinePenalty {
            open: -16,
            extend: -4,
        }
    }

    /// Liberal penalties for distant sequences
    pub fn liberal() -> Self {
        AffinePenalty {
            open: -8,
            extend: -1,
        }
    }
}

impl Default for AffinePenalty {
    fn default() -> Self {
        Self::default_protein()
    }
}

/// Scoring matrix types available
///
/// # Matrix Selection Guide
///
/// Choose the matrix based on sequence similarity:
/// - **BLOSUM62** (default): General purpose, good for sequences ~62% identical
/// - **BLOSUM80**: More stringent, for closely related sequences (>80% identity)
/// - **BLOSUM45**: More permissive, for distant sequences (~45% identity)
/// - **PAM70**: For ~70 substitutions per 100 amino acids
/// - **PAM30**: For ~30 substitutions per 100 amino acids (more divergent)
///
/// Incorrectly choosing a matrix can severely impact alignment quality.
/// Always validate your choice against your sequence divergence and biological question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixType {
    /// BLOSUM62 - most commonly used for protein alignment
    Blosum62,
    /// BLOSUM45 - for more distant sequences
    Blosum45,
    /// BLOSUM80 - for closely related sequences
    Blosum80,
    /// PAM30 - for distant sequences
    Pam30,
    /// PAM70 - for moderate divergence
    Pam70,
}

impl fmt::Display for MatrixType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixType::Blosum62 => write!(f, "BLOSUM62"),
            MatrixType::Blosum45 => write!(f, "BLOSUM45"),
            MatrixType::Blosum80 => write!(f, "BLOSUM80"),
            MatrixType::Pam30 => write!(f, "PAM30"),
            MatrixType::Pam70 => write!(f, "PAM70"),
        }
    }
}

/// Scoring matrix for amino acid substitutions
///
/// # Biological Accuracy
///
/// The scoring matrices represent decades of research on amino acid substitution patterns.
/// They encode biochemical similarities - similar amino acids have higher scores.
///
/// ## Matrix Choice Impact
///
/// | Matrix | Best For | Risk if Misused |
/// |--------|----------|-----------------|
/// | BLOSUM62 | General protein alignment | May miss distant homologs |
/// | BLOSUM45 | Distant sequences (<50%) | May produce spurious alignments |
/// | BLOSUM80 | Close homologs (>80%) | Misses real divergent regions |
/// | PAM30 | Highly divergent sequences | Can over-penalize conservative changes |
/// | PAM70 | Moderately divergent sequences | General purpose alternative |
///
/// ## Warning: Silent Matrix Fallback (★★★ CRITICAL)
///
/// Prior to v0.8.1, requesting unsupported matrices silently fell back to BLOSUM62.
/// **This has been fixed** - unsupported matrices now return an error.
/// Always handle matrix creation errors in production code.
///
/// ```rust,no_run
/// use omics_simd::scoring::{ScoringMatrix, MatrixType};
///
/// // ✓ Correct: Handle potential errors
/// let matrix = ScoringMatrix::new(MatrixType::Blosum45)?;
///
/// // ✗ Wrong: Panics on unsupported matrix
/// let matrix = ScoringMatrix::new(MatrixType::Blosum45).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringMatrix {
    matrix_type: MatrixType,
    scores: Vec<Vec<i32>>,
    size: usize,
}

impl ScoringMatrix {
    /// Create a new scoring matrix
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix type cannot be loaded (e.g., corrupted data
    /// or dimensions don't match 24x24 for standard amino acids + special codes).
    ///
    /// # Example
    ///
    /// ```
    /// use omics_simd::scoring::{ScoringMatrix, MatrixType};
    ///
    /// let matrix = ScoringMatrix::new(MatrixType::Blosum62).expect("BLOSUM62 should always work");
    /// let score = matrix.score(
    ///     omics_simd::protein::AminoAcid::Alanine,
    ///     omics_simd::protein::AminoAcid::Alanine
    /// );
    /// assert_eq!(score, 4); // A-A match score in BLOSUM62
    /// ```
    pub fn new(matrix_type: MatrixType) -> Result<Self> {
        let scores = match matrix_type {
            MatrixType::Blosum62 => Self::blosum62_data(),
            MatrixType::Blosum45 => Self::blosum45_data(),
            MatrixType::Blosum80 => Self::blosum80_data(),
            MatrixType::Pam30 => Self::pam30_data(),
            MatrixType::Pam70 => Self::pam70_data(),
        };

        let size = scores.len();
        if size != 24 {
            return Err(Error::InvalidMatrixDimensions);
        }

        Ok(ScoringMatrix {
            matrix_type,
            scores,
            size,
        })
    }

    /// Get score for amino acid pair
    ///
    /// Returns the substitution score (match or mismatch penalty) for the given amino acid pair.
    /// Scores are symmetric - score(A, B) == score(B, A).
    ///
    /// # Arguments
    ///
    /// * `aa1` - First amino acid
    /// * `aa2` - Second amino acid
    ///
    /// # Returns
    ///
    /// Score as i32:
    /// - Positive: favorable substitution or match
    /// - Negative: unfavorable substitution (mismatch penalty)
    /// - Large negative (-100): penalty for invalid amino acid indices
    ///
    /// # Example
    ///
    /// ```
    /// use omics_simd::scoring::{ScoringMatrix, MatrixType};
    /// use omics_simd::protein::AminoAcid;
    ///
    /// let matrix = ScoringMatrix::new(MatrixType::Blosum62).unwrap();
    /// 
    /// // Same amino acid (match) - always positive
    /// let match_score = matrix.score(AminoAcid::Leucine, AminoAcid::Leucine);
    /// assert!(match_score > 0);
    /// 
    /// // Similar amino acids (conservative) - usually positive
    /// let conservative = matrix.score(AminoAcid::Asparticacid, AminoAcid::Glutamicacid);
    /// assert!(conservative > matrix.score(AminoAcid::Asparticacid, AminoAcid::Tryptophan));
    /// ```
    pub fn score(&self, aa1: AminoAcid, aa2: AminoAcid) -> i32 {
        let i = aa1.index();
        let j = aa2.index();
        if i < self.size && j < self.size {
            self.scores[i][j]
        } else {
            -100 // Penalty for invalid indices
        }
    }

    /// Get matrix type
    pub fn matrix_type(&self) -> MatrixType {
        self.matrix_type
    }

    /// Get scores as a reference (for bridge conversions)
    pub fn raw_scores(&self) -> &Vec<Vec<i32>> {
        &self.scores
    }

    /// Get matrix size
    pub fn size(&self) -> usize {
        self.size
    }

    /// BLOSUM62 scoring matrix (most commonly used)
    fn blosum62_data() -> Vec<Vec<i32>> {
        vec![
            vec![4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, -1, -4], // A
            vec![-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4], // R
            vec![-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4], // N
            vec![-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4], // D
            vec![0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4], // C
            vec![-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 4, -1, -4], // E
            vec![-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4], // Q
            vec![0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -4, -3, -3, -1, -2, -1, -4], // G
            vec![-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4], // H
            vec![-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4], // I
            vec![-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4], // L
            vec![-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4], // K
            vec![-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4], // M
            vec![-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4], // F
            vec![-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4], // P
            vec![1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4], // S
            vec![0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4], // T
            vec![-3, -3, -4, -4, -2, -2, -3, -4, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4], // W
            vec![-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4], // Y
            vec![0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4], // V
            vec![-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4], // B
            vec![-1, 0, 0, 1, -3, 4, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 5, -1, -4], // Z
            vec![-1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4], // X
            vec![-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1], // *
        ]
    }

    /// BLOSUM45 scoring matrix (for more distant sequences)
    fn blosum45_data() -> Vec<Vec<i32>> {
        vec![
            vec![5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, -1, -5],
            vec![-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -2, -1, -1, -3, -1, -3, -1, 0, -1, -5],
            vec![-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3, 4, 0, -1, -5],
            vec![-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -4, -1, 0, -1, -4, -3, -3, 5, 1, -1, -5],
            vec![-1, -4, -2, -4, 13, -3, -3, -3, -2, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -2, -3, -2, -5],
            vec![-1, 1, 0, 0, -3, 6, 2, -2, 0, -3, -2, 1, 0, -4, -1, 0, -1, -3, -2, -2, 0, 5, -1, -5],
            vec![-1, 0, 0, 2, -3, 2, 6, -2, 0, -4, -3, 1, -2, -3, -1, 0, -1, -4, -2, -2, 1, 5, -1, -5],
            vec![0, -3, 0, -1, -3, -2, -2, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -4, -3, -3, -1, -2, -1, -5],
            vec![-2, 0, 1, -1, -2, 0, 0, -2, 10, -3, -3, -1, -2, -1, -2, -1, -2, -3, 2, -3, 0, 0, -1, -5],
            vec![-1, -4, -3, -4, -2, -3, -4, -4, -3, 5, 2, -3, 2, 0, -3, -2, -1, -3, -1, 4, -3, -3, -1, -5],
            vec![-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1, -4, -3, -1, -5],
            vec![-1, 3, 0, -1, -3, 1, 1, -2, -1, -3, -3, 5, -2, -3, -1, 0, -1, -3, -2, -2, -1, 1, -1, -5],
            vec![-1, -2, -2, -4, -2, 0, -2, -3, -2, 2, 3, -2, 6, 0, -3, -2, -1, -2, -1, 1, -3, -1, -1, -5],
            vec![-2, -3, -4, -4, -2, -4, -3, -4, -1, 0, 1, -3, 0, 9, -4, -2, -2, 0, 4, 0, -4, -3, -1, -5],
            vec![-1, -2, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1, -2, -5],
            vec![1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -3, 0, -2, -2, -1, 5, 2, -4, -2, -2, 0, 0, 0, -5],
            vec![0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0, -1, -1, 0, -5],
            vec![-3, -3, -4, -4, -5, -3, -4, -4, -3, -3, -2, -3, -2, 0, -4, -4, -3, 15, 2, -3, -4, -4, -2, -5],
            vec![-2, -1, -2, -3, -3, -2, -2, -3, 2, -1, -1, -2, -1, 4, -3, -2, -2, 2, 8, -1, -3, -2, -1, -5],
            vec![0, -3, -3, -3, -1, -2, -2, -3, -3, 4, 1, -2, 1, 0, -3, -2, 0, -3, -1, 5, -3, -2, -1, -5],
            vec![-2, -1, 4, 5, -2, 0, 1, -1, 0, -3, -4, -1, -3, -4, -2, 0, -1, -4, -3, -3, 5, 2, -1, -5],
            vec![-1, 0, 0, 1, -3, 5, 5, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -4, -2, -2, 2, 5, -1, -5],
            vec![-1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -5],
            vec![-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, 1],
        ]
    }

    /// BLOSUM80 scoring matrix (for closely related sequences)
    fn blosum80_data() -> Vec<Vec<i32>> {
        vec![
            vec![7, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 2, 0, -4, -2, 0, -2, -1, -1, -6],
            vec![-1, 6, 0, -2, -4, 1, -1, -3, -1, -4, -3, 2, -1, -3, -2, -1, -1, -4, -3, -4, -1, 0, -1, -6],
            vec![-2, 0, 8, 1, -4, 0, -1, -1, 1, -4, -4, 0, -2, -4, -2, 1, 0, -5, -3, -4, 5, -1, -1, -6],
            vec![-2, -2, 1, 8, -4, -1, 2, -2, -1, -4, -4, -1, -3, -4, -1, 0, -1, -5, -4, -4, 5, 1, -1, -6],
            vec![0, -4, -4, -4, 10, -4, -4, -4, -4, -1, -2, -4, -2, -2, -4, -1, -1, -3, -3, -1, -4, -4, -2, -6],
            vec![-1, 1, 0, -1, -4, 7, 2, -2, 0, -4, -3, 1, 0, -4, -1, 0, -1, -3, -2, -3, 0, 4, -1, -6],
            vec![-1, -1, -1, 2, -4, 2, 7, -2, -1, -4, -3, 1, -2, -4, -1, 0, -1, -4, -3, -3, 1, 5, -1, -6],
            vec![0, -3, -1, -2, -4, -2, -2, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -4, -3, -4, -1, -2, -1, -6],
            vec![-2, -1, 1, -1, -4, 0, -1, -2, 10, -4, -3, 0, -2, -1, -2, -1, -2, -3, 3, -4, 0, -1, -1, -6],
            vec![-1, -4, -4, -4, -1, -4, -4, -4, -4, 5, 2, -4, 2, 0, -4, -3, -1, -3, -1, 4, -4, -4, -1, -6],
            vec![-1, -3, -4, -4, -2, -3, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1, -4, -3, -1, -6],
            vec![-1, 2, 0, -1, -4, 1, 1, -2, 0, -4, -3, 6, -1, -4, -1, 0, -1, -4, -2, -3, -1, 1, -1, -6],
            vec![-1, -1, -2, -3, -2, 0, -2, -3, -2, 2, 3, -1, 7, 0, -3, -2, -1, -1, -1, 1, -3, -1, -1, -6],
            vec![-2, -3, -4, -4, -2, -4, -4, -4, -1, 0, 1, -4, 0, 8, -4, -2, -2, 2, 4, -1, -4, -4, -1, -6],
            vec![-1, -2, -2, -1, -4, -1, -1, -2, -2, -4, -4, -1, -3, -4, 9, -1, -1, -4, -3, -3, -2, -1, -2, -6],
            vec![2, -1, 1, 0, -1, 0, 0, 0, -1, -3, -3, 0, -2, -2, -1, 7, 2, -4, -2, -2, 0, 0, 0, -6],
            vec![0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 6, -3, -2, 0, -1, -1, 0, -6],
            vec![-4, -4, -5, -5, -3, -3, -4, -4, -3, -3, -2, -4, -1, 2, -4, -4, -3, 15, 3, -3, -5, -4, -2, -6],
            vec![-2, -3, -3, -4, -3, -2, -3, -3, 3, -1, -1, -2, -1, 4, -3, -2, -2, 3, 8, -1, -4, -3, -1, -6],
            vec![0, -4, -4, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5, -4, -3, -1, -6],
            vec![-2, -1, 5, 5, -4, 0, 1, -1, 0, -4, -4, -1, -3, -4, -2, 0, -1, -5, -4, -4, 6, 2, -1, -6],
            vec![-1, 0, -1, 1, -4, 4, 5, -2, -1, -4, -3, 1, -1, -4, -1, 0, -1, -4, -3, -3, 2, 6, -1, -6],
            vec![-1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -6],
            vec![-6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, 1],
        ]
    }

    /// PAM30 scoring matrix
    fn pam30_data() -> Vec<Vec<i32>> {
        vec![
            vec![6, -7, -4, -3, -6, -4, -2, -2, -7, -5, -6, -4, -5, -5, -4, 1, 0, -13, -8, -2, -4, -4, -1, -17],
            vec![-7, 8, -6, -10, -8, -7, -9, -7, -1, -6, -5, -4, -4, -9, -8, -6, -7, -2, -10, -8, -7, -8, -1, -17],
            vec![-4, -6, 8, 2, -11, -4, -2, -4, -7, -7, -8, -4, -9, -9, -6, 0, -2, -13, -8, -5, 4, -3, -1, -17],
            vec![-3, -10, 2, 8, -14, -4, 2, -3, -10, -9, -9, -6, -10, -14, -3, -1, -3, -17, -13, -9, 5, 1, -1, -17],
            vec![-6, -8, -11, -14, 10, -14, -14, -9, -7, -6, -15, -8, -7, -13, -8, -3, -8, -15, -4, -6, -12, -14, -1, -17],
            vec![-4, -7, -4, -4, -14, 7, 1, -7, -4, -8, -5, -7, -3, -13, -6, -5, -5, -13, -12, -7, -4, 5, -1, -17],
            vec![-2, -9, -2, 2, -14, 1, 8, -7, -4, -9, -7, -4, -4, -13, -5, -4, -4, -12, -9, -8, 0, 6, -1, -17],
            vec![-2, -7, -4, -3, -9, -7, -7, 6, -9, -9, -10, -7, -8, -9, -6, -2, -4, -15, -11, -5, -4, -7, -1, -17],
            vec![-7, -1, -7, -10, -7, -4, -4, -9, 9, -9, -6, -6, -5, -2, -10, -6, -7, -3, 0, -7, -8, -4, -1, -17],
            vec![-5, -6, -7, -9, -6, -8, -9, -9, -9, 8, -1, -6, -2, -1, -8, -6, -4, -14, -6, 2, -8, -8, -1, -17],
            vec![-6, -5, -8, -9, -15, -5, -7, -10, -6, -1, 7, -8, 0, -4, -7, -7, -4, -6, -7, -2, -8, -6, -1, -17],
            vec![-4, -4, -4, -6, -8, -7, -4, -7, -6, -6, -8, 5, -4, -9, -4, -3, -4, -15, -8, -6, -5, -6, -1, -17],
            vec![-5, -4, -9, -10, -7, -3, -4, -8, -5, -2, 0, -4, 11, -4, -7, -5, -2, -14, -7, -1, -10, -4, -1, -17],
            vec![-5, -9, -9, -14, -13, -13, -13, -9, -2, -1, -4, -9, -4, 9, -9, -7, -8, -8, 0, -3, -12, -13, -1, -17],
            vec![-4, -8, -6, -3, -8, -6, -5, -6, -10, -8, -7, -4, -7, -9, 8, -3, -4, -14, -9, -5, -5, -5, -1, -17],
            vec![1, -6, 0, -1, -3, -5, -4, -2, -6, -6, -7, -3, -5, -7, -3, 5, 2, -12, -7, -4, -1, -4, -1, -17],
            vec![0, -7, -2, -3, -8, -5, -4, -4, -7, -4, -4, -4, -2, -8, -4, 2, 6, -10, -7, -2, -3, -5, -1, -17],
            vec![-13, -2, -13, -17, -15, -13, -12, -15, -3, -14, -6, -15, -14, -8, -14, -12, -10, 12, 2, -14, -15, -12, -1, -17],
            vec![-8, -10, -8, -13, -4, -12, -9, -11, 0, -6, -7, -8, -7, 0, -9, -7, -7, 2, 9, -6, -10, -11, -1, -17],
            vec![-2, -8, -5, -9, -6, -7, -8, -5, -7, 2, -2, -6, -1, -3, -5, -4, -2, -14, -6, 7, -7, -7, -1, -17],
            vec![-4, -7, 4, 5, -12, -4, 0, -4, -8, -8, -8, -5, -10, -12, -5, -1, -3, -15, -10, -7, 5, 0, -1, -17],
            vec![-4, -8, -3, 1, -14, 5, 6, -7, -4, -8, -6, -6, -4, -13, -5, -4, -5, -12, -11, -7, 0, 6, -1, -17],
            vec![-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -17],
            vec![-17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, 1],
        ]
    }

    /// PAM70 scoring matrix
    fn pam70_data() -> Vec<Vec<i32>> {
        vec![
            vec![5, -2, -1, -1, -2, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 1, -6, -3, 0, -1, -1, -1, -8],
            vec![-2, 6, -1, -2, -4, 0, -2, -3, 0, -3, -3, -1, 0, -4, -2, -1, -1, 2, -4, -3, -1, -1, -1, -8],
            vec![-1, -1, 4, 2, -4, 0, 1, -1, 0, -2, -3, 1, -2, -3, -2, 0, 0, -4, -2, -2, 3, 0, -1, -8],
            vec![-1, -2, 2, 4, -5, 0, 2, 0, 0, -3, -4, 0, -3, -4, 0, 0, 0, -5, -3, -3, 3, 1, -1, -8],
            vec![-2, -4, -4, -5, 12, -5, -5, -3, -3, -3, -6, -4, -5, -5, -4, -2, -2, -8, -1, -2, -4, -5, -2, -8],
            vec![-1, 0, 0, 0, -5, 5, 2, -2, 1, -3, -2, 0, 0, -4, -1, 0, 0, -5, -3, -2, 0, 3, -1, -8],
            vec![-1, -2, 1, 2, -5, 2, 6, -2, 0, -3, -3, 0, -2, -3, 0, 0, -1, -5, -2, -2, 1, 4, -1, -8],
            vec![0, -3, -1, 0, -3, -2, -2, 5, -2, -4, -4, -2, -3, -3, -2, 0, -1, -7, -3, -2, -1, -2, -1, -8],
            vec![-2, 0, 0, 0, -3, 1, 0, -2, 6, -3, -1, 0, -2, 0, -2, -1, -1, -3, 2, -2, 0, 0, -1, -8],
            vec![-1, -3, -2, -3, -3, -3, -3, -4, -3, 5, 2, -3, 2, 1, -2, -2, -1, -5, -1, 4, -3, -3, -1, -8],
            vec![-2, -3, -3, -4, -6, -2, -3, -4, -1, 2, 6, -3, 4, 2, -3, -3, -2, -2, -1, 2, -3, -3, -1, -8],
            vec![-1, -1, 1, 0, -4, 0, 0, -2, 0, -3, -3, 3, -1, -3, -1, 0, 0, -5, -2, -2, 0, 0, -1, -8],
            vec![-1, 0, -2, -3, -5, 0, -2, -3, -2, 2, 4, -1, 5, 1, -3, -2, 0, -5, -1, 2, -3, -1, -1, -8],
            vec![-3, -4, -3, -4, -5, -4, -3, -3, 0, 1, 2, -3, 1, 6, -4, -2, -2, 0, 4, 1, -4, -4, -1, -8],
            vec![-1, -2, -2, 0, -4, -1, 0, -2, -2, -2, -3, -1, -3, -4, 6, -1, -1, -5, -3, -1, -1, 0, -1, -8],
            vec![1, -1, 0, 0, -2, 0, 0, 0, -1, -2, -3, 0, -2, -2, -1, 3, 1, -5, -2, -1, 0, 0, -1, -8],
            vec![1, -1, 0, 0, -2, 0, -1, -1, -1, -1, -2, 0, 0, -2, -1, 1, 3, -4, -2, 0, 0, 0, -1, -8],
            vec![-6, 2, -4, -5, -8, -5, -5, -7, -3, -5, -2, -5, -5, 0, -5, -5, -4, 17, 0, -6, -5, -5, -2, -8],
            vec![-3, -4, -2, -3, -1, -3, -2, -3, 2, -1, -1, -2, -1, 4, -3, -2, -2, 0, 7, -1, -3, -3, -1, -8],
            vec![0, -3, -2, -3, -2, -2, -2, -2, -2, 4, 2, -2, 2, 1, -1, -1, 0, -6, -1, 4, -2, -2, -1, -8],
            vec![-1, -1, 3, 3, -4, 0, 1, -1, 0, -3, -3, 0, -3, -4, -1, 0, 0, -5, -3, -2, 3, 1, -1, -8],
            vec![-1, -1, 0, 1, -5, 3, 4, -2, 0, -3, -3, 0, -1, -4, 0, 0, 0, -5, -3, -2, 1, 4, -1, -8],
            vec![-1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -8],
            vec![-8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 1],
        ]
    }
}

impl Default for ScoringMatrix {
    fn default() -> Self {
        Self::new(MatrixType::Blosum62).expect("BLOSUM62 matrix should be valid")
    }
}

use std::fmt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_penalty() {
        let penalty = AffinePenalty::new(-11, -1).unwrap();
        assert_eq!(penalty.open, -11);
        assert_eq!(penalty.extend, -1);
    }

    #[test]
    fn test_invalid_penalty() {
        assert!(AffinePenalty::new(11, -1).is_err());
        assert!(AffinePenalty::new(-11, 1).is_err());
    }

    #[test]
    fn test_scoring_matrix() {
        let matrix = ScoringMatrix::default();
        let aa1 = AminoAcid::Alanine;
        let aa2 = AminoAcid::Alanine;
        assert_eq!(matrix.score(aa1, aa2), 4); // Diagonal should be positive
    }

    #[test]
    fn test_blosum45_matrix() -> Result<()> {
        let matrix = ScoringMatrix::new(MatrixType::Blosum45)?;
        let aa1 = AminoAcid::Alanine;
        let aa2 = AminoAcid::Alanine;
        // BLOSUM45 diagonal for A should be 5
        assert_eq!(matrix.score(aa1, aa2), 5);
        
        // A-C should be negative (mismatch penalty)
        let aa_c = AminoAcid::Cysteine;
        assert!(matrix.score(aa1, aa_c) < 0);
        Ok(())
    }

    #[test]
    fn test_blosum80_matrix() -> Result<()> {
        let matrix = ScoringMatrix::new(MatrixType::Blosum80)?;
        let aa1 = AminoAcid::Alanine;
        let aa2 = AminoAcid::Alanine;
        // BLOSUM80 diagonal for A should be 7 (higher than BLOSUM45/62 - for close homologs)
        assert_eq!(matrix.score(aa1, aa2), 7);
        Ok(())
    }

    #[test]
    fn test_pam30_matrix() -> Result<()> {
        let matrix = ScoringMatrix::new(MatrixType::Pam30)?;
        let aa1 = AminoAcid::Alanine;
        let aa2 = AminoAcid::Alanine;
        // PAM30 diagonal for A should be 6
        assert_eq!(matrix.score(aa1, aa2), 6);
        Ok(())
    }

    #[test]
    fn test_pam70_matrix() -> Result<()> {
        let matrix = ScoringMatrix::new(MatrixType::Pam70)?;
        let aa1 = AminoAcid::Alanine;
        let aa2 = AminoAcid::Alanine;
        // PAM70 diagonal for A should be 5
        assert_eq!(matrix.score(aa1, aa2), 5);
        Ok(())
    }

    #[test]
    fn test_matrix_symmetry_blosum45() -> Result<()> {
        let matrix = ScoringMatrix::new(MatrixType::Blosum45)?;
        let aa1 = AminoAcid::Alanine;
        let aa2 = AminoAcid::GlutamicAcid;
        // Scoring matrices should be symmetric
        assert_eq!(matrix.score(aa1, aa2), matrix.score(aa2, aa1));
        Ok(())
    }

    #[test]
    fn test_matrix_symmetry_pam70() -> Result<()> {
        let matrix = ScoringMatrix::new(MatrixType::Pam70)?;
        let aa1 = AminoAcid::Valine;
        let aa2 = AminoAcid::Isoleucine;
        // Scoring matrices should be symmetric
        assert_eq!(matrix.score(aa1, aa2), matrix.score(aa2, aa1));
        Ok(())
    }

    #[test]
    fn test_matrix_type_display() {
        assert_eq!(MatrixType::Blosum62.to_string(), "BLOSUM62");
        assert_eq!(MatrixType::Blosum45.to_string(), "BLOSUM45");
        assert_eq!(MatrixType::Blosum80.to_string(), "BLOSUM80");
        assert_eq!(MatrixType::Pam30.to_string(), "PAM30");
        assert_eq!(MatrixType::Pam70.to_string(), "PAM70");
    }
}
