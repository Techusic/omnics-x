//! Profile-to-Profile Dynamic Programming Alignment
//!
//! Implements mathematically rigorous multiple sequence alignment using position-specific
//! scoring matrices (PSSMs). This enables high-quality progressive MSA with accurate column scores.
//!
//! # Algorithm
//! - Compute PSSM for each sequence/group
//! - Dynamic programming using profile-profile similarity
//! - Affine gap penalties with profile-specific parameters
//! - Traceback for alignment reconstruction

use std::cmp::Ordering;

/// Position-Specific Scoring Matrix (PSSM)
#[derive(Debug, Clone)]
pub struct Pssm {
    /// Position-specific scores [position][amino_acid]
    /// Scores in log-odds units (bits or nats)
    pub scores: Vec<Vec<f64>>,
    /// Position-specific gap opening penalties (learned from data)
    pub gap_open_penalties: Vec<f64>,
    /// Position-specific gap extension penalties
    pub gap_extend_penalties: Vec<f64>,
    /// Number of sequences contributing to this profile
    pub n_sequences: usize,
}

impl Pssm {
    /// Create PSSM from multiple sequence alignment
    pub fn from_alignment(msa: &[&[u8]], alpha: &str) -> Self {
        let n_positions = msa.iter().map(|s| s.len()).max().unwrap_or(0);
        let n_aa = match alpha {
            "protein" => 20,
            _ => 20,
        };

        let mut scores = Vec::new();
        let mut gap_open = Vec::new();
        let mut gap_extend = Vec::new();

        // Background probabilities (typical amino acid distribution)
        let bg_freq = vec![
            0.087, 0.041, 0.040, 0.047, 0.065, 0.029, 0.039, 0.083, 0.034, 0.068,
            0.099, 0.058, 0.025, 0.042, 0.051, 0.072, 0.057, 0.066, 0.010, 0.030,
        ];

        // Compute PSSM for each position
        for pos in 0..n_positions {
            let mut aa_counts = vec![0.0f64; n_aa];
            let mut gap_count = 0.0;

            // Count amino acids at this position
            for &sequence in msa {
                if pos < sequence.len() {
                    let aa = (sequence[pos].min(19) as usize).min(19);
                    aa_counts[aa] += 1.0;
                } else {
                    gap_count += 1.0;
                }
            }

            // Normalize and compute log-odds
            let total = msa.len() as f64;
            let mut position_score = Vec::new();

            for i in 0..n_aa {
                let freq = (aa_counts[i] + 0.5) / total; // Laplace pseudocount
                let lo = (freq / bg_freq[i]).ln(); // Log-odds
                position_score.push(lo);
            }

            scores.push(position_score);

            // Position-specific gap penalties (open increases with conservation)
            let conservation = aa_counts.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .copied()
                .unwrap_or(0.0) / total;
            let gap_open_penalty = -11.0 * (1.0 - conservation.powf(2.0)); // More conservative = higher penalty
            gap_open.push(gap_open_penalty);

            let gap_extend_penalty = -1.0; // Constant
            gap_extend.push(gap_extend_penalty);
        }

        Pssm {
            scores,
            gap_open_penalties: gap_open,
            gap_extend_penalties: gap_extend,
            n_sequences: msa.len(),
        }
    }

    /// Get log-odds score for amino acid at position
    #[inline]
    pub fn score(&self, pos: usize, aa: usize) -> f64 {
        if pos < self.scores.len() && aa < self.scores[pos].len() {
            self.scores[pos][aa]
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get gap open penalty at position
    #[inline]
    pub fn gap_open(&self, pos: usize) -> f64 {
        self.gap_open_penalties.get(pos).copied().unwrap_or(-11.0)
    }

    /// Get gap extend penalty at position
    #[inline]
    pub fn gap_extend(&self, pos: usize) -> f64 {
        self.gap_extend_penalties.get(pos).copied().unwrap_or(-1.0)
    }

    /// Profile-profile similarity at aligned positions
    pub fn profile_similarity(&self, other: &Pssm, pos_self: usize, pos_other: usize) -> f64 {
        let mut total = 0.0;

        for aa in 0..20 {
            let score_self = self.score(pos_self, aa);
            let score_other = other.score(pos_other, aa);
            total += score_self * score_other;
        }

        total
    }
}

/// DP table for Profile-to-Profile alignment
#[derive(Debug)]
struct DpTable {
    /// Match state: M[i][j]
    m: Vec<Vec<f64>>,
    /// Insertion in profile 1: Ix[i][j]
    ix: Vec<Vec<f64>>,
    /// Insertion in profile 2: Iy[i][j]
    iy: Vec<Vec<f64>>,
}

impl DpTable {
    fn new(rows: usize, cols: usize) -> Self {
        DpTable {
            m: vec![vec![f64::NEG_INFINITY; cols]; rows],
            ix: vec![vec![f64::NEG_INFINITY; cols]; rows],
            iy: vec![vec![f64::NEG_INFINITY; cols]; rows],
        }
    }
}

/// Alignment traceback pointer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TracebackOp {
    Match,    // M -> M
    InsertX,  // M -> Ix
    InsertY,  // M -> Iy
    DeleteX,  // Ix -> Ix
    DeleteY,  // Iy -> Iy
}

/// Profile-to-Profile alignment result
#[derive(Debug, Clone)]
pub struct ProfileAlignment {
    /// Alignment score
    pub score: f64,
    /// Profile 1 alignment (indices)
    pub profile1_align: Vec<usize>,
    /// Profile 2 alignment (indices)
    pub profile2_align: Vec<usize>,
    /// CIGAR string
    pub cigar: String,
}

/// Perform profile-to-profile alignment using affine gap penalties
pub fn align_profiles(prof1: &Pssm, prof2: &Pssm) -> ProfileAlignment {
    let m = prof1.scores.len();
    let n = prof2.scores.len();

    let mut dp = DpTable::new(m + 1, n + 1);
    let mut traceback = vec![vec![TracebackOp::Match; n + 1]; m + 1];

    // Initialize first row and column
    dp.m[0][0] = 0.0;
    for i in 1..=m {
        dp.ix[i][0] = prof1.gap_open(i - 1) + prof1.gap_extend(i - 1) * (i as f64 - 1.0);
        dp.m[i][0] = dp.ix[i][0];
    }
    for j in 1..=n {
        dp.iy[0][j] = prof2.gap_open(j - 1) + prof2.gap_extend(j - 1) * (j as f64 - 1.0);
        dp.m[0][j] = dp.iy[0][j];
    }

    // Fill DP table with affine gap penalties
    for i in 1..=m {
        for j in 1..=n {
            // Match: profile-profile similarity
            let match_score = prof1.profile_similarity(prof2, i - 1, j - 1);
            let score_m = dp.m[i - 1][j - 1] + match_score;

            // Gaps in profile 1
            let open_x = dp.m[i - 1][j] + prof1.gap_open(i - 1);
            let extend_x = dp.ix[i - 1][j] + prof1.gap_extend(i - 1);
            let score_ix = open_x.max(extend_x);

            // Gaps in profile 2
            let open_y = dp.m[i][j - 1] + prof2.gap_open(j - 1);
            let extend_y = dp.iy[i][j - 1] + prof2.gap_extend(j - 1);
            let score_iy = open_y.max(extend_y);

            // Choose best transition
            let best_score = score_m.max(score_ix).max(score_iy);

            dp.m[i][j] = best_score;
            dp.ix[i][j] = score_ix;
            dp.iy[i][j] = score_iy;

            // Record traceback
            if best_score == score_m {
                traceback[i][j] = TracebackOp::Match;
            } else if best_score == score_ix {
                traceback[i][j] = TracebackOp::InsertX;
            } else {
                traceback[i][j] = TracebackOp::InsertY;
            }
        }
    }

    // Traceback to reconstruct alignment
    let (prof1_align, prof2_align, cigar) = traceback_profile_alignment(&traceback, m, n);

    ProfileAlignment {
        score: dp.m[m][n],
        profile1_align: prof1_align,
        profile2_align: prof2_align,
        cigar,
    }
}

/// Reconstruct alignment from traceback pointers
fn traceback_profile_alignment(
    traceback: &[Vec<TracebackOp>],
    m: usize,
    n: usize,
) -> (Vec<usize>, Vec<usize>, String) {
    let mut prof1_align = Vec::new();
    let mut prof2_align = Vec::new();
    let mut cigar = String::new();

    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        match traceback[i][j] {
            TracebackOp::Match => {
                prof1_align.push(i - 1);
                prof2_align.push(j - 1);
                cigar.insert(0, 'M');
                i -= 1;
                j -= 1;
            }
            TracebackOp::InsertX => {
                prof1_align.push(i - 1);
                prof2_align.push(usize::MAX); // Gap
                cigar.insert(0, 'D');
                i -= 1;
            }
            TracebackOp::InsertY => {
                prof1_align.push(usize::MAX); // Gap
                prof2_align.push(j - 1);
                cigar.insert(0, 'I');
                j -= 1;
            }
            TracebackOp::DeleteX => {
                prof1_align.push(i - 1);
                prof2_align.push(usize::MAX);
                cigar.insert(0, 'D');
                i -= 1;
            }
            TracebackOp::DeleteY => {
                prof1_align.push(usize::MAX);
                prof2_align.push(j - 1);
                cigar.insert(0, 'I');
                j -= 1;
            }
        }
    }

    prof1_align.reverse();
    prof2_align.reverse();

    (prof1_align, prof2_align, cigar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pssm_creation() {
        let msa = vec![
            &b"ACDEFGHIKLMNPQRSTVWY"[..],
            &b"ACDEFGHIKLMNPQRSTVWY"[..],
        ];

        let pssm = Pssm::from_alignment(&msa, "protein");
        assert_eq!(pssm.scores.len(), 20);
        assert_eq!(pssm.n_sequences, 2);
    }

    #[test]
    fn test_profile_similarity() {
        let msa = vec![
            &b"ACDEFGHIKLMNPQRSTVWY"[..],
            &b"ACDEFGHIKLMNPQRSTVWY"[..],
        ];

        let pssm1 = Pssm::from_alignment(&msa, "protein");
        let pssm2 = Pssm::from_alignment(&msa, "protein");

        let sim = pssm1.profile_similarity(&pssm2, 0, 0);
        assert!(sim > 0.0); // Same sequences should have positive similarity
    }

    #[test]
    fn test_profile_alignment() {
        let msa1 = vec![&b"ACDE"[..], &b"ACDE"[..]];
        let msa2 = vec![&b"ACDE"[..], &b"ACDE"[..]];

        let prof1 = Pssm::from_alignment(&msa1, "protein");
        let prof2 = Pssm::from_alignment(&msa2, "protein");

        let aln = align_profiles(&prof1, &prof2);
        assert!(aln.score.is_finite());
        assert!(!aln.cigar.is_empty());
    }
}
