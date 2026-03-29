//! MSA Profile-based alignment with progressive computation
//! Connects profile alignment into the progressive MSA loop

use crate::error::Result;
use std::collections::HashMap;

/// Profile alignment state tracking
#[derive(Debug, Clone)]
pub struct ProfileAlignmentState {
    /// Aligned sequences
    pub sequences: Vec<String>,
    /// Position-specific score matrices (PSSMs)
    pub pssm: Vec<Vec<f32>>,
    /// Alignment columns
    pub columns: Vec<String>,
    /// Position weights
    pub weights: Vec<f32>,
    /// Consensus sequence
    pub consensus: String,
    /// Gapped column flags
    pub gapped: Vec<bool>,
}

impl ProfileAlignmentState {
    /// Create new profile state
    pub fn new(sequences: Vec<String>) -> Result<Self> {
        let num_seqs = sequences.len();
        if num_seqs == 0 {
            return Err(crate::error::Error::AlignmentError("Empty sequence list".to_string()));
        }

        let seq_len = sequences[0].len();
        let pssm = vec![vec![0.0; 20]; seq_len]; // 20 for amino acids
        let weights = vec![1.0 / num_seqs as f32; num_seqs];

        Ok(ProfileAlignmentState {
            sequences,
            pssm,
            columns: vec![String::new(); seq_len],
            weights,
            consensus: String::new(),
            gapped: vec![false; seq_len],
        })
    }

    /// Align profile to sequence using dynamic programming
    pub fn align_profile_to_sequence(
        &self,
        query: &str,
        gap_open: f32,
        gap_extend: f32,
    ) -> ProfileAlignment {
        let m = self.sequences.len() + 1; // Profile columns + gap
        let n = query.len() + 1; // Query + gap

        // DP matrices
        let mut dp_match = vec![vec![f32::NEG_INFINITY; n]; m];
        let mut dp_gap_profile = vec![vec![f32::NEG_INFINITY; n]; m];
        let mut dp_gap_query = vec![vec![f32::NEG_INFINITY; n]; m];

        // Initialize
        dp_match[0][0] = 0.0;
        for i in 1..m {
            dp_gap_profile[i][0] = -gap_open - (i - 1) as f32 * gap_extend;
        }
        for j in 1..n {
            dp_gap_query[0][j] = -gap_open - (j - 1) as f32 * gap_extend;
        }

        // Fill DP matrices
        for i in 1..m {
            for j in 1..n {
                let query_char = query.chars().nth(j - 1).unwrap_or('*') as usize;

                // Match score: average over profile
                let mut match_score = 0.0;
                for (seq_idx, seq) in self.sequences.iter().enumerate() {
                    if let Some(ch) = seq.chars().nth(i - 1) {
                        let aa_idx = aa_to_index(ch);
                        let score = if aa_idx < self.pssm[i - 1].len() {
                            self.pssm[i - 1][aa_idx]
                        } else {
                            0.0
                        };
                        match_score += self.weights[seq_idx] * score;
                    }
                }

                // DP recurrence
                dp_match[i][j] = (dp_match[i - 1][j - 1]
                    .max(dp_gap_profile[i - 1][j - 1])
                    .max(dp_gap_query[i - 1][j - 1]))
                    + match_score;

                dp_gap_profile[i][j] = dp_match[i - 1][j] - gap_open;
                dp_gap_profile[i][j] = dp_gap_profile[i - 1][j] - gap_extend
                    .max(dp_gap_profile[i][j]);

                dp_gap_query[i][j] = dp_match[i][j - 1] - gap_open;
                dp_gap_query[i][j] = dp_gap_query[i][j - 1] - gap_extend
                    .max(dp_gap_query[i][j]);
            }
        }

        // Traceback
        let mut profile_align = String::new();
        let mut query_align = String::new();
        let mut i = m - 1;
        let mut j = n - 1;

        while i > 0 && j > 0 {
            let current_match = dp_match[i][j];
            let current_gap_p = dp_gap_profile[i][j];
            let current_gap_q = dp_gap_query[i][j];

            if current_match >= current_gap_p && current_match >= current_gap_q {
                profile_align.push(self.sequences[i - 1].chars().next().unwrap_or('-'));
                query_align.push(query.chars().nth(j - 1).unwrap_or('-'));
                i -= 1;
                j -= 1;
            } else if current_gap_p >= current_gap_q {
                profile_align.push('-');
                query_align.push(query.chars().nth(j - 1).unwrap_or('-'));
                j -= 1;
            } else {
                profile_align.push(self.sequences[i - 1].chars().next().unwrap_or('-'));
                query_align.push('-');
                i -= 1;
            }
        }

        while i > 0 {
            profile_align.push('-');
            i -= 1;
        }
        while j > 0 {
            query_align.push('-');
            j -= 1;
        }

        profile_align = profile_align.chars().rev().collect();
        query_align = query_align.chars().rev().collect();

        ProfileAlignment {
            profile_alignment: profile_align,
            query_alignment: query_align,
            score: dp_match[m - 1][n - 1],
        }
    }

    /// Update profile with new sequence alignment
    pub fn update_with_sequence(&mut self, aligned_seq: &str) {
        self.sequences.push(aligned_seq.to_string());
        let avg_weight = 1.0 / self.sequences.len() as f32;
        self.weights = vec![avg_weight; self.sequences.len()];
    }

    /// Compute consensus sequence from profile
    pub fn compute_consensus(&mut self, threshold: f32) {
        let width = self.sequences[0].len();
        self.consensus.clear();

        for col in 0..width {
            let mut aa_counts: HashMap<char, f32> = HashMap::new();

            for (seq_idx, seq) in self.sequences.iter().enumerate() {
                if let Some(ch) = seq.chars().nth(col) {
                    if ch != '-' && ch != '*' {
                        let count = aa_counts.entry(ch).or_insert(0.0);
                        *count += self.weights[seq_idx];
                    }
                }
            }

            let best_aa = aa_counts
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(aa, _)| *aa)
                .unwrap_or('X');

            self.consensus.push(best_aa);
        }
    }

    /// Build PSSM from aligned sequences
    pub fn build_pssm(&mut self) -> Result<()> {
        let width = self.sequences[0].len();
        let mut pssm = vec![vec![0.0; 20]; width];

        for col in 0..width {
            let mut aa_freqs = vec![0.0; 20];
            let mut total_weight = 0.0;

            for (seq_idx, seq) in self.sequences.iter().enumerate() {
                if let Some(ch) = seq.chars().nth(col) {
                    let aa_idx = aa_to_index(ch);
                    if aa_idx < 20 {
                        aa_freqs[aa_idx] += self.weights[seq_idx];
                        total_weight += self.weights[seq_idx];
                    }
                }
            }

            if total_weight > 0.0 {
                for i in 0..20 {
                    pssm[col][i] = (aa_freqs[i] / total_weight).ln();
                }
            }
        }

        self.pssm = pssm;
        Ok(())
    }
}

/// Result of profile alignment
#[derive(Debug, Clone)]
pub struct ProfileAlignment {
    pub profile_alignment: String,
    pub query_alignment: String,
    pub score: f32,
}

/// Convert amino acid to index (0-19)
fn aa_to_index(ch: char) -> usize {
    match ch.to_ascii_uppercase() {
        'A' => 0,
        'C' => 1,
        'D' => 2,
        'E' => 3,
        'F' => 4,
        'G' => 5,
        'H' => 6,
        'I' => 7,
        'K' => 8,
        'L' => 9,
        'M' => 10,
        'N' => 11,
        'P' => 12,
        'Q' => 13,
        'R' => 14,
        'S' => 15,
        'T' => 16,
        'V' => 17,
        'W' => 18,
        'Y' => 19,
        _ => 20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_alignment_state_creation() {
        let sequences = vec!["ACGTACGT".to_string(), "ACGTACGT".to_string()];
        let profile = ProfileAlignmentState::new(sequences).unwrap();
        assert_eq!(profile.sequences.len(), 2);
        assert_eq!(profile.weights.len(), 2);
    }

    #[test]
    fn test_aa_to_index() {
        assert_eq!(aa_to_index('A'), 0);
        assert_eq!(aa_to_index('a'), 0);
        assert_eq!(aa_to_index('Z'), 20);
    }

    #[test]
    fn test_consensus_computation() {
        let sequences = vec!["ACGT".to_string(), "ACGT".to_string()];
        let mut profile = ProfileAlignmentState::new(sequences).unwrap();
        profile.compute_consensus(0.5);
        assert!(!profile.consensus.is_empty());
    }

    #[test]
    fn test_pssm_building() {
        let sequences = vec!["ACGT".to_string(), "ACGT".to_string()];
        let mut profile = ProfileAlignmentState::new(sequences).unwrap();
        profile.build_pssm().unwrap();
        assert_eq!(profile.pssm.len(), 4);
    }

    #[test]
    fn test_profile_to_sequence_alignment() {
        let sequences = vec!["ACGT".to_string()];
        let profile = ProfileAlignmentState::new(sequences).unwrap();
        let result = profile.align_profile_to_sequence("ACGT", 1.0, 0.5);
        assert!(result.score.is_finite());
    }
}
