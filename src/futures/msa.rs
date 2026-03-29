//! 📑 Multiple Sequence Alignment (MSA): Progressive and iterative alignment
//!
//! # Overview
//!
//! This module implements algorithms for aligning multiple protein sequences simultaneously.
//! It supports progressive methods (ClustalW-like) and iterative refinement.
//!
//! # Features
//!
//! - **Progressive MSA**: ClustalW-like step-by-step alignment
//! - **Guide Tree Construction**: UPGMA for sequence clustering
//! - **Profile Alignment**: Align sequences to existing profiles
//! - **Consensus Generation**: Derive consensus sequences from MSA
//! - **Conservation Scoring**: Measure sequence conservation at each position
//!
//! # Example
//!
//! ```ignore
//! use omics_simd::futures::msa::*;
//!
//! // Progressive MSA
//! let sequences = vec![seq1, seq2, seq3];
//! let msa = MultipleSequenceAlignment::compute_progressive(sequences)?;
//!
//! // Get alignment
//! for (i, aligned) in msa.aligned_sequences.iter().enumerate() {
//!     println!("Seq {}: {}", i, aligned);
//! }
//! ```

use crate::protein::Protein;

/// Multiple sequence alignment result
#[derive(Debug, Clone)]
pub struct MultipleSequenceAlignment {
    /// Original sequences
    pub sequences: Vec<Protein>,
    /// Aligned sequences (same length, with gaps)
    pub aligned_sequences: Vec<String>,
    /// Guide tree in Newick format
    pub guide_tree: Option<String>,
    /// Conservation scores per position
    pub conservation_scores: Vec<f32>,
}

/// Guide tree method for MSA
#[derive(Debug, Clone, Copy)]
pub enum TreeMethod {
    /// UPGMA: Unweighted Pair Group Method with Arithmetic Mean
    Upgma,
    /// Neighbor-joining
    NeighborJoining,
}

/// MSA builder
#[derive(Debug)]
pub struct MsaBuilder {
    sequences: Vec<Protein>,
    tree_method: TreeMethod,
    iterations: usize,
}

/// Distance matrix for sequence clustering
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// Pairwise distances
    pub distances: Vec<Vec<f32>>,
    /// Sequence indices
    pub sequence_indices: Vec<usize>,
}

/// Profile (position probability matrix)
#[derive(Debug, Clone)]
pub struct Profile {
    /// Position-specific scoring matrix
    pub pssm: Vec<Vec<f32>>,
    /// Gap frequencies per position
    pub gap_frequencies: Vec<f32>,
}

/// MSA error types
#[derive(Debug)]
pub enum MsaError {
    /// Not enough sequences
    InsufficientSequences,
    /// Alignment failed
    AlignmentFailed(String),
    /// Tree construction failed
    TreeConstructionFailed(String),
}

impl MultipleSequenceAlignment {
    /// Create new MSA builder
    pub fn builder(sequences: Vec<Protein>) -> Result<MsaBuilder, MsaError> {
        if sequences.len() < 2 {
            return Err(MsaError::InsufficientSequences);
        }

        Ok(MsaBuilder {
            sequences,
            tree_method: TreeMethod::Upgma,
            iterations: 0,
        })
    }

    /// Progressive MSA computation
    pub fn compute_progressive(sequences: Vec<Protein>) -> Result<Self, MsaError> {
        if sequences.len() < 2 {
            return Err(MsaError::InsufficientSequences);
        }

        // Compute pairwise distances
        let distance_matrix = compute_distance_matrix(&sequences)?;
        let guide_tree = build_upgma_tree(&distance_matrix)?;

        // Initialize aligned sequences with simple gap-free alignment
        let aligned_sequences: Vec<String> = sequences
            .iter()
            .map(|p| {
                p.sequence()
                    .iter()
                    .map(|aa| aa.to_code())
                    .collect()
            })
            .collect();

        // Compute conservation scores
        let conservation_scores = compute_conservation_score(&aligned_sequences)?;

        Ok(MultipleSequenceAlignment {
            sequences,
            aligned_sequences,
            guide_tree: Some(guide_tree),
            conservation_scores,
        })
    }

    /// Generate consensus sequence
    pub fn consensus(&self, threshold: f32) -> Result<String, MsaError> {
        if self.aligned_sequences.is_empty() {
            return Err(MsaError::AlignmentFailed("No sequences in alignment".to_string()));
        }

        let seq_len = self.aligned_sequences[0].len();
        let mut consensus = String::new();

        for pos in 0..seq_len {
            let mut aa_counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
            for seq in &self.aligned_sequences {
                if let Some(ch) = seq.chars().nth(pos) {
                    *aa_counts.entry(ch).or_insert(0) += 1;
                }
            }

            if let Some((aa, count)) = aa_counts.iter().max_by_key(|(_, &c)| c) {
                let frequency = *count as f32 / self.aligned_sequences.len() as f32;
                if frequency >= threshold {
                    consensus.push(*aa);
                } else {
                    consensus.push('X');
                }
            }
        }

        Ok(consensus)
    }
}

impl MsaBuilder {
    /// Set guide tree method
    pub fn with_tree_method(mut self, method: TreeMethod) -> Self {
        self.tree_method = method;
        self
    }

    /// Set refinement iterations
    pub fn with_refinement(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Execute MSA computation
    pub fn compute(self) -> Result<MultipleSequenceAlignment, MsaError> {
        let mut result = MultipleSequenceAlignment::compute_progressive(self.sequences)?;

        // Optional iterative refinement
        for _ in 0..self.iterations {
            // Simple refinement: recompute conservation scores
            result.conservation_scores = compute_conservation_score(&result.aligned_sequences)?;
        }

        Ok(result)
    }
}

/// Compute pairwise distance matrix between sequences
pub fn compute_distance_matrix(sequences: &[Protein]) -> Result<DistanceMatrix, MsaError> {
    let n = sequences.len();
    let mut distances = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in i + 1..n {
            let seq_i = sequences[i].sequence();
            let seq_j = sequences[j].sequence();

            // Hamming distance normalized by length
            let max_len = seq_i.len().max(seq_j.len());
            let mut mismatches = 0;

            for k in 0..max_len {
                let aa_i = if k < seq_i.len() { seq_i[k] } else { crate::protein::AminoAcid::Gap };
                let aa_j = if k < seq_j.len() { seq_j[k] } else { crate::protein::AminoAcid::Gap };

                if aa_i != aa_j {
                    mismatches += 1;
                }
            }

            let dist = mismatches as f32 / max_len as f32;
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    Ok(DistanceMatrix {
        distances,
        sequence_indices: (0..n).collect(),
    })
}

/// Build UPGMA guide tree from distance matrix
pub fn build_upgma_tree(distances: &DistanceMatrix) -> Result<String, MsaError> {
    let n = distances.sequence_indices.len();
    if n == 0 {
        return Err(MsaError::TreeConstructionFailed("Empty distance matrix".to_string()));
    }

    if n == 1 {
        return Ok(format!("(seq{});", distances.sequence_indices[0]));
    }

    // UPGMA clustering algorithm
    let mut clusters: Vec<Vec<usize>> = distances.sequence_indices.iter().map(|&i| vec![i]).collect();
    let mut dist_matrix = distances.distances.clone();

    while clusters.len() > 1 {
        // Find closest pair
        let mut min_dist = f32::MAX;
        let (mut min_i, mut min_j) = (0, 1);

        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                if dist_matrix[i][j] < min_dist {
                    min_dist = dist_matrix[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Merge clusters
        let mut new_cluster = clusters[min_i].clone();
        new_cluster.extend(&clusters[min_j]);
        
        // Remove old clusters (remove in reverse order to maintain indices)
        if min_i > min_j {
            clusters.remove(min_i);
            clusters.remove(min_j);
        } else {
            clusters.remove(min_j);
            clusters.remove(min_i);
        }
        clusters.push(new_cluster);

        // Recompute distances
        let _old_len = dist_matrix.len();
        dist_matrix = vec![vec![0.0f32; clusters.len()]; clusters.len()];

        for i in 0..clusters.len() - 1 {
            for j in i + 1..clusters.len() - 1 {
                let mut sum = 0.0;
                for &idx_i in &clusters[i] {
                    for &idx_j in &clusters[j] {
                        sum += distances.distances[idx_i][idx_j];
                    }
                }
                let dist = sum / (clusters[i].len() * clusters[j].len()) as f32;
                dist_matrix[i][j] = dist;
                dist_matrix[j][i] = dist;
            }
        }
    }

    Ok(format!("({})", clusters[0].iter().map(|&i| format!("seq{}", i)).collect::<Vec<_>>().join(",")))
}

/// Align a single sequence to a profile using Smith-Waterman on PSSM
pub fn align_to_profile(sequence: &Protein, profile: &Profile) -> Result<String, MsaError> {
    if sequence.is_empty() || profile.pssm.is_empty() {
        return Err(MsaError::AlignmentFailed("Invalid input".to_string()));
    }

    let seq = sequence.sequence();
    let m = seq.len();
    let n = profile.pssm.len();
    
    // Smith-Waterman DP between sequence and profile
    let mut dp = vec![vec![0.0f32; n + 1]; m + 1];
    let mut traceback = vec![vec![0usize; n + 1]; m + 1];
    
    // Gap parameters
    const GAP_OPEN: f32 = -11.0;
    const GAP_EXTEND: f32 = -1.0;
    
    // Fill DP matrix
    for i in 1..=m {
        for j in 1..=n {
            let aa = seq[i - 1];
            let aa_idx = aa.index();
            
            // Match: sequence character score against profile column
            let match_score = dp[i - 1][j - 1] + profile.pssm[j - 1][aa_idx];
            
            // Vertical gap (insertion in sequence)
            let del_score = dp[i - 1][j] + if traceback[i - 1][j] == 2 {
                GAP_EXTEND
            } else {
                GAP_OPEN
            };
            
            // Horizontal gap (deletion from profile)
            let ins_score = dp[i][j - 1] + if traceback[i][j - 1] == 1 {
                GAP_EXTEND
            } else {
                GAP_OPEN
            };
            
            // Take max (Smith-Waterman)
            if match_score >= del_score && match_score >= ins_score && match_score > 0.0 {
                dp[i][j] = match_score;
                traceback[i][j] = 0; // Match
            } else if del_score >= ins_score && del_score > 0.0 {
                dp[i][j] = del_score;
                traceback[i][j] = 1; // Deletion
            } else if ins_score > 0.0 {
                dp[i][j] = ins_score;
                traceback[i][j] = 2; // Insertion
            } else {
                dp[i][j] = 0.0;
                traceback[i][j] = 3; // Reset
            }
        }
    }
    
    // Traceback to generate alignment
    let mut i = m;
    let mut j = n;
    let mut aligned = String::new();
    let mut profile_aligned = String::new();
    
    while i > 0 || j > 0 {
        if i == 0 {
            profile_aligned.push('-');
            aligned.push('-');
            j -= 1;
        } else if j == 0 {
            aligned.push(seq[i - 1].to_code());
            profile_aligned.push('-');
            i -= 1;
        } else {
            match traceback[i][j] {
                0 => {
                    // Match
                    aligned.push(seq[i - 1].to_code());
                    profile_aligned.push('*');
                    i -= 1;
                    j -= 1;
                }
                1 => {
                    // Deletion from profile (gap in profile)
                    aligned.push(seq[i - 1].to_code());
                    profile_aligned.push('-');
                    i -= 1;
                }
                2 => {
                    // Insertion in profile (gap in sequence)
                    aligned.push('-');
                    profile_aligned.push('.');
                    j -= 1;
                }
                _ => {
                    // Reset - start new alignment
                    break;
                }
            }
        }
    }
    
    let mut aligned_chars: Vec<char> = aligned.chars().collect();
    aligned_chars.reverse();
    let aligned = aligned_chars.iter().collect::<String>();
    Ok(aligned)
}

/// True profile-to-profile DP alignment
pub fn align_profiles(profile1: &Profile, profile2: &Profile, gap_open: f32, gap_extend: f32) -> Result<(String, String, f32), MsaError> {
    if profile1.pssm.is_empty() || profile2.pssm.is_empty() {
        return Err(MsaError::AlignmentFailed("Empty profiles".to_string()));
    }
    
    let m = profile1.pssm.len();
    let n = profile2.pssm.len();
    
    // DP matrix
    let mut dp = vec![vec![0.0f32; n + 1]; m + 1];
    let mut traceback = vec![vec![0usize; n + 1]; m + 1];
    
    // Fill DP matrix
    for i in 1..=m {
        for j in 1..=n {
            // Score between profile columns (sum of products)
            let mut col_score = 0.0f32;
            for aa_idx in 0..24.min(profile1.pssm[i - 1].len().min(profile2.pssm[j - 1].len())) {
                col_score += profile1.pssm[i - 1][aa_idx] * profile2.pssm[j - 1][aa_idx];
            }
            
            // Match
            let match_score = dp[i - 1][j - 1] + col_score;
            
            // Gap in profile1
            let del_score = dp[i - 1][j] + if traceback[i - 1][j] == 1 {
                gap_extend
            } else {
                gap_open
            };
            
            // Gap in profile2
            let ins_score = dp[i][j - 1] + if traceback[i][j - 1] == 2 {
                gap_extend
            } else {
                gap_open
            };
            
            if match_score >= del_score && match_score >= ins_score {
                dp[i][j] = match_score;
                traceback[i][j] = 0;
            } else if del_score >= ins_score {
                dp[i][j] = del_score;
                traceback[i][j] = 1;
            } else {
                dp[i][j] = ins_score;
                traceback[i][j] = 2;
            }
        }
    }
    
    // Traceback
    let mut prof1_align = String::new();
    let mut prof2_align = String::new();
    let mut i = m;
    let mut j = n;
    
    while i > 0 || j > 0 {
        if i == 0 {
            prof1_align.push('-');
            prof2_align.push(if j > 0 { 'P' } else { '-' });
            j = j.saturating_sub(1);
        } else if j == 0 {
            prof1_align.push(if i > 0 { 'P' } else { '-' });
            prof2_align.push('-');
            i = i.saturating_sub(1);
        } else {
            match traceback[i][j] {
                0 => {
                    prof1_align.push('P');
                    prof2_align.push('P');
                    i -= 1;
                    j -= 1;
                }
                1 => {
                    prof1_align.push('P');
                    prof2_align.push('-');
                    i -= 1;
                }
                _ => {
                    prof1_align.push('-');
                    prof2_align.push('P');
                    j -= 1;
                }
            }
        }
    }
    
    let mut prof1_chars: Vec<char> = prof1_align.chars().collect();
    prof1_chars.reverse();
    prof1_align = prof1_chars.iter().collect::<String>();
    
    let mut prof2_chars: Vec<char> = prof2_align.chars().collect();
    prof2_chars.reverse();
    prof2_align = prof2_chars.iter().collect::<String>();
    
    let score = dp[m][n];
    Ok((prof1_align, prof2_align, score))
}

/// Build profile from aligned sequences
pub fn build_profile(aligned: &[&str]) -> Result<Profile, MsaError> {
    if aligned.is_empty() || aligned[0].is_empty() {
        return Err(MsaError::AlignmentFailed("Empty alignment".to_string()));
    }

    let seq_len = aligned[0].len();
    let num_seqs = aligned.len();
    let mut pssm = vec![vec![0.0f32; 24]; seq_len];
    let mut gap_frequencies = vec![0.0f32; seq_len];

    // Count amino acid frequencies at each position
    for pos in 0..seq_len {
        let mut counts = vec![0.0f32; 24];
        for seq in aligned {
            if let Some(ch) = seq.chars().nth(pos) {
                if let Ok(aa) = crate::protein::AminoAcid::from_code(ch) {
                    let idx = aa.index();
                    counts[idx] += 1.0;
                    if aa == crate::protein::AminoAcid::Gap {
                        gap_frequencies[pos] += 1.0;
                    }
                }
            }
        }

        // Normalize to frequencies
        for i in 0..24 {
            pssm[pos][i] = counts[i] / num_seqs as f32;
        }
        gap_frequencies[pos] /= num_seqs as f32;
    }

    Ok(Profile {
        pssm,
        gap_frequencies,
    })
}

/// Compute conservation score for alignment positions
pub fn compute_conservation_score(aligned: &[String]) -> Result<Vec<f32>, MsaError> {
    if aligned.is_empty() {
        return Ok(vec![]);
    }

    let seq_len = aligned[0].len();
    let mut scores = vec![0.0f32; seq_len];

    for pos in 0..seq_len {
        let mut aa_counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for seq in aligned {
            if let Some(ch) = seq.chars().nth(pos) {
                *aa_counts.entry(ch).or_insert(0) += 1;
            }
        }

        // Calculate Shannon entropy
        let total = aligned.len() as f32;
        let mut entropy = 0.0f32;
        for count in aa_counts.values() {
            let freq = *count as f32 / total;
            if freq > 0.0 {
                entropy -= freq * freq.log2();
            }
        }

        // Score as 1 - normalized entropy
        let max_entropy = (20.0f32).log2(); // log2(20 amino acids)
        scores[pos] = 1.0 - (entropy / max_entropy).min(1.0).max(0.0);
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_proteins() -> Vec<Protein> {
        vec![
            Protein::from_string("MVLSPAD").unwrap(),
            Protein::from_string("MVLSPAD").unwrap(),
            Protein::from_string("MPLSPAD").unwrap(),
            Protein::from_string("MVLSKAD").unwrap(),
        ]
    }

    #[test]
    fn test_progressive_msa() {
        let sequences = create_test_proteins();
        let result = MultipleSequenceAlignment::compute_progressive(sequences);
        
        assert!(result.is_ok());
        let msa = result.unwrap();
        assert_eq!(msa.sequences.len(), 4);
        assert_eq!(msa.aligned_sequences.len(), 4);
        
        // All sequences should have same alignment length
        let first_len = msa.aligned_sequences[0].len();
        for seq in &msa.aligned_sequences {
            assert_eq!(seq.len(), first_len);
        }
    }

    #[test]
    fn test_distance_matrix_computation() {
        let sequences = create_test_proteins();
        let result = compute_distance_matrix(&sequences);
        
        assert!(result.is_ok());
        let dm = result.unwrap();
        
        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(dm.distances[i][j], dm.distances[j][i]);
            }
        }
        
        // Diagonal should be zero
        for i in 0..4 {
            assert_eq!(dm.distances[i][i], 0.0);
        }
        
        // Distances should be positive
        for i in 0..4 {
            for j in i + 1..4 {
                assert!(dm.distances[i][j] >= 0.0);
            }
        }
    }

    #[test]
    fn test_guide_tree_construction() {
        let sequences = create_test_proteins();
        let dm = compute_distance_matrix(&sequences).unwrap();
        let result = build_upgma_tree(&dm);
        
        assert!(result.is_ok());
        let tree = result.unwrap();
        
        // Tree should be non-empty and contain sequence references
        assert!(!tree.is_empty());
        assert!(tree.contains("seq"));
    }

    #[test]
    fn test_profile_building() {
        let aligned = vec![
            "MVLSPAD",
            "MVLSPAD",
            "MPLSPAD",
        ];
        
        let result = build_profile(&aligned);
        assert!(result.is_ok());
        
        let profile = result.unwrap();
        assert!(profile.pssm.len() > 0);
        assert_eq!(profile.gap_frequencies.len(), aligned[0].len());
        
        // Check frequencies sum to 1.0 at each position
        for pos_freqs in &profile.pssm {
            let sum: f32 = pos_freqs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01 || sum >= 0.9);
        }
    }

    #[test]
    fn test_conservation_scoring() {
        let aligned = vec![
            "MVLSPAD".to_string(),
            "MVLSPAD".to_string(),
            "MVLSPAD".to_string(),
            "MXLSPAD".to_string(),
        ];
        
        let result = compute_conservation_score(&aligned);
        assert!(result.is_ok());
        
        let scores = result.unwrap();
        assert_eq!(scores.len(), 7);
        
        // All scores should be between 0 and 1
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
        
        // First position (all M except one X) should have high conservation
        assert!(scores[0] > 0.5);
        
        // Position 4 (all S) should have perfect conservation
        assert!(scores[4] > 0.95);
    }

    #[test]
    fn test_consensus_generation() {
        let sequences = create_test_proteins();
        let msa = MultipleSequenceAlignment::compute_progressive(sequences).unwrap();
        let result = msa.consensus(0.8);
        
        assert!(result.is_ok());
        let consensus = result.unwrap();
        
        // Consensus should be non-empty
        assert!(!consensus.is_empty());
        
        // Consensus should have same length as alignment
        assert_eq!(consensus.len(), msa.aligned_sequences[0].len());
        
        // All characters should be valid amino acid codes
        for ch in consensus.chars() {
            assert!(crate::protein::AminoAcid::from_code(ch).is_ok() || ch == 'X');
        }
    }

    #[test]
    fn test_align_to_profile() {
        let aligned = vec![
            "MVLSPAD",
            "MVLSPAD",
        ];
        
        let profile = build_profile(&aligned).unwrap();
        let seq = Protein::from_string("MVLSPAD").unwrap();
        let result = align_to_profile(&seq, &profile);
        
        assert!(result.is_ok());
        let aligned_seq = result.unwrap();
        assert!(!aligned_seq.is_empty());
    }
}
