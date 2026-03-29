//! Probabilistic phylogeny with Maximum Likelihood (ML) tree building
//! Supports Jukes-Cantor, Kimura, and GTR substitution models

use std::collections::HashMap;
use crate::error::Result;

/// Substitution model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstitutionModel {
    /// Jukes-Cantor: single parameter (α)
    JukesCantor,
    /// Kimura 2-Parameter: transition/transversion ratio
    Kimura2P,
    /// General Time Reversible: 6 rate parameters
    GTR,
    /// HKY (Hasegawa-Kishino-Yano): hybrid model
    HKY,
}

/// Phylogenetic likelihood tree builder
#[derive(Debug, Clone)]
pub struct LikelihoodTreeBuilder {
    /// Model type
    pub model: SubstitutionModel,
    /// Rate matrix (Q matrix)
    pub rate_matrix: Vec<Vec<f64>>,
    /// Transition probabilities cache: (edge_length, matrix)
    pub p_matrix_cache: Vec<(f64, Vec<Vec<f64>>)>,
    /// Edge lengths
    pub edge_lengths: HashMap<String, f64>,
    /// Likelihood score
    pub likelihood: f64,
}

/// Jukes-Cantor substitution model parameters
#[derive(Debug, Clone)]
pub struct JukesCantor {
    /// Rate parameter (α)
    pub alpha: f64,
}

/// Kimura 2-Parameter model
#[derive(Debug, Clone)]
pub struct Kimura2P {
    /// Transition rate (A ↔ G, C ↔ T)
    pub transition_rate: f64,
    /// Transversion rate (A ↔ C, A ↔ T, G ↔ C, G ↔ T)
    pub transversion_rate: f64,
}

/// GTR (General Time Reversible) model
#[derive(Debug, Clone)]
pub struct GTR {
    /// Rate parameters: AC, AG, AT, CG, CT, GT
    pub rates: [f64; 6],
    /// Base frequencies
    pub frequencies: [f64; 4],
}

impl LikelihoodTreeBuilder {
    /// Create new likelihood tree builder with model
    pub fn new(model: SubstitutionModel) -> Result<Self> {
        let rate_matrix = match model {
            SubstitutionModel::JukesCantor => Self::jukes_cantor_matrix(),
            SubstitutionModel::Kimura2P => Self::kimura2p_matrix(),
            SubstitutionModel::GTR => Self::gtr_matrix(),
            SubstitutionModel::HKY => Self::hky_matrix(),
        };

        Ok(LikelihoodTreeBuilder {
            model,
            rate_matrix,
            p_matrix_cache: vec![],
            edge_lengths: HashMap::new(),
            likelihood: 0.0,
        })
    }

    /// Get Jukes-Cantor rate matrix (uniform base frequencies, single rate)
    fn jukes_cantor_matrix() -> Vec<Vec<f64>> {
        // JC model: all rates equal, rate = α
        let alpha = 1.0;
        let beta = -alpha / 3.0;

        vec![
            vec![beta, alpha / 3.0, alpha / 3.0, alpha / 3.0],
            vec![alpha / 3.0, beta, alpha / 3.0, alpha / 3.0],
            vec![alpha / 3.0, alpha / 3.0, beta, alpha / 3.0],
            vec![alpha / 3.0, alpha / 3.0, alpha / 3.0, beta],
        ]
    }

    /// Get Kimura 2-Parameter rate matrix
    fn kimura2p_matrix() -> Vec<Vec<f64>> {
        // K2P: transition rate (κ) and transversion rate (1)
        let kappa = 2.0; // Transition/transversion ratio
        let beta = -(2.0 * kappa + 1.0) / 4.0;

        vec![
            vec![beta, kappa, 1.0, 1.0],        // A
            vec![kappa, beta, 1.0, 1.0],        // C
            vec![1.0, 1.0, beta, kappa],        // G
            vec![1.0, 1.0, kappa, beta],        // T
        ]
    }

    /// Get GTR rate matrix (most complex)
    fn gtr_matrix() -> Vec<Vec<f64>> {
        // GTR uses 6 parameters: rAC, rAG, rAT, rCG, rCT, rGT
        // With base frequencies: πA, πC, πG, πT
        // Simplified version with default parameters
        let rAC = 1.0;
        let rAG = 5.0;
        let rAT = 1.0;
        let rCG = 1.0;
        let rCT = 10.0;
        let rGT = 1.0;

        let pi = [0.25, 0.25, 0.25, 0.25]; // Uniform base frequencies

        let beta = -(rAC * pi[1] + rAG * pi[2] + rAT * pi[3]) / pi[0];

        vec![
            vec![beta, rAC * pi[1], rAG * pi[2], rAT * pi[3]],
            vec![rAC * pi[0], -(rAC * pi[0] + rCG * pi[2] + rCT * pi[3]) / pi[1], rCG * pi[2], rCT * pi[3]],
            vec![rAG * pi[0], rCG * pi[1], -(rAG * pi[0] + rCG * pi[1] + rGT * pi[3]) / pi[2], rGT * pi[3]],
            vec![rAT * pi[0], rCT * pi[1], rGT * pi[2], -(rAT * pi[0] + rCT * pi[1] + rGT * pi[2]) / pi[3]],
        ]
    }

    /// Get HKY model rate matrix
    fn hky_matrix() -> Vec<Vec<f64>> {
        // HKY: like K2P but allows base frequency variation
        let kappa = 2.0;
        let pi = [0.25, 0.25, 0.25, 0.25];

        let beta_a = -(kappa * pi[2] + pi[1] + pi[3]) / pi[0];
        let beta_c = -(pi[0] + pi[2] + kappa * pi[3]) / pi[1];
        let beta_g = -(kappa * pi[0] + pi[1] + pi[3]) / pi[2];
        let beta_t = -(pi[0] + kappa * pi[1] + pi[2]) / pi[3];

        vec![
            vec![beta_a, pi[1], kappa * pi[2], pi[3]],
            vec![pi[0], beta_c, pi[2], kappa * pi[3]],
            vec![kappa * pi[0], pi[1], beta_g, pi[3]],
            vec![pi[0], kappa * pi[1], pi[2], beta_t],
        ]
    }

    /// Compute transition probability matrix for given edge length (time)
    pub fn p_matrix(&mut self, t: f64) -> Result<Vec<Vec<f64>>> {
        // Check cache first - linear search for f64 value
        for (cached_t, p) in &self.p_matrix_cache {
            if (cached_t - t).abs() < 1e-10 {
                return Ok(p.clone());
            }
        }

        // For JC: P(t) = 1/4 + 3/4 * exp(-4αt/3)
        let mut p = vec![vec![0.0; 4]; 4];

        match self.model {
            SubstitutionModel::JukesCantor => {
                let exp_term = (-4.0 * t / 3.0).exp();
                let diag = 0.25 + 0.75 * exp_term;
                let off_diag = 0.25 - 0.25 * exp_term;

                for i in 0..4 {
                    for j in 0..4 {
                        p[i][j] = if i == j { diag } else { off_diag };
                    }
                }
            }
            SubstitutionModel::Kimura2P => {
                // K2P transition probabilities
                let kappa = 2.0;
                let exp_term1 = (-t * (kappa + 2.0) / 4.0).exp();
                let exp_term2 = (-t / 2.0).exp();

                for i in 0..4 {
                    for j in 0..4 {
                        if i == j {
                            p[i][j] = 0.25 + 0.25 * exp_term2 + 0.5 * exp_term1;
                        } else if (i == 0 && j == 2) || (i == 1 && j == 3) || 
                                  (i == 2 && j == 0) || (i == 3 && j == 1) {
                            // Transitions
                            p[i][j] = 0.25 + 0.25 * exp_term2 - 0.5 * exp_term1;
                        } else {
                            // Transversions
                            p[i][j] = 0.25 - 0.25 * exp_term2;
                        }
                    }
                }
            }
            _ => {
                // For other models, use matrix exponential approximation
                for i in 0..4 {
                    for j in 0..4 {
                        if i == j {
                            p[i][j] = 1.0 + t * self.rate_matrix[i][j];
                        } else {
                            p[i][j] = t * self.rate_matrix[i][j];
                        }
                    }
                }
            }
        }

        self.p_matrix_cache.push((t, p.clone()));
        Ok(p)
    }

    /// Compute log-likelihood of sequences under the model
    pub fn likelihood_score(&mut self, seq1: &str, seq2: &str, edge_length: f64) -> Result<f64> {
        let p = self.p_matrix(edge_length)?;

        let mut log_likelihood = 0.0;

        for (c1, c2) in seq1.chars().zip(seq2.chars()) {
            let idx1 = nucleotide_to_index(c1);
            let idx2 = nucleotide_to_index(c2);

            if idx1 < 4 && idx2 < 4 {
                if p[idx1][idx2] > 0.0 {
                    log_likelihood += p[idx1][idx2].ln();
                }
            }
        }

        self.likelihood = log_likelihood;
        Ok(log_likelihood)
    }

    /// Optimize edge length using golden section search
    pub fn optimize_edge_length(
        &mut self,
        seq1: &str,
        seq2: &str,
    ) -> Result<f64> {
        let mut lower = 0.0001;
        let mut upper = 1.0;

        // Golden ratio
        let phi = 0.381966;

        for _ in 0..10 {
            let x1 = lower + (1.0 - phi) * (upper - lower);
            let x2 = lower + phi * (upper - lower);

            let l1 = self.likelihood_score(seq1, seq2, x1)?;
            let l2 = self.likelihood_score(seq1, seq2, x2)?;

            if l1 > l2 {
                upper = x2;
            } else {
                lower = x1;
            }
        }

        let optimal = (lower + upper) / 2.0;
        self.edge_lengths.insert(format!("{}_{}", seq1, seq2), optimal);
        Ok(optimal)
    }

    /// Get model name
    pub fn model_name(&self) -> &'static str {
        match self.model {
            SubstitutionModel::JukesCantor => "Jukes-Cantor",
            SubstitutionModel::Kimura2P => "Kimura 2-Parameter",
            SubstitutionModel::GTR => "General Time Reversible",
            SubstitutionModel::HKY => "HKY",
        }
    }
}

/// Convert nucleotide character to matrix index (0=A, 1=C, 2=G, 3=T)
fn nucleotide_to_index(c: char) -> usize {
    match c.to_ascii_uppercase() {
        'A' => 0,
        'C' => 1,
        'G' => 2,
        'T' => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likelihood_builder_jc_creation() {
        let builder = LikelihoodTreeBuilder::new(SubstitutionModel::JukesCantor).unwrap();
        assert_eq!(builder.model_name(), "Jukes-Cantor");
        assert_eq!(builder.rate_matrix.len(), 4);
    }

    #[test]
    fn test_likelihood_builder_kimura_creation() {
        let builder = LikelihoodTreeBuilder::new(SubstitutionModel::Kimura2P).unwrap();
        assert_eq!(builder.model_name(), "Kimura 2-Parameter");
        assert!(!builder.rate_matrix.is_empty());
    }

    #[test]
    fn test_likelihood_builder_gtr_creation() {
        let builder = LikelihoodTreeBuilder::new(SubstitutionModel::GTR).unwrap();
        assert_eq!(builder.model_name(), "General Time Reversible");
    }

    #[test]
    fn test_jukes_cantor_pmatrix() {
        let mut builder = LikelihoodTreeBuilder::new(SubstitutionModel::JukesCantor).unwrap();
        let p = builder.p_matrix(0.1).unwrap();
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].len(), 4);
        // Diagonal should be > off-diagonal
        assert!(p[0][0] > p[0][1]);
    }

    #[test]
    fn test_kimura_pmatrix() {
        let mut builder = LikelihoodTreeBuilder::new(SubstitutionModel::Kimura2P).unwrap();
        let p = builder.p_matrix(0.1).unwrap();
        assert_eq!(p.len(), 4);
        // Transitions should be more likely than transversions
        // A->G is transition, A->C is transversion
        assert!(p[0][2] > p[0][1]); // More likely to transition than transvert
    }

    #[test]
    fn test_pmatrix_caching() {
        let mut builder = LikelihoodTreeBuilder::new(SubstitutionModel::JukesCantor).unwrap();
        let p1 = builder.p_matrix(0.1).unwrap();
        assert_eq!(builder.p_matrix_cache.len(), 1);
        
        let p2 = builder.p_matrix(0.1).unwrap();
        assert_eq!(p1, p2); // Should be identical
        assert_eq!(builder.p_matrix_cache.len(), 1); // Still 1 entry - cached
    }

    #[test]
    fn test_likelihood_score_identical_sequences() {
        let mut builder = LikelihoodTreeBuilder::new(SubstitutionModel::JukesCantor).unwrap();
        let score = builder.likelihood_score("ACGT", "ACGT", 0.01).unwrap();
        // Identical sequences should have high score
        assert!(score.is_finite());
    }

    #[test]
    fn test_likelihood_score_different_sequences() {
        let mut builder = LikelihoodTreeBuilder::new(SubstitutionModel::JukesCantor).unwrap();
        let score1 = builder.likelihood_score("ACGT", "ACGT", 0.01).unwrap();
        let score2 = builder.likelihood_score("ACGT", "TGCA", 0.01).unwrap();
        // Identical should score better than different
        assert!(score1 > score2);
    }

    #[test]
    fn test_edge_length_optimization() {
        let mut builder = LikelihoodTreeBuilder::new(SubstitutionModel::JukesCantor).unwrap();
        let optimal = builder.optimize_edge_length("ACGT", "AGGT").unwrap();
        assert!(optimal > 0.0);
        assert!(optimal < 1.0);
        assert_eq!(builder.edge_lengths.len(), 1);
    }

    #[test]
    fn test_nucleotide_to_index() {
        assert_eq!(nucleotide_to_index('A'), 0);
        assert_eq!(nucleotide_to_index('C'), 1);
        assert_eq!(nucleotide_to_index('G'), 2);
        assert_eq!(nucleotide_to_index('T'), 3);
        assert_eq!(nucleotide_to_index('a'), 0); // lowercase
        assert_eq!(nucleotide_to_index('X'), 4); // invalid
    }

    #[test]
    fn test_hky_model_matrix() {
        let builder = LikelihoodTreeBuilder::new(SubstitutionModel::HKY).unwrap();
        assert_eq!(builder.model_name(), "HKY");
        assert_eq!(builder.rate_matrix.len(), 4);
    }
}
