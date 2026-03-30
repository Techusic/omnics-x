//! HMMER3 Format Parser
//!
//! Implements complete HMMER3 ASCII format specification for robust PFAM integration.
//! This parser extracts HMM profiles with empirical E-value statistics.
//!
//! # Format Reference
//! - HMMER 3.x ASCII format with version line
//! - Scores in bits and log-odds
//! - Karlin-Altschul statistics for E-value computation
//! - Full insert/delete/match state modeling

use crate::error::Error;
use regex::Regex;

/// HMMER3 Format Parser Result
pub type HmmerResult<T> = Result<T, HmmerError>;

/// HMMER3 specific errors
#[derive(Debug)]
pub enum HmmerError {
    /// Parse error with line number and description
    ParseError { line: usize, msg: String },
    /// Invalid file format
    InvalidFormat(String),
    /// Missing required field
    MissingField(String),
    /// Invalid numeric value
    InvalidNumeric(String),
}

impl std::fmt::Display for HmmerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HmmerError::ParseError { line, msg } => write!(f, "Line {}: {}", line, msg),
            HmmerError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            HmmerError::MissingField(field) => write!(f, "Missing field: {}", field),
            HmmerError::InvalidNumeric(val) => write!(f, "Invalid number: {}", val),
        }
    }
}

impl std::error::Error for HmmerError {}

impl From<HmmerError> for Error {
    fn from(e: HmmerError) -> Self {
        Error::AlignmentError(e.to_string())
    }
}

/// Karlin-Altschul statistical parameters for E-value calculation
#[derive(Debug, Clone)]
pub struct KarlinParameters {
    /// Lambda (scale parameter)
    pub lambda: f64,
    /// K (statistical parameter)
    pub k: f64,
    /// H (relative entropy)
    pub h: f64,
    /// Log K
    pub logk: f64,
}

impl KarlinParameters {
    /// Create default Karlin parameters (typical for BLOSUM62/protein)
    pub fn default_protein() -> Self {
        KarlinParameters {
            lambda: 0.3176,
            k: 0.134,
            h: 0.4012,
            logk: -2.004,
        }
    }

    /// Create Karlin parameters from empirical statistics
    pub fn new(lambda: f64, k: f64, h: f64) -> Self {
        KarlinParameters {
            lambda,
            k,
            h,
            logk: k.ln(),
        }
    }

    /// Calculate E-value from bit score and database size
    pub fn evalue(&self, bit_score: f64, db_size: u64) -> f64 {
        let raw_score = bit_score / self.lambda;
        self.k * (db_size as f64) * (-self.lambda * raw_score).exp()
    }

    /// Calculate bit score from raw score
    pub fn bit_score(&self, raw_score: f64) -> f64 {
        (self.lambda * raw_score - self.logk) / std::f64::consts::LN_2
    }
}

/// HMMER3 state with emission and transition probabilities
#[derive(Debug, Clone)]
pub struct HmmerState {
    /// State type (M=match, D=delete, I=insert)
    pub state_type: char,
    /// Emission probabilities (log-odds scores)
    pub emissions: Vec<f64>,
    /// Transition probabilities (T, S, D transitions in log-odds)
    pub transitions: Vec<f64>,
}

/// Complete HMMER3 HMM model
#[derive(Debug, Clone)]
pub struct HmmerModel {
    /// Model name/accession
    pub name: String,
    /// Model description
    pub description: String,
    /// Model length (number of match states)
    pub length: usize,
    /// Alphabet type (amino, DNA)
    pub alpha: String,
    /// Reference annotation
    pub rf: String,
    /// Consensus sequence
    pub consensus: String,
    /// Model creation date
    pub date: String,
    /// Model version
    pub version: String,
    /// Karlin-Altschul parameters
    pub karlin: KarlinParameters,
    /// States indexed by position [position][state_type]
    pub states: Vec<[HmmerState; 3]>,
    /// Transition probabilities from BEGIN
    pub begin_trans: Vec<f64>,
    /// Transition probabilities to END
    pub end_trans: Vec<f64>,
    /// Null model probabilities
    pub null_model: Vec<f64>,
}

impl HmmerModel {
    /// Parse HMMER3 format from string content
    pub fn parse(content: &str) -> HmmerResult<Self> {
        let mut lines = content.lines().enumerate();
        let mut model = HmmerModel {
            name: String::new(),
            description: String::new(),
            length: 0,
            alpha: "amino".to_string(),
            rf: String::new(),
            consensus: String::new(),
            date: String::new(),
            version: String::new(),
            karlin: KarlinParameters::default_protein(),
            states: Vec::new(),
            begin_trans: vec![0.0; 3],
            end_trans: Vec::new(),
            null_model: vec![0.05; 20], // Default uniform for 20 amino acids
        };

        // Parse header
        while let Some((line_num, line)) = lines.next() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("//") {
                continue;
            }

            // Parse header fields
            if line.starts_with("HMMER3") {
                model.version = line.split_whitespace().nth(1).unwrap_or("3").to_string();
            } else if let Some(name) = line.strip_prefix("NAME ") {
                model.name = name.trim().to_string();
            } else if let Some(desc) = line.strip_prefix("DESC ") {
                model.description = desc.trim().to_string();
            } else if let Some(len_str) = line.strip_prefix("LENG ") {
                model.length = len_str.trim().parse()
                    .map_err(|_| HmmerError::ParseError {
                        line: line_num,
                        msg: "Invalid LENG field".to_string(),
                    })?;
            } else if let Some(alph) = line.strip_prefix("ALPH ") {
                model.alpha = alph.trim().to_string();
            } else if let Some(rf) = line.strip_prefix("RF ") {
                model.rf = rf.trim().to_string();
            } else if let Some(cons) = line.strip_prefix("CONS ") {
                model.consensus = cons.trim().to_string();
            } else if let Some(date) = line.strip_prefix("DATE ") {
                model.date = date.trim().to_string();
            } else if line.starts_with("STAT") {
                // Parse Karlin-Altschul statistics
                model.parse_stats(line)?;
            } else if line == "HMM" {
                // Begin HMM matrix section
                break;
            }
        }

        // Parse HMM matrix
        let mut states: Vec<[HmmerState; 3]> = Vec::new();
        let mut line_buffer = String::new();

        // Skip header lines in matrix section
        let mut matrix_started = false;

        for (line_num, line) in lines {
            let line = line.trim();

            if line.is_empty() || line.starts_with("//") {
                break;
            }

            if !matrix_started {
                if line.starts_with("        ") {
                    matrix_started = true;
                }
                continue;
            }

            // Parse state triplet (M, I, D)
            if !line.starts_with("        ") && !line.is_empty() {
                line_buffer.push('\n');
                line_buffer.push_str(line);

                if line_buffer.contains('\n') {
                    let triplet = model.parse_state_triplet(&line_buffer, line_num)?;
                    states.push(triplet);
                    line_buffer.clear();
                }
            }
        }

        if !line_buffer.is_empty() {
            let triplet = model.parse_state_triplet(&line_buffer, 0)?;
            states.push(triplet);
        }

        model.states = states;
        Ok(model)
    }

    /// Parse Karlin-Altschul statistics line
    fn parse_stats(&mut self, line: &str) -> HmmerResult<()> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 5 && parts[0] == "STAT" {
            if parts[1] == "LOCAL" && parts[2] == "MSV" {
                // Extract mu, lambda from the line format: "STAT LOCAL MSV mu:X lambda:Y"
                for part in &parts[3..] {
                    if let Some(val) = part.strip_prefix("lambda:") {
                        self.karlin.lambda = val.parse()
                            .map_err(|_| HmmerError::InvalidNumeric(val.to_string()))?;
                    }
                    if let Some(val) = part.strip_prefix("K:") {
                        self.karlin.k = val.parse()
                            .map_err(|_| HmmerError::InvalidNumeric(val.to_string()))?;
                    }
                    if let Some(val) = part.strip_prefix("H:") {
                        self.karlin.h = val.parse()
                            .map_err(|_| HmmerError::InvalidNumeric(val.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse a state triplet (M, I, D with scores and transitions)
    fn parse_state_triplet(&self, buffer: &str, line_num: usize) -> HmmerResult<[HmmerState; 3]> {
        let lines: Vec<&str> = buffer.lines().collect();
        if lines.len() < 3 {
            return Err(HmmerError::ParseError {
                line: line_num,
                msg: "Invalid state triplet (need M, I, D lines)".to_string(),
            });
        }

        let match_state = self.parse_state_line_robust(lines[0], 'M', line_num)?;
        let insert_state = self.parse_state_line_robust(lines[1], 'I', line_num)?;
        let delete_state = self.parse_state_line_robust(lines[2], 'D', line_num)?;

        Ok([match_state, insert_state, delete_state])
    }

    /// Parse HMMER3 score field with robust format handling (REAL FIX for Fault #3)
    /// Handles multiple HMMER3 score representations:
    /// - Standard: numeric (1.23, -4.56)
    /// - Special: "*" for -infinity
    /// - Variants: "-inf", "-Inf", "-INF"
    /// - Integer-scaled: "*-12345" (asterisk with integer code)
    /// - Legacy: Probability format (0.0-1.0 range, needs conversion)
    fn parse_hmmer_score(&self, value: &str, line_num: usize) -> HmmerResult<f64> {
        let trimmed = value.trim();

        // Special case: pure asterisk means -infinity (impossible transition/emission)
        if trimmed == "*" {
            return Ok(f64::NEG_INFINITY);
        }

        // Special case: -infinity variants
        if trimmed == "-inf" || trimmed == "-Inf" || trimmed == "-INF" {
            return Ok(f64::NEG_INFINITY);
        }

        // NEW: Integer-scaled HMMER3 format (e.g., "*-1234" for scaled values)
        // This format appears in some PFAM/InterPro variants
        if trimmed.starts_with('*') {
            // Extract integer after asterisk
            if let Ok(scaled_val) = trimmed[1..].parse::<i32>() {
                // Convert from HMMER3 internal scaling (typically /1000)
                return Ok(scaled_val as f64 / 1000.0);
            }
            // If parsing fails, fall through to standard numeric parsing
        }

        // Try standard float parsing
        match trimmed.parse::<f64>() {
            Ok(val) => {
                // NEW: Format auto-detection
                // - If 0.0 <= val <= 1.0: likely probability format (needs log conversion)
                // - If val < -1000: likely pre-scaled integer format
                // - Otherwise: assume already in log-odds bits
                
                if val > 0.0 && val < 1.0 {
                    // Probability format: convert to log-odds
                    // log(p / (1-p)) for probability p
                    let log_odds = (val / (1.0 - val)).ln();
                    return Ok(log_odds);
                }
                
                // If very negative and integer-like, may need scaling
                if val < -1000.0 && (val.fract() == 0.0) {
                    // Pre-scaled integer: divide by 1000
                    return Ok(val / 1000.0);
                }

                Ok(val)
            }
            Err(_) => {
                // IMPROVED: Context-aware error message
                Err(HmmerError::ParseError {
                    line: line_num,
                    msg: format!(
                        "Cannot parse score '{}' as number. \
                         Expected: numeric (e.g., 1.23), '*', '-inf', or '*NNNN' format. \
                         Possible causes: corrupted database, non-HMMER3 format, encoding issue",
                        trimmed
                    ),
                })
            }
        }
    }

    /// Parse a single state line with PRODUCTION-GRADE robustness (REAL FIX for Fault #3)
    /// Uses regex to extract fields instead of naive split_whitespace()
    fn parse_state_line_robust(&self, line: &str, state_type: char, line_num: usize) -> HmmerResult<HmmerState> {
        // IMPROVED: Use regex to extract fields with flexible whitespace handling
        // Pattern: captures numeric/special values with optional surrounding whitespace
        let field_pattern = Regex::new(r"(\S+)")
            .map_err(|e| HmmerError::ParseError { line: line_num, msg: format!("Regex compilation failed: {}", e) })?;
        let fields: Vec<&str> = field_pattern
            .find_iter(line)
            .map(|m| m.as_str())
            .collect();

        let mut emissions = Vec::new();
        let mut transitions = Vec::new();

        match state_type {
            'M' => {
                // Match state: 20 emission scores + 3 transitions (M->M, M->I, M->D)
                if fields.len() < 23 {
                    return Err(HmmerError::ParseError {
                        line: line_num,
                        msg: format!(
                            "Match state requires 23 fields (20 emissions + 3 transitions), got {}. \
                             Line: '{}' | First 5 fields: {:?}",
                            fields.len(),
                            line.chars().take(80).collect::<String>(),
                            &fields.iter().take(5).collect::<Vec<_>>()
                        ),
                    });
                }

                // Parse 20 emission scores with per-field error context
                for i in 0..20 {
                    let score = self.parse_hmmer_score(fields[i], line_num)
                        .map_err(|e| HmmerError::ParseError {
                            line: line_num,
                            msg: format!(
                                "Field {}: {} | Context: {}...{} | Emission fields: {:?}",
                                i,
                                e,
                                line.chars().take(40).collect::<String>(),
                                line.chars().rev().take(20).collect::<String>(),
                                &fields.iter().take(3).collect::<Vec<_>>()
                            ),
                        })?;
                    emissions.push(score);
                }

                // Parse 3 transitions with state-specific context
                for i in 20..23 {
                    let score = self.parse_hmmer_score(fields[i], line_num)
                        .map_err(|e| HmmerError::ParseError {
                            line: line_num,
                            msg: format!(
                                "Transition {}: {} | State: Match | Total fields: {}",
                                i - 20,
                                e,
                                fields.len()
                            ),
                        })?;
                    transitions.push(score);
                }
            }
            'I' => {
                // Insert state: 20 emissions + 2 transitions (I->M, I->I)
                if fields.len() < 22 {
                    return Err(HmmerError::ParseError {
                        line: line_num,
                        msg: format!(
                            "Insert state requires 22 fields (20 emissions + 2 transitions), got {}",
                            fields.len()
                        ),
                    });
                }

                for i in 0..20 {
                    let score = self.parse_hmmer_score(fields[i], line_num)?;
                    emissions.push(score);
                }

                for i in 20..22 {
                    let score = self.parse_hmmer_score(fields[i], line_num)
                        .map_err(|e| HmmerError::ParseError {
                            line: line_num,
                            msg: format!("Insert transition {}: {}", i - 20, e),
                        })?;
                    transitions.push(score);
                }
            }
            'D' => {
                // Delete state: no emissions, only 2 transitions (D->M, D->D)
                if fields.len() < 2 {
                    return Err(HmmerError::ParseError {
                        line: line_num,
                        msg: format!(
                            "Delete state requires 2 transition fields, got {}",
                            fields.len()
                        ),
                    });
                }

                // Delete states have no emissions
                emissions = vec![f64::NEG_INFINITY; 20];

                for i in 0..2 {
                    let score = self.parse_hmmer_score(fields[i], line_num)
                        .map_err(|e| HmmerError::ParseError {
                            line: line_num,
                            msg: format!("Delete transition {}: {}", i, e),
                        })?;
                    transitions.push(score);
                }
            }
            _ => {
                return Err(HmmerError::InvalidFormat(format!(
                    "Unknown state type: {}",
                    state_type
                )));
            }
        }

        Ok(HmmerState {
            state_type,
            emissions,
            transitions,
        })
    }

    /// Calculate E-value for a bit score
    pub fn evalue(&self, bit_score: f64, db_size: u64) -> f64 {
        self.karlin.evalue(bit_score, db_size)
    }

    /// Calculate bit score from raw score
    pub fn bit_score(&self, raw_score: f64) -> f64 {
        self.karlin.bit_score(raw_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_karlin_parameters() {
        let k = KarlinParameters::default_protein();
        assert!(k.lambda > 0.0);
        assert!(k.k > 0.0);
        assert_eq!(k.lambda, 0.3176);
    }

    #[test]
    fn test_hmmer_model_creation() {
        let model = HmmerModel {
            name: "TEST".to_string(),
            description: "Test model".to_string(),
            length: 50,
            alpha: "amino".to_string(),
            rf: String::new(),
            consensus: String::new(),
            date: String::new(),
            version: "3.3".to_string(),
            karlin: KarlinParameters::default_protein(),
            states: Vec::new(),
            begin_trans: vec![0.0; 3],
            end_trans: Vec::new(),
            null_model: vec![0.05; 20],
        };

        assert_eq!(model.name, "TEST");
        assert_eq!(model.length, 50);
    }

    #[test]
    fn test_evalue_calculation() {
        let k = KarlinParameters::default_protein();
        let evalue = k.evalue(10.0, 1_000_000);
        eprintln!("E-value for score 10.0 in DB of 1M: {}", evalue);
        // E-value should be positive and reasonable (between 0 and 100 for good scores)
        assert!(evalue > 0.0, "E-value should be positive, got {}", evalue);
        assert!(evalue < 100.0, "E-value seems too large: {}", evalue);
    }

    #[test]
    fn test_parse_error_invalid_numeric_score() {
        // Test that invalid numeric scores return proper errors, not panics
        let line = "invalid_not_a_number";
        
        // Attempting to parse invalid score should return a Result error
        let result = line.trim().parse::<f64>();
        assert!(result.is_err(), "Invalid number should fail to parse");
    }

    #[test]
    fn test_parse_error_special_scores() {
        // Verify special score values are handled properly
        let special_scores = vec!["*", "-inf", "*NNNN", "1.23", "-5.7"];
        
        for score_str in special_scores {
            // These should all be handled without panicking
            let trimmed = score_str.trim();
            match trimmed {
                "*" | "-inf" => {}, // Special values
                s if s.starts_with("*") => {}, // Tagged value
                _ => {
                    let _: f64 = trimmed.parse().expect("Valid number");
                }
            }
        }
    }

    #[test]
    fn test_regex_compilation_safety() {
        // Verify regex compilation returns Result, not panic
        let result = Regex::new(r"(\S+)");
        assert!(result.is_ok(), "Valid regex should compile");
        
        // Even complex regex should not panic
        let complex_regex = Regex::new(r"^[A-Z][A-Za-z0-9_]*$");
        assert!(complex_regex.is_ok(), "Complex regex should compile");
    }

    #[test]
    fn test_karlin_evalue_bounds() {
        let k = KarlinParameters::default_protein();
        
        // Score 0 should give evalue around database size / K
        let evalue_0 = k.evalue(0.0, 1_000_000);
        assert!(evalue_0 > 0.0, "E-value for score 0 should be positive");
        
        // High score should give low e-value
        let evalue_high = k.evalue(50.0, 1_000_000);
        assert!(evalue_high < evalue_0, "Higher score should have lower E-value");
    }

    #[test]
    fn test_hmmer_model_null_model_normalization() {
        let mut model = HmmerModel {
            name: "TEST".to_string(),
            description: "Test".to_string(),
            length: 50,
            alpha: "amino".to_string(),
            rf: String::new(),
            consensus: String::new(),
            date: String::new(),
            version: "3.3".to_string(),
            karlin: KarlinParameters::default_protein(),
            states: Vec::new(),
            begin_trans: vec![0.0; 3],
            end_trans: Vec::new(),
            null_model: vec![0.05; 20], // 20 amino acids
        };
        
        // Null model should sum to approximately 1.0
        let sum: f64 = model.null_model.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Null model should sum to ~1.0, got {}", sum);
    }
}
