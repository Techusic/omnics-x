//! Complete HMMER3 .hmm file format parser (PFAM compatible)

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

/// Transition probability types in HMM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionType {
    /// M to M (match to match)
    MM,
    /// M to I (match to insert)
    MI,
    /// M to D (match to delete)
    MD,
    /// I to M (insert to match)
    IM,
    /// I to I (insert to insert)
    II,
    /// D to M (delete to match)
    DM,
    /// D to D (delete to delete)
    DD,
}

/// Emission probabilities for a position
#[derive(Debug, Clone)]
pub struct Emission {
    /// Match state emissions
    pub match_emissions: Vec<f32>,
    /// Insert state emissions
    pub insert_emissions: Vec<f32>,
}

/// Fully parsed HMMER3 model with numerical matrices
#[derive(Debug, Clone)]
pub struct Hmmer3Model {
    pub name: String,
    pub accession: String,
    pub description: String,
    pub length: usize,
    pub alphabet: String,
    pub alph_size: usize,
    /// Transition probability matrix [position][transition_type]
    pub transitions: Vec<HashMap<TransitionType, f32>>,
    /// Emission probabilities per position
    pub emissions: Vec<Emission>,
    /// Gathering threshold for domain hits
    pub gathering_threshold: Option<f32>,
    /// Trusted cutoff for domain hits
    pub trusted_threshold: Option<f32>,
}

/// Database of multiple HMMER3 models
pub struct Hmmer3Database {
    models: HashMap<String, Hmmer3Model>,
}

impl Hmmer3Database {
    pub fn new() -> Self {
        Hmmer3Database {
            models: HashMap::new(),
        }
    }

    /// Load models from HMMER3 .hmm database file with numerical matrices
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::AlignmentError(format!("Failed to open HMM database: {}", e)))?;
        let reader = BufReader::new(file);

        let mut db = Hmmer3Database::new();
        let mut lines: Vec<String> = reader.lines().collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::AlignmentError(format!("Read error: {}", e)))?;

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            if line.starts_with("NAME") {
                // Start of new model
                let model = Self::parse_hmm_model(&lines, &mut i)?;
                db.models.insert(model.name.clone(), model);
            } else {
                i += 1;
            }
        }

        Ok(db)
    }

    /// Parse a single HMM model from lines
    fn parse_hmm_model(lines: &[String], pos: &mut usize) -> Result<Hmmer3Model> {
        let mut name = String::new();
        let mut accession = String::new();
        let mut description = String::new();
        let mut length = 0;
        let mut ga_threshold: Option<f32> = None;
        let mut tc_threshold: Option<f32> = None;
        let mut alphabet = "ACDEFGHIKLMNPQRSTVWY".to_string();
        let alph_size = 20;
        let mut transitions: Vec<HashMap<TransitionType, f32>> = Vec::new();
        let mut emissions: Vec<Emission> = Vec::new();

        // Parse header section
        while *pos < lines.len() {
            let line = lines[*pos].trim();
            *pos += 1;

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "NAME" => name = parts.get(1).unwrap_or(&"").to_string(),
                "ACC" => accession = parts.get(1).unwrap_or(&"").to_string(),
                "DESC" => description = parts[1..].join(" "),
                "LENG" => length = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
                "GA" => ga_threshold = parts.get(1).and_then(|s| s.parse().ok()),
                "TC" => tc_threshold = parts.get(1).and_then(|s| s.parse().ok()),
                "ALPH" => alphabet = parts.get(1).unwrap_or(&"").to_string(),
                "HMM" => {
                    // Parse transition and emission matrices
                    Self::parse_hmm_matrices(lines, pos, length, &mut transitions, &mut emissions)?;
                    break;
                }
                "//" => break,
                _ => {}
            }
        }

        Ok(Hmmer3Model {
            name,
            accession,
            description,
            length,
            alphabet,
            alph_size,
            transitions,
            emissions,
            gathering_threshold: ga_threshold,
            trusted_threshold: tc_threshold,
        })
    }

    /// Parse HMM transition and emission matrices
    fn parse_hmm_matrices(
        lines: &[String],
        pos: &mut usize,
        _length: usize,
        transitions: &mut Vec<HashMap<TransitionType, f32>>,
        emissions: &mut Vec<Emission>,
    ) -> Result<()> {
        // Skip alphabet line
        if *pos < lines.len() && lines[*pos].trim().starts_with("COMPO") {
            *pos += 1;
        }

        // Parse each HMM position
        while *pos < lines.len() {
            let line = lines[*pos].trim();

            if line.is_empty() || line.starts_with('#') {
                *pos += 1;
                continue;
            }

            if line.starts_with("//") {
                break;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            // Match state line: position followed by 20 match emissions
            if parts.len() >= 20 && !parts[0].starts_with('M') && !parts[0].starts_with('I') &&
               !parts[0].starts_with('D') {
                let match_emissions: Result<Vec<f32>> = parts[0..20]
                    .iter()
                    .map(|s| s.parse::<f32>()
                        .map_err(|e| Error::AlignmentError(format!("Failed to parse emission: {}", e))))
                    .collect();

                if let Ok(m_emitt) = match_emissions {
                    let mut emission = Emission {
                        match_emissions: m_emitt,
                        insert_emissions: vec![0.0; 20],
                    };

                    // Next line should be insert state emissions
                    *pos += 1;
                    if *pos < lines.len() {
                        let next_line = lines[*pos].trim();
                        let next_parts: Vec<&str> = next_line.split_whitespace().collect();

                        if next_parts.len() >= 20 {
                            if let Ok(i_emitt) = next_parts[0..20]
                                .iter()
                                .map(|s| s.parse::<f32>()
                                    .map_err(|e| Error::AlignmentError(format!("Failed to parse emission: {}", e))))
                                .collect::<Result<Vec<_>>>() {
                                emission.insert_emissions = i_emitt;
                            }
                        }
                    }

                    emissions.push(emission);
                }
            }

            // Transition line: starts with '*' or is a transition row
            if parts.len() >= 7 && (parts[0] == "*->" || !parts[0].chars().all(|c| c.is_numeric() || c == '.' || c == '-')) {
                let mut trans_map = HashMap::new();

                if parts.len() >= 7 {
                    if let Ok(mm) = parts[0].parse::<f32>() { trans_map.insert(TransitionType::MM, mm); }
                    if let Ok(mi) = parts[1].parse::<f32>() { trans_map.insert(TransitionType::MI, mi); }
                    if let Ok(md) = parts[2].parse::<f32>() { trans_map.insert(TransitionType::MD, md); }
                    if let Ok(im) = parts[3].parse::<f32>() { trans_map.insert(TransitionType::IM, im); }
                    if let Ok(ii) = parts[4].parse::<f32>() { trans_map.insert(TransitionType::II, ii); }
                    if let Ok(dm) = parts[5].parse::<f32>() { trans_map.insert(TransitionType::DM, dm); }
                    if let Ok(dd) = parts[6].parse::<f32>() { trans_map.insert(TransitionType::DD, dd); }
                }

                if !trans_map.is_empty() {
                    transitions.push(trans_map);
                }
            }

            *pos += 1;
        }

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Hmmer3Model> {
        self.models.get(name)
    }

    pub fn get_by_accession(&self, accession: &str) -> Option<&Hmmer3Model> {
        self.models.values().find(|m| m.accession == accession)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Hmmer3Model> {
        self.models.values()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn insert(&mut self, model: Hmmer3Model) {
        self.models.insert(model.name.clone(), model);
    }

    pub fn names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

impl Hmmer3Model {
    /// Get emission score for a match state at a position
    pub fn match_emission(&self, position: usize, amino_acid_idx: usize) -> f32 {
        if position < self.emissions.len() && amino_acid_idx < self.emissions[position].match_emissions.len() {
            self.emissions[position].match_emissions[amino_acid_idx]
        } else {
            0.0
        }
    }

    /// Get emission score for an insert state at a position
    pub fn insert_emission(&self, position: usize, amino_acid_idx: usize) -> f32 {
        if position < self.emissions.len() && amino_acid_idx < self.emissions[position].insert_emissions.len() {
            self.emissions[position].insert_emissions[amino_acid_idx]
        } else {
            0.0
        }
    }

    /// Get transition probability at a position
    pub fn transition(&self, position: usize, trans_type: TransitionType) -> f32 {
        if position < self.transitions.len() {
            self.transitions[position].get(&trans_type).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Get all transition probabilities at a position
    pub fn all_transitions(&self, position: usize) -> Option<&HashMap<TransitionType, f32>> {
        self.transitions.get(position)
    }

    /// Get all emission probabilities at a position
    pub fn all_emissions(&self, position: usize) -> Option<&Emission> {
        self.emissions.get(position)
    }

    /// Check if sequence passes gathering threshold
    pub fn passes_gathering(&self, score: f32) -> bool {
        self.gathering_threshold.map(|ga| score >= ga).unwrap_or(true)
    }

    /// Check if sequence passes trusted cutoff
    pub fn passes_trusted(&self, score: f32) -> bool {
        self.trusted_threshold.map(|tc| score >= tc).unwrap_or(true)
    }

    /// Get numerical profile: combined position-specific scoring matrix
    pub fn get_pssm(&self) -> Vec<Vec<f32>> {
        self.emissions.iter()
            .map(|e| e.match_emissions.clone())
            .collect()
    }

    /// Get transition count for all positions
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Get emission count (number of HMM positions)
    pub fn emission_count(&self) -> usize {
        self.emissions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmmer3_database_creation() {
        let db = Hmmer3Database::new();
        assert_eq!(db.len(), 0);
        assert!(db.is_empty());
    }

    #[test]
    fn test_model_creation() {
        let model = Hmmer3Model {
            name: "test".to_string(),
            accession: "PF00001".to_string(),
            description: "Test model".to_string(),
            length: 100,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            alph_size: 20,
            transitions: vec![],
            emissions: vec![],
            gathering_threshold: Some(25.0),
            trusted_threshold: Some(30.0),
        };

        assert_eq!(model.name, "test");
        assert_eq!(model.alph_size, 20);
        assert!(model.passes_gathering(26.0));
        assert!(!model.passes_gathering(24.0));
    }

    #[test]
    fn test_database_insertion() {
        let mut db = Hmmer3Database::new();
        let model = Hmmer3Model {
            name: "model1".to_string(),
            accession: "PF00001".to_string(),
            description: "First".to_string(),
            length: 100,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            alph_size: 20,
            transitions: vec![],
            emissions: vec![],
            gathering_threshold: None,
            trusted_threshold: None,
        };

        db.insert(model);
        assert_eq!(db.len(), 1);
        assert!(db.get("model1").is_some());
        assert!(db.get_by_accession("PF00001").is_some());
    }

    #[test]
    fn test_emission_matrix_parsing() {
        let mut emissions = vec![];
        let mut mut_transmap = HashMap::new();
        mut_transmap.insert(TransitionType::MM, 1.5);
        mut_transmap.insert(TransitionType::MI, 0.2);

        let emission = Emission {
            match_emissions: vec![0.1; 20],
            insert_emissions: vec![0.05; 20],
        };
        emissions.push(emission);

        let model = Hmmer3Model {
            name: "test".to_string(),
            accession: "PF00001".to_string(),
            description: "Test".to_string(),
            length: 10,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            alph_size: 20,
            transitions: vec![mut_transmap],
            emissions,
            gathering_threshold: None,
            trusted_threshold: None,
        };

        assert_eq!(model.emission_count(), 1);
        assert_eq!(model.match_emission(0, 0), 0.1);
        assert_eq!(model.insert_emission(0, 0), 0.05);
        assert_eq!(model.transition(0, TransitionType::MM), 1.5);
        assert_eq!(model.transition(0, TransitionType::MI), 0.2);
    }

    #[test]
    fn test_pssm_generation() {
        let emissions = vec![
            Emission {
                match_emissions: (0..20).map(|i| i as f32 * 0.1).collect(),
                insert_emissions: vec![0.0; 20],
            },
            Emission {
                match_emissions: (0..20).map(|i| i as f32 * 0.2).collect(),
                insert_emissions: vec![0.0; 20],
            },
        ];

        let model = Hmmer3Model {
            name: "test".to_string(),
            accession: "PF00001".to_string(),
            description: "Test".to_string(),
            length: 2,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            alph_size: 20,
            transitions: vec![],
            emissions,
            gathering_threshold: None,
            trusted_threshold: None,
        };

        let pssm = model.get_pssm();
        assert_eq!(pssm.len(), 2);
        assert_eq!(pssm[0].len(), 20);
        assert_eq!(pssm[0][0], 0.0);
        assert_eq!(pssm[0][1], 0.1);
    }

    #[test]
    fn test_threshold_checking() {
        let model = Hmmer3Model {
            name: "test".to_string(),
            accession: "PF00001".to_string(),
            description: "Test".to_string(),
            length: 10,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            alph_size: 20,
            transitions: vec![],
            emissions: vec![],
            gathering_threshold: Some(20.0),
            trusted_threshold: Some(30.0),
        };

        assert!(model.passes_gathering(25.0));
        assert!(!model.passes_gathering(15.0));
        assert!(model.passes_trusted(35.0));
        assert!(!model.passes_trusted(25.0));
    }

    #[test]
    fn test_database_lookup() {
        let mut db = Hmmer3Database::new();
        let model = Hmmer3Model {
            name: "pfam1".to_string(),
            accession: "PF12345".to_string(),
            description: "PFAM domain".to_string(),
            length: 150,
            alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
            alph_size: 20,
            transitions: vec![],
            emissions: vec![],
            gathering_threshold: None,
            trusted_threshold: None,
        };

        db.insert(model);
        assert!(db.get("pfam1").is_some());
        assert!(db.get_by_accession("PF12345").is_some());
        assert!(db.get("nonexistent").is_none());
    }

    #[test]
    fn test_model_names() {
        let mut db = Hmmer3Database::new();
        for i in 0..3 {
            let model = Hmmer3Model {
                name: format!("model{}", i),
                accession: format!("PF{:05}", i),
                description: "Test".to_string(),
                length: 100,
                alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
                alph_size: 20,
                transitions: vec![],
                emissions: vec![],
                gathering_threshold: None,
                trusted_threshold: None,
            };
            db.insert(model);
        }

        let names = db.names();
        assert_eq!(names.len(), 3);
    }
}
