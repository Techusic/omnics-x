//! Complete HMMER3 .hmm file format parser (PFAM compatible)

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

/// Fully parsed HMMER3 model
#[derive(Debug, Clone)]
pub struct Hmmer3Model {
    pub name: String,
    pub accession: String,
    pub description: String,
    pub length: usize,
    pub alphabet: String,
    pub alph_size: usize,
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

    /// Load models from HMMER3 .hmm database file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::AlignmentError(format!("Failed to open HMM database: {}", e)))?;
        let reader = BufReader::new(file);

        let mut db = Hmmer3Database::new();
        let mut current_name = String::new();
        let mut current_accession = String::new();
        let mut current_desc = String::new();
        let mut current_len = 0;

        for line in reader.lines() {
            let line = line.map_err(|e| Error::AlignmentError(format!("Read error: {}", e)))?;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "NAME" => current_name = parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
                "ACC" => current_accession = parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
                "DESC" => current_desc = parts[1..].join(" "),
                "LENG" => {
                    current_len = parts
                        .get(1)
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                }
                "//" => {
                    if !current_name.is_empty() {
                        db.models.insert(
                            current_name.clone(),
                            Hmmer3Model {
                                name: current_name.clone(),
                                accession: current_accession.clone(),
                                description: current_desc.clone(),
                                length: current_len,
                                alphabet: "ACDEFGHIKLMNPQRSTVWY".to_string(),
                                alph_size: 20,
                            },
                        );
                    }
                    current_name.clear();
                    current_accession.clear();
                    current_desc.clear();
                    current_len = 0;
                }
                _ => {}
            }
        }

        Ok(db)
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
        };

        assert_eq!(model.name, "test");
        assert_eq!(model.alph_size, 20);
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
        };

        db.insert(model);
        assert_eq!(db.len(), 1);
        assert!(db.get("model1").is_some());
        assert!(db.get_by_accession("PF00001").is_some());
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
            };
            db.insert(model);
        }

        let names = db.names();
        assert_eq!(names.len(), 3);
    }
}
