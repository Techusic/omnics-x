/// Multi-format HMM parser supporting HMMER3, PFAM, HMMSearch, and InterPro formats
///
/// Provides unified interface for parsing various HMM profile formats from
/// different tools and databases.

use crate::error::{Error, Result};
use std::path::Path;

/// Supported HMM file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HmmFormat {
    /// HMMER3 binary/ASCII format (.hmm files)
    Hmmer3,
    /// PFAM ASCII format
    PfamAscii,
    /// HMMSearch text output format
    HmmSearch,
    /// InterPro format
    InterPro,
}

/// Universal HMM profile representation
#[derive(Debug, Clone)]
pub struct UniversalHmmProfile {
    pub name: String,
    pub accession: Option<String>,
    pub description: Option<String>,
    pub length: usize,
    pub alphabet: String,
    pub emission_scores: Vec<Vec<f32>>,
    pub transition_scores: Vec<Vec<f32>>,
    pub meta: HmmMetadata,
}

/// Metadata common across all HMM formats
#[derive(Debug, Clone, Default)]
pub struct HmmMetadata {
    pub ga_threshold: Option<f32>,
    pub tc_threshold: Option<f32>,
    pub nc_threshold: Option<f32>,
    pub author: Option<String>,
    pub source: Option<String>,
    pub date: Option<String>,
    pub command_line: Option<String>,
}

/// Trait for HMM format parsers
pub trait HmmParser: Send + Sync {
    /// Parse HMM file and return universal profile
    fn parse(&self, content: &str) -> Result<UniversalHmmProfile>;
    
    /// Detect if content matches this format
    fn detect(&self, content: &str) -> bool;
    
    /// Get format name
    fn format_name(&self) -> &'static str;
}

/// HMMER3 format parser
pub struct Hmmer3Parser;

impl HmmParser for Hmmer3Parser {
    fn parse(&self, content: &str) -> Result<UniversalHmmProfile> {
        let mut profile = UniversalHmmProfile {
            name: String::new(),
            accession: None,
            description: None,
            length: 0,
            alphabet: "amino".to_string(),
            emission_scores: Vec::new(),
            transition_scores: Vec::new(),
            meta: HmmMetadata::default(),
        };

        for line in content.lines() {
            if line.starts_with("NAME") {
                profile.name = line.split_whitespace().nth(1).unwrap_or("").to_string();
            } else if line.starts_with("ACC") {
                profile.accession = Some(line.split_whitespace().nth(1).unwrap_or("").to_string());
            } else if line.starts_with("DESC") {
                profile.description = Some(line[5..].trim().to_string());
            } else if line.starts_with("LENG") {
                if let Ok(len) = line.split_whitespace().nth(1).unwrap_or("0").parse::<usize>() {
                    profile.length = len;
                }
            } else if line.starts_with("ALPH") {
                profile.alphabet = line.split_whitespace().nth(1).unwrap_or("amino").to_string();
            } else if line.starts_with("GA") {
                profile.meta.ga_threshold = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            } else if line.starts_with("TC") {
                profile.meta.tc_threshold = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            } else if line.starts_with("NC") {
                profile.meta.nc_threshold = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            }
        }

        Ok(profile)
    }

    fn detect(&self, content: &str) -> bool {
        content.contains("HMMER") && content.contains("NAME") && content.contains("LENG")
    }

    fn format_name(&self) -> &'static str {
        "HMMER3"
    }
}

/// PFAM ASCII format parser
pub struct PfamParser;

impl HmmParser for PfamParser {
    fn parse(&self, content: &str) -> Result<UniversalHmmProfile> {
        let mut profile = UniversalHmmProfile {
            name: String::new(),
            accession: None,
            description: None,
            length: 0,
            alphabet: "amino".to_string(),
            emission_scores: Vec::new(),
            transition_scores: Vec::new(),
            meta: HmmMetadata::default(),
        };

        for line in content.lines() {
            if line.starts_with("#=GF ID") {
                profile.name = line.split_whitespace().nth(2).unwrap_or("").to_string();
            } else if line.starts_with("#=GF AC") {
                profile.accession = Some(line.split_whitespace().nth(2).unwrap_or("").to_string());
            } else if line.starts_with("#=GF DE") {
                profile.description = Some(line[8..].trim().to_string());
            } else if line.starts_with("#=GF LEN") {
                if let Ok(len) = line.split_whitespace().nth(2).unwrap_or("0").parse::<usize>() {
                    profile.length = len;
                }
            }
        }

        Ok(profile)
    }

    fn detect(&self, content: &str) -> bool {
        content.contains("#=GF ID") || content.contains("# STOCKHOLM")
    }

    fn format_name(&self) -> &'static str {
        "PFAM"
    }
}

/// HMMSearch text output parser
pub struct HmmSearchParser;

impl HmmParser for HmmSearchParser {
    fn parse(&self, content: &str) -> Result<UniversalHmmProfile> {
        let mut profile = UniversalHmmProfile {
            name: String::new(),
            accession: None,
            description: None,
            length: 0,
            alphabet: "amino".to_string(),
            emission_scores: Vec::new(),
            transition_scores: Vec::new(),
            meta: HmmMetadata::default(),
        };

        for line in content.lines() {
            if line.contains("HMM name:") {
                profile.name = line.split(':').nth(1).unwrap_or("").trim().to_string();
            } else if line.contains("HMM length:") {
                if let Some(len_str) = line.split(':').nth(1).and_then(|s| s.trim().split_whitespace().next()) {
                    if let Ok(len) = len_str.parse::<usize>() {
                        profile.length = len;
                    }
                }
            } else if line.contains("Alphabet:") {
                profile.alphabet = line.split(':').nth(1).unwrap_or("").trim().to_string();
            }
        }

        Ok(profile)
    }

    fn detect(&self, content: &str) -> bool {
        content.contains("HMM name:") && content.contains("HMM length:")
    }

    fn format_name(&self) -> &'static str {
        "HMMSearch"
    }
}

/// InterPro format parser
pub struct InterProParser;

impl HmmParser for InterProParser {
    fn parse(&self, content: &str) -> Result<UniversalHmmProfile> {
        let mut profile = UniversalHmmProfile {
            name: String::new(),
            accession: None,
            description: None,
            length: 0,
            alphabet: "amino".to_string(),
            emission_scores: Vec::new(),
            transition_scores: Vec::new(),
            meta: HmmMetadata::default(),
        };

        for line in content.lines() {
            if line.starts_with("ID") {
                profile.name = line.split_whitespace().nth(1).unwrap_or("").to_string();
            } else if line.starts_with("AC") {
                profile.accession = Some(line.split_whitespace().nth(1).unwrap_or("").to_string());
            } else if line.starts_with("DE") {
                profile.description = Some(line[3..].trim().to_string());
            }
        }

        Ok(profile)
    }

    fn detect(&self, content: &str) -> bool {
        (content.contains("^ID ") || content.starts_with("ID ")) 
            && content.contains("^AC ") || content.contains("AC ")
    }

    fn format_name(&self) -> &'static str {
        "InterPro"
    }
}

/// Multi-format HMM parser registry
pub struct MultiFormatHmmParser {
    parsers: Vec<Box<dyn HmmParser>>,
}

impl MultiFormatHmmParser {
    /// Create a new multi-format parser with all supported formats
    pub fn new() -> Self {
        let parsers: Vec<Box<dyn HmmParser>> = vec![
            Box::new(Hmmer3Parser),
            Box::new(PfamParser),
            Box::new(HmmSearchParser),
            Box::new(InterProParser),
        ];

        MultiFormatHmmParser { parsers }
    }

    /// Parse HMM file, auto-detecting format
    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<UniversalHmmProfile> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::AlignmentError(format!("Failed to read HMM file: {}", e)))?;
        
        self.parse_string(&content)
    }

    /// Parse HMM content from string, auto-detecting format
    pub fn parse_string(&self, content: &str) -> Result<UniversalHmmProfile> {
        // Try parsers in order of specificity
        for parser in &self.parsers {
            if parser.detect(content) {
                eprintln!("Detected format: {}", parser.format_name());
                return parser.parse(content);
            }
        }

        Err(Error::AlignmentError(
            "Unable to detect HMM format. Supported: HMMER3, PFAM, HMMSearch, InterPro".to_string()
        ))
    }

    /// Get list of supported formats
    pub fn supported_formats(&self) -> Vec<&'static str> {
        self.parsers.iter().map(|p| p.format_name()).collect()
    }
}

impl Default for MultiFormatHmmParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmmer3_detection() {
        let hmmer3_content = r#"HMMER3/f [3.3 | Nov 2019]
NAME  TestProfile
ACC   PF00001
DESC  Test HMM
LENG  100
ALPH  amino
"#;
        let parser = Hmmer3Parser;
        assert!(parser.detect(hmmer3_content));
    }

    #[test]
    fn test_pfam_detection() {
        let pfam_content = r#"# STOCKHOLM 1.0
#=GF ID TestProfile
#=GF AC PF00001
#=GF LEN 100
"#;
        let parser = PfamParser;
        assert!(parser.detect(pfam_content));
    }

    #[test]
    fn test_multi_format_parser() {
        let parser = MultiFormatHmmParser::new();
        assert_eq!(parser.supported_formats().len(), 4);
    }
}
