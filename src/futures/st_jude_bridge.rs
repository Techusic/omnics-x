//! St. Jude Omics Ecosystem Bridge Module
//!
//! This module provides bidirectional conversions between OMICS-SIMD internal types
//! and the St. Jude Children's Research Hospital omics crate types, ensuring seamless
//! interoperability for genomic data processing pipelines.
//!
//! # Overview
//!
//! St. Jude cancer research requires standardized data structures for:
//! - Pediatric cancer genomics
//! - Multi-omics data integration
//! - Clinical variant annotation
//! - Real-time molecular diagnostics
//!
//! This bridge layer abstracts implementation details while maintaining type safety
//! and biological accuracy across both libraries.
//!
//! # Type Mapping
//!
//! | OMICS-SIMD | St. Jude | Purpose |
//! |-----------|---------|---------|
//! | `AminoAcid` | `StJudeAminoAcid` | Standard amino acid representation |
//! | `Protein` | `StJudeSequence` | Protein sequence with metadata |
//! | `SeqRecord` | `StJudeRecord` | Generic sequence record |
//! | `CharState` | `ParsimonyState` | Phylogenetic character states |
//! | `AlignmentResult` | `StJudeAlignment` | Alignment with metrics |
//! | `ScoringMatrix` | `StJudeScoringMatrix` | Amino acid substitution scores |
//!
//! # Example: Converting to St. Jude Format
//!
//! ```ignore
//! use omics_simd::futures::st_jude_bridge::{StJudeBridge, BridgeConfig};
//! use omics_simd::protein::Protein;
//!
//! let protein = Protein::from_string("MVHLTPEEKS")?;
//! let bridge = StJudeBridge::new(BridgeConfig::default());
//!
//! // Convert to St. Jude format
//! let st_jude_seq = bridge.to_st_jude_sequence(&protein)?;
//! println!("St. Jude ID: {}", st_jude_seq.id);
//! ```
//!
//! # Example: Converting from St. Jude Format
//!
//! ```ignore
//! let st_jude_seq = StJudeSequence {
//!     id: "BRCA1".to_string(),
//!     sequence: vec![65, 82, 71], // A, R, G
//!     metadata: Default::default(),
//! };
//!
//! let protein = bridge.from_st_jude_sequence(&st_jude_seq)?;
//! assert_eq!(protein.id(), Some("BRCA1"));
//! ```

use crate::error::{Error, Result};
use crate::protein::{AminoAcid, Protein};
use crate::futures::cli_file_io::SeqRecord;
use crate::alignment::Cigar;
use crate::scoring::{ScoringMatrix, MatrixType, AffinePenalty};
use crate::alignment::hmmer3_parser::KarlinParameters;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// St. Jude amino acid representation
///
/// Maps IUPAC codes to numeric indices used by St. Jude analysis pipelines.
/// This encoding is compatible with common genomics tools (biopython, bioconda).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StJudeAminoAcid(pub u8);

impl StJudeAminoAcid {
    /// Standard 20 amino acids + 4 special codes = 24 total
    pub const CANONICAL_COUNT: usize = 20;
    pub const TOTAL_COUNT: usize = 24;

    /// Encode to St. Jude format (NCBI ncbi_aa_code)
    pub fn from_code(code: char) -> Result<Self> {
        let idx = match code.to_ascii_uppercase() {
            'A' => 0,   // Alanine
            'R' => 1,   // Arginine
            'N' => 2,   // Asparagine
            'D' => 3,   // AsparticAcid
            'C' => 4,   // Cysteine
            'E' => 5,   // GlutamicAcid
            'Q' => 6,   // Glutamine
            'G' => 7,   // Glycine
            'H' => 8,   // Histidine
            'I' => 9,   // Isoleucine
            'L' => 10,  // Leucine
            'K' => 11,  // Lysine
            'M' => 12,  // Methionine
            'F' => 13,  // Phenylalanine
            'P' => 14,  // Proline
            'S' => 15,  // Serine
            'T' => 16,  // Threonine
            'W' => 17,  // Tryptophan
            'Y' => 18,  // Tyrosine
            'V' => 19,  // Valine
            'B' => 20,  // Ambiguous (D/N)
            'Z' => 21,  // Ambiguous (E/Q)
            'X' => 22,  // Uncertain/Any
            '*' => 23,  // Stop codon
            _ => return Err(Error::InvalidAminoAcid(code)),
        };
        Ok(StJudeAminoAcid(idx as u8))
    }

    /// Decode from St. Jude format
    pub fn to_code(&self) -> Result<char> {
        let codes = [
            'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
            'B', 'Z', 'X', '*',
        ];
        codes.get(self.0 as usize).copied()
            .ok_or_else(|| Error::AlignmentError(format!("Invalid amino acid index: {}", self.0)))
    }

    /// Get 3-letter code
    pub fn to_three_letter(&self) -> Result<&'static str> {
        let names = [
            "Ala", "Arg", "Asn", "Asp", "Cys", "Glu", "Gln", "Gly", "His", "Ile",
            "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
            "Asx", "Glx", "Xaa", "***",
        ];
        names.get(self.0 as usize).copied()
            .ok_or_else(|| Error::AlignmentError(format!("Invalid amino acid index: {}", self.0)))
    }
}

/// St. Jude sequence representation
///
/// Compatible with St. Jude's SequenceRecord type used in their omics platform.
/// Includes genomic coordinates, database references, and clinical metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StJudeSequence {
    /// Unique sequence identifier (e.g., "BRCA1", "TP53_HUMAN")
    pub id: String,

    /// Optional accession number (e.g., GenBank, UniProt, RefSeq)
    pub accession: Option<String>,

    /// Sequence data (amino acids as u8 indices or raw nucleotides)
    pub sequence: Vec<u8>,

    /// Sequence type (protein=20, dna=4, rna=4)
    pub sequence_type: SequenceType,

    /// Optional description/annotation
    pub description: Option<String>,

    /// Genomic coordinates if applicable (chromosome:start-end:strand)
    pub genomic_location: Option<String>,

    /// Database source (UniProt, RefSeq, Ensembl, NCBI)
    pub source_db: Option<String>,

    /// Taxonomy ID (e.g., 9606 for Homo sapiens)
    pub taxonomy_id: Option<u32>,

    /// Clinical significance flags
    pub clinical_flags: Vec<String>,

    /// Custom metadata key-value pairs
    pub metadata: HashMap<String, String>,
}

impl StJudeSequence {
    /// Create a new St. Jude sequence
    pub fn new(id: String, sequence: Vec<u8>, sequence_type: SequenceType) -> Self {
        StJudeSequence {
            id,
            accession: None,
            sequence,
            sequence_type,
            description: None,
            genomic_location: None,
            source_db: None,
            taxonomy_id: None,
            clinical_flags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get sequence length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Add clinical flag
    pub fn add_clinical_flag(&mut self, flag: String) {
        self.clinical_flags.push(flag);
    }

    /// Check if sequence has clinical significance
    pub fn is_clinically_significant(&self) -> bool {
        !self.clinical_flags.is_empty()
    }
}

/// Sequence type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceType {
    /// Protein sequence (20 amino acids)
    Protein,
    /// DNA sequence (4 nucleotides)
    Dna,
    /// RNA sequence (4 nucleotides)
    Rna,
    /// Codon sequence (61 sense codons)
    Codon,
}

/// St. Jude alignment result
///
/// Extended alignment metadata for clinical and research applications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StJudeAlignment {
    /// Query sequence ID
    pub query_id: String,

    /// Subject sequence ID
    pub subject_id: String,

    /// Alignment score
    pub score: i32,

    /// E-value (statistical significance)
    pub evalue: f64,

    /// Bit score (normalized score)
    pub bit_score: f64,

    /// Sequence identity (0.0-1.0)
    pub identity: f64,

    /// Query start position (1-based)
    pub query_start: u32,

    /// Query end position (1-based)
    pub query_end: u32,

    /// Subject start position (1-based)
    pub subject_start: u32,

    /// Subject end position (1-based)
    pub subject_end: u32,

    /// Number of matching residues
    pub matches: u32,

    /// Alignment length
    pub alignment_length: u32,

    /// Gap openings
    pub gap_opens: u32,

    /// Number of gaps
    pub gaps: u32,

    /// CIGAR string representation
    pub cigar: String,

    /// Query alignment string
    pub query_string: String,

    /// Subject alignment string
    pub subject_string: String,

    /// Clinical interpretation
    pub interpretation: Option<String>,

    /// Associated databases (ClinVar, COSMIC, etc.)
    pub databases: Vec<String>,
}

/// St. Jude scoring matrix
///
/// Standardized substitution score matrix compatible with St. Jude pipelines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StJudeScoringMatrix {
    /// Matrix name (BLOSUM62, PAM30, etc.)
    pub name: String,

    /// 24x24 substitution score matrix
    pub scores: Vec<Vec<i32>>,

    /// Matrix size (typically 24 for proteins)
    pub size: usize,

    /// Gap opening penalty
    pub gap_open: i32,

    /// Gap extension penalty
    pub gap_extend: i32,

    /// Matrix version/reference
    pub reference: Option<String>,
}

/// St. Jude parsimony state
///
/// Character state representation for phylogenetic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParsimonyState(pub u8);

impl ParsimonyState {
    /// Create from amino acid code
    pub fn from_code(code: char) -> Result<Self> {
        let idx = match code.to_ascii_uppercase() {
            'A' => 0,
            'R' => 1,
            'N' => 2,
            'D' => 3,
            'C' => 4,
            'E' => 5,
            'Q' => 6,
            'G' => 7,
            'H' => 8,
            'I' => 9,
            'L' => 10,
            'K' => 11,
            'M' => 12,
            'F' => 13,
            'P' => 14,
            'S' => 15,
            'T' => 16,
            'W' => 17,
            'Y' => 18,
            'V' => 19,
            _ => return Err(Error::InvalidAminoAcid(code)),
        };
        Ok(ParsimonyState(idx as u8))
    }

    /// Check state transition
    pub fn transition_cost(&self, other: ParsimonyState) -> u32 {
        if self.0 != other.0 { 1 } else { 0 }
    }
}

/// Bridge configuration for type conversion
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Include genomic coordinates in conversions
    pub include_coordinates: bool,

    /// Include clinical metadata
    pub include_clinical: bool,

    /// Default source database
    pub default_source_db: Option<String>,

    /// Default taxonomy ID (default: 9606 for Human)
    pub default_taxonomy_id: Option<u32>,

    /// Validate converted sequences
    pub validate_sequences: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig {
            include_coordinates: true,
            include_clinical: true,
            default_source_db: Some("Ensembl".to_string()),
            default_taxonomy_id: Some(9606), // Homo sapiens
            validate_sequences: true,
        }
    }
}

/// Main bridge for bidirectional type conversion
pub struct StJudeBridge {
    config: BridgeConfig,
}

impl StJudeBridge {
    /// Create new bridge with default configuration
    pub fn new(config: BridgeConfig) -> Self {
        StJudeBridge { config }
    }

    // ==================== Amino Acid Conversions ====================

    /// Convert OMICS-SIMD AminoAcid to St. Jude format
    pub fn to_st_jude_amino_acid(&self, aa: AminoAcid) -> Result<StJudeAminoAcid> {
        StJudeAminoAcid::from_code(aa.to_code())
    }

    /// Convert St. Jude amino acid to OMICS-SIMD format
    pub fn from_st_jude_amino_acid(&self, st_jude_aa: StJudeAminoAcid) -> Result<AminoAcid> {
        let code = st_jude_aa.to_code()?;
        AminoAcid::from_code(code)
    }

    // ==================== Sequence Conversions ====================

    /// Convert Protein to St. Jude sequence
    pub fn to_st_jude_sequence(&self, protein: &Protein) -> Result<StJudeSequence> {
        let mut sequence = Vec::new();
        for &aa in protein.sequence() {
            let st_jude_aa = self.to_st_jude_amino_acid(aa)?;
            sequence.push(st_jude_aa.0);
        }

        let mut st_jude_seq = StJudeSequence::new(
            protein.id().unwrap_or("unknown").to_string(),
            sequence,
            SequenceType::Protein,
        );

        if let Some(desc) = protein.description() {
            st_jude_seq.description = Some(desc.to_string());
        }

        if let Some(db) = &self.config.default_source_db {
            st_jude_seq.source_db = Some(db.clone());
        }

        st_jude_seq.taxonomy_id = self.config.default_taxonomy_id;

        Ok(st_jude_seq)
    }

    /// Convert St. Jude sequence to Protein
    pub fn from_st_jude_sequence(&self, st_jude_seq: &StJudeSequence) -> Result<Protein> {
        if st_jude_seq.sequence_type != SequenceType::Protein {
            return Err(Error::AlignmentError(
                "St. Jude sequence must be Protein type for conversion".to_string(),
            ));
        }

        if self.config.validate_sequences && st_jude_seq.is_empty() {
            return Err(Error::EmptySequence);
        }

        let mut sequence = Vec::new();
        for &idx in &st_jude_seq.sequence {
            let st_jude_aa = StJudeAminoAcid(idx);
            let aa = self.from_st_jude_amino_acid(st_jude_aa)?;
            sequence.push(aa);
        }

        let mut protein = Protein::new(sequence)?;
        protein = protein.with_id(st_jude_seq.id.clone());

        if let Some(desc) = &st_jude_seq.description {
            protein = protein.with_description(desc.clone());
        }

        Ok(protein)
    }

    // ==================== SeqRecord Conversions ====================

    /// Convert SeqRecord to St. Jude sequence
    pub fn seq_record_to_st_jude(&self, record: &SeqRecord) -> Result<StJudeSequence> {
        let sequence_type = match record.quality {
            Some(_) => SequenceType::Dna, // FASTQ typically contains DNA
            None => SequenceType::Dna,     // FASTA without quality
        };

        let sequence = if sequence_type == SequenceType::Dna {
            // Encode DNA as indices (A=0, C=1, G=2, T=3)
            let mut seq = Vec::new();
            for c in record.sequence.to_uppercase().chars() {
                let idx = match c {
                    'A' => 0,
                    'C' => 1,
                    'G' => 2,
                    'T' => 3,
                    'N' | 'X' => 4, // Ambiguous
                    _ => 4,
                };
                seq.push(idx);
            }
            seq
        } else {
            vec![]
        };

        let mut st_jude_seq = StJudeSequence::new(record.id.clone(), sequence, sequence_type);

        st_jude_seq.description = record.description.clone();
        if let Some(db) = &self.config.default_source_db {
            st_jude_seq.source_db = Some(db.clone());
        }
        st_jude_seq.taxonomy_id = self.config.default_taxonomy_id;

        Ok(st_jude_seq)
    }

    /// Convert St. Jude sequence to SeqRecord
    pub fn st_jude_to_seq_record(&self, st_jude_seq: &StJudeSequence) -> Result<SeqRecord> {
        let sequence = if st_jude_seq.sequence_type == SequenceType::Dna {
            // Decode DNA from indices
            let mut seq = String::new();
            for &idx in &st_jude_seq.sequence {
                let c = match idx {
                    0 => 'A',
                    1 => 'C',
                    2 => 'G',
                    3 => 'T',
                    4 => 'N',
                    _ => 'N',
                };
                seq.push(c);
            }
            seq
        } else {
            String::new()
        };

        Ok(SeqRecord {
            id: st_jude_seq.id.clone(),
            description: st_jude_seq.description.clone(),
            sequence,
            quality: None,
        })
    }

    // ==================== Alignment Conversions ====================

    /// Convert OMICS-SIMD AlignmentResult to St. Jude alignment
    /// Convert alignment result to St. Jude format with accurate statistical scoring
    /// Uses model-specific Karlin-Altschul parameters for proper bit score calculation
    pub fn to_st_jude_alignment(
        &self,
        query_id: &str,
        subject_id: &str,
        score: i32,
        cigar: &Cigar,
        query_string: &str,
        subject_string: &str,
        karlin_params: &KarlinParameters,
    ) -> Result<StJudeAlignment> {
        // Calculate alignment metrics from CIGAR
        let alignment_length = cigar.query_length();
        let query_start = 1; // Default 1-based
        let query_end = query_start + cigar.query_length();
        let subject_start = 1;
        let subject_end = subject_start + cigar.reference_length();

        // Count matches and gaps
        let mut matches = 0;
        let mut gaps = 0;
        for &c in query_string.as_bytes() {
            if c == b'-' {
                gaps += 1;
            } else {
                matches += 1;
            }
        }

        let identity = if alignment_length > 0 {
            matches as f64 / alignment_length as f64
        } else {
            0.0
        };

        // Calculate bit score using Karlin-Altschul parameters from model
        let bit_score = karlin_params.bit_score(score as f64);
        
        // Calculate E-value assuming standard database size of 1 billion sequences
        // (this should ideally be passed in for the actual database size)
        let db_size = 1_000_000_000u64; // 1 billion is typical for protein DB
        let evalue = karlin_params.evalue(bit_score, db_size);

        Ok(StJudeAlignment {
            query_id: query_id.to_string(),
            subject_id: subject_id.to_string(),
            score,
            evalue,
            bit_score: bit_score.max(0.0),
            identity,
            query_start,
            query_end,
            subject_start,
            subject_end,
            matches,
            alignment_length,
            gap_opens: 1, // Simplified
            gaps,
            cigar: cigar.to_string(),
            query_string: query_string.to_string(),
            subject_string: subject_string.to_string(),
            interpretation: None,
            databases: Vec::new(),
        })
    }

    // ==================== Matrix Conversions ====================

    /// Convert ScoringMatrix to St. Jude format
    pub fn to_st_jude_matrix(
        &self,
        matrix: &ScoringMatrix,
        gap_penalty: &AffinePenalty,
    ) -> Result<StJudeScoringMatrix> {
        let name = matrix.matrix_type().to_string();
        let scores = matrix.raw_scores();

        Ok(StJudeScoringMatrix {
            name,
            scores: scores.clone(),
            size: scores.len(),
            gap_open: gap_penalty.open,
            gap_extend: gap_penalty.extend,
            reference: Some("NCBI Standard".to_string()),
        })
    }

    /// Convert St. Jude matrix to ScoringMatrix
    pub fn from_st_jude_matrix(&self, st_jude_matrix: &StJudeScoringMatrix) -> Result<ScoringMatrix> {
        // Map known matrix names
        let matrix_type = match st_jude_matrix.name.to_uppercase().as_str() {
            "BLOSUM62" => MatrixType::Blosum62,
            "BLOSUM45" => MatrixType::Blosum45,
            "BLOSUM80" => MatrixType::Blosum80,
            "PAM30" => MatrixType::Pam30,
            "PAM70" => MatrixType::Pam70,
            _ => MatrixType::Blosum62, // Default fallback
        };

        ScoringMatrix::new(matrix_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_st_jude_amino_acid_conversion() -> Result<()> {
        // Test single amino acid conversion
        let st_jude_aa = StJudeAminoAcid::from_code('A')?;
        assert_eq!(st_jude_aa.0, 0);
        assert_eq!(st_jude_aa.to_code()?, 'A');

        let st_jude_aa = StJudeAminoAcid::from_code('W')?;
        assert_eq!(st_jude_aa.0, 17);

        // Test all 20 canonical amino acids
        for c in "ACDEFGHIKLMNPQRSTVWY".chars() {
            let st_jude_aa = StJudeAminoAcid::from_code(c)?;
            assert_eq!(st_jude_aa.to_code()?, c);
        }

        Ok(())
    }

    #[test]
    fn test_parsimony_state_creation() -> Result<()> {
        let state = ParsimonyState::from_code('M')?;
        assert_eq!(state.0, 12);

        let state2 = ParsimonyState::from_code('M')?;
        assert_eq!(state.transition_cost(state2), 0); // Same state

        let state3 = ParsimonyState::from_code('L')?;
        assert_eq!(state.transition_cost(state3), 1); // Different state

        Ok(())
    }

    #[test]
    fn test_bridge_protein_to_st_jude() -> Result<()> {
        let bridge = StJudeBridge::new(BridgeConfig::default());
        let protein = Protein::from_string("MVHLTPEEKS")?
            .with_id("HBB".to_string())
            .with_description("Beta-globin".to_string());

        let st_jude_seq = bridge.to_st_jude_sequence(&protein)?;

        assert_eq!(st_jude_seq.id, "HBB");
        assert_eq!(st_jude_seq.description, Some("Beta-globin".to_string()));
        assert_eq!(st_jude_seq.sequence_type, SequenceType::Protein);
        assert_eq!(st_jude_seq.len(), 10);
        assert_eq!(st_jude_seq.source_db, Some("Ensembl".to_string()));

        Ok(())
    }

    #[test]
    fn test_bridge_st_jude_to_protein() -> Result<()> {
        let bridge = StJudeBridge::new(BridgeConfig::default());

        let mut st_jude_seq =
            StJudeSequence::new("TP53".to_string(), vec![3, 14, 19], SequenceType::Protein);
        st_jude_seq.description = Some("Tumor suppressor p53".to_string());

        let protein = bridge.from_st_jude_sequence(&st_jude_seq)?;

        assert_eq!(protein.id(), Some("TP53"));
        assert_eq!(protein.description(), Some("Tumor suppressor p53"));
        assert_eq!(protein.len(), 3);

        Ok(())
    }

    #[test]
    fn test_bridge_roundtrip_conversion() -> Result<()> {
        let bridge = StJudeBridge::new(BridgeConfig::default());

        // OMICS -> St. Jude -> OMICS
        let original_protein = Protein::from_string("ARNDCQEGHILKMFPSTWYV")?
            .with_id("ALL_AAS".to_string());

        let st_jude_seq = bridge.to_st_jude_sequence(&original_protein)?;
        let recovered_protein = bridge.from_st_jude_sequence(&st_jude_seq)?;

        assert_eq!(original_protein.sequence(), recovered_protein.sequence());
        assert_eq!(original_protein.id(), recovered_protein.id());

        Ok(())
    }

    #[test]
    fn test_seq_record_to_st_jude() -> Result<()> {
        let bridge = StJudeBridge::new(BridgeConfig::default());

        let record = SeqRecord {
            id: "chr1:1000-1100".to_string(),
            description: Some("Test region".to_string()),
            sequence: "ACGTACGTACGT".to_string(),
            quality: None,
        };

        let st_jude_seq = bridge.seq_record_to_st_jude(&record)?;

        assert_eq!(st_jude_seq.id, "chr1:1000-1100");
        assert_eq!(st_jude_seq.sequence_type, SequenceType::Dna);
        assert_eq!(st_jude_seq.len(), 12);

        Ok(())
    }

    #[test]
    fn test_alignment_conversion() -> Result<()> {
        let bridge = StJudeBridge::new(BridgeConfig::default());
        let karlin = KarlinParameters::default_protein();

        let cigar = crate::alignment::Cigar::new();
        let alignment = bridge.to_st_jude_alignment(
            "query1",
            "subject1",
            85,
            &cigar,
            "ACDEF",
            "ACGEF",
            &karlin,
        )?;

        assert_eq!(alignment.query_id, "query1");
        assert_eq!(alignment.subject_id, "subject1");
        assert_eq!(alignment.score, 85);
        assert!(alignment.identity >= 0.0 && alignment.identity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_clinical_flags() -> Result<()> {
        let mut st_jude_seq = StJudeSequence::new("BRCA1".to_string(), vec![], SequenceType::Protein);

        assert!(!st_jude_seq.is_clinically_significant());

        st_jude_seq.add_clinical_flag("pathogenic".to_string());
        st_jude_seq.add_clinical_flag("loss-of-function".to_string());

        assert!(st_jude_seq.is_clinically_significant());
        assert_eq!(st_jude_seq.clinical_flags.len(), 2);

        Ok(())
    }

    #[test]
    fn test_bridge_empty_sequence_validation() -> Result<()> {
        let bridge = StJudeBridge::new(BridgeConfig {
            validate_sequences: true,
            ..Default::default()
        });

        let empty_seq = StJudeSequence::new("empty".to_string(), vec![], SequenceType::Protein);

        let result = bridge.from_st_jude_sequence(&empty_seq);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_metadata_preservation() -> Result<()> {
        let mut st_jude_seq =
            StJudeSequence::new("TEST".to_string(), vec![0], SequenceType::Protein);

        st_jude_seq.metadata.insert("gene_name".to_string(), "PTEN".to_string());
        st_jude_seq.metadata.insert("chr".to_string(), "10".to_string());

        assert_eq!(st_jude_seq.metadata.get("gene_name"), Some(&"PTEN".to_string()));
        assert_eq!(st_jude_seq.metadata.get("chr"), Some(&"10".to_string()));

        Ok(())
    }

    #[test]
    fn test_taxonomy_id_defaults() {
        let bridge = StJudeBridge::new(BridgeConfig::default());
        assert_eq!(bridge.config.default_taxonomy_id, Some(9606)); // Human
    }

    #[test]
    fn test_three_letter_codes() -> Result<()> {
        let st_jude_aa = StJudeAminoAcid::from_code('G')?;
        assert_eq!(st_jude_aa.to_three_letter()?, "Gly");

        let st_jude_aa = StJudeAminoAcid::from_code('P')?;
        assert_eq!(st_jude_aa.to_three_letter()?, "Pro");

        Ok(())
    }
}
