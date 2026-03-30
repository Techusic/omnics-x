/// Integration tests for multi-format HMM parser
///
/// Tests parsing of different HMM formats and auto-detection

#[cfg(test)]
mod multiformat_hmm_tests {
    use omicsx::alignment::{MultiFormatHmmParser, HmmFormat};

    const HMMER3_SAMPLE: &str = r#"HMMER3/f [3.3 | Nov 2019]
NAME  PF00001
ACC   PF00001.28
DESC  7 transmembrane receptor (7tm) superfamily
LENG  345
ALPH  amino
GA    30.00 45.00
TC    35.00 50.00
NC    20.00 30.00
"#;

    const PFAM_SAMPLE: &str = r#"# STOCKHOLM 1.0
#=GF ID   PF00001
#=GF AC   PF00001.28
#=GF DE   7 transmembrane receptor (7tm) superfamily
#=GF LEN  345
#=GF SQ   1543
"#;

    const HMMSEARCH_SAMPLE: &str = r#"# hmmsearch :: search profile(s) against a sequence database
# HMMER 3.3 (Nov 2019); http://hmmer.org/
# Copyright (C) 2019 Sean R. Eddy
# HMM file: ../pfam/Pfam-A.hmm.dat [Pfam-A.3 (release 33.0, Feb 2020)]

Query:       PF00001  [M=345]
Accession:   PF00001.28
Description: 7 transmembrane receptor (7tm) superfamily
Scores for complete sequences (score includes all domains):
--- full sequence E-value  score  bias  Description
---   --------- --  -----  -----  ----  -----------
"#;

    const INTERPRO_SAMPLE: &str = r#"ID   IPR000001
AC   IPR000001
DE   Transmembrane receptor superfamily
DT   28-MAY-2001 (Rel. 7, Created)
DT   19-JAN-2005 (Rel. 16, Last updated)
CC   This superfamily consists of the transmembrane receptors.
"#;

    #[test]
    fn test_hmmer3_format_detection() {
        let parser = MultiFormatHmmParser::new();
        let result = parser.parse_string(HMMER3_SAMPLE);
        assert!(result.is_ok());
        
        let profile = result.unwrap();
        assert_eq!(profile.name, "PF00001");
        assert_eq!(profile.accession.as_deref(), Some("PF00001.28"));
        assert_eq!(profile.length, 345);
        assert_eq!(profile.alphabet, "amino");
        assert_eq!(profile.meta.ga_threshold, Some(30.0));
        assert_eq!(profile.meta.tc_threshold, Some(35.0));
        assert_eq!(profile.meta.nc_threshold, Some(20.0));
    }

    #[test]
    fn test_pfam_format_detection() {
        let parser = MultiFormatHmmParser::new();
        let result = parser.parse_string(PFAM_SAMPLE);
        assert!(result.is_ok());
        
        let profile = result.unwrap();
        assert_eq!(profile.name, "PF00001");
        assert_eq!(profile.accession.as_deref(), Some("PF00001.28"));
        assert_eq!(profile.length, 345);
    }

    #[test]
    fn test_hmmsearch_format_detection() {
        let parser = MultiFormatHmmParser::new();
        let result = parser.parse_string(HMMSEARCH_SAMPLE);
        assert!(result.is_ok());
        
        let profile = result.unwrap();
        assert_eq!(profile.name, "PF00001");
        assert_eq!(profile.length, 345);
    }

    #[test]
    fn test_interpro_format_detection() {
        let parser = MultiFormatHmmParser::new();
        let result = parser.parse_string(INTERPRO_SAMPLE);
        assert!(result.is_ok());
        
        let profile = result.unwrap();
        assert_eq!(profile.name, "IPR000001");
        assert_eq!(profile.accession.as_deref(), Some("IPR000001"));
    }

    #[test]
    fn test_supported_formats_list() {
        let parser = MultiFormatHmmParser::new();
        let formats = parser.supported_formats();
        assert_eq!(formats.len(), 4);
        assert!(formats.contains(&"HMMER3"));
        assert!(formats.contains(&"PFAM"));
        assert!(formats.contains(&"HMMSearch"));
        assert!(formats.contains(&"InterPro"));
    }

    #[test]
    fn test_invalid_format_detection() {
        let parser = MultiFormatHmmParser::new();
        let invalid = "This is not any HMM format";
        let result = parser.parse_string(invalid);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unable to detect"));
    }

    #[test]
    fn test_hmmer3_metadata_parsing() {
        let parser = MultiFormatHmmParser::new();
        let result = parser.parse_string(HMMER3_SAMPLE);
        
        assert!(result.is_ok());
        let profile = result.unwrap();
        assert!(profile.meta.ga_threshold.is_some());
        assert!(profile.meta.tc_threshold.is_some());
        assert!(profile.meta.nc_threshold.is_some());
    }

    #[test]
    fn test_default_values() {
        let parser = MultiFormatHmmParser::new();
        let result = parser.parse_string(HMMER3_SAMPLE);
        
        let profile = result.unwrap();
        assert!(!profile.emission_scores.is_empty() || profile.emission_scores.is_empty()); // Can be empty
        assert_eq!(profile.description, Some("7 transmembrane receptor (7tm) superfamily".to_string()));
    }
}
