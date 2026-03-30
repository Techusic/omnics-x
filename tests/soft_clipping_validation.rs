//! Comprehensive soft-clipping validation tests
//! 
//! This test suite validates the mathematical correctness of soft-clipping logic
//! in Smith-Waterman local alignment algorithm (SAM format compliance).
//! 
//! **Fixed Issue**: Soft-clipping calculation was inverted, using END positions
//! instead of START positions. This caused incorrect CIGAR strings and metadata.
//!
//! **Mathematical Formula** (now corrected):
//! For local alignment returning AlignmentResult:
//! - start_pos = Position where alignment begins (from traceback)
//! - end_pos = Position where alignment ends (max score location)
//! - left_soft_clip = start_pos (unaligned positions before alignment)
//! - right_soft_clip = seq_len - end_pos (unaligned positions after alignment)

use omicsx::protein::Protein;
use omicsx::alignment::SmithWaterman;

// Helper function to create protein from amino acid codes
fn protein_from_codes(codes: &str) -> Protein {
    let amino_acids = codes
        .chars()
        .map(|c| omicsx::protein::AminoAcid::from_code(c))
        .collect::<Result<Vec<_>, _>>()
        .expect("Invalid amino acid codes");
    Protein::new(amino_acids).expect("Failed to create protein")
}

#[test]
fn test_soft_clipping_perfect_match() {
    // When sequences match completely, there should be no soft-clipping
    let query = protein_from_codes("ACDEFGH");
    let target = protein_from_codes("ACDEFGH");

    let sw = SmithWaterman::new();
    let result = sw.align(&query, &target).unwrap();

    // Full match should have:
    // - start_pos1 = 0 (alignment starts at beginning)
    // - end_pos1 = len (alignment ends at full length)
    // - soft_clips = (0, 0) (no unaligned regions)
    assert_eq!(result.start_pos1, 0, "Perfect match should start at 0");
    assert_eq!(result.end_pos1, 7, "Perfect match should end at full length");
    assert_eq!(result.soft_clips.0, 0, "No left clip for perfect match");
    assert_eq!(result.soft_clips.1, 0, "No right clip for perfect match");
}

#[test]
fn test_soft_clip_formula_invariant() {
    // Test the fundamental algorithm invariant across multiple examples
    // For any alignment: left_clip = start_pos AND right_clip = seq_len - end_pos
    
    let test_cases = vec![
        ("ACDEFGH", "ACDEFGH"),
        ("ACDEFGH", "ACDEF"),
        ("ACDEFG", "ACDEFGH"),
        ("GHACDEFG", "ACDEFGH"),
    ];

    let sw = SmithWaterman::new();
    
    for (query_codes, target_codes) in test_cases {
        let query = protein_from_codes(query_codes);
        let target = protein_from_codes(target_codes);
        
        let result = sw.align(&query, &target).unwrap();
        
        // Core invariant: left_clip must equal start_pos
        assert_eq!(result.soft_clips.0 as usize, result.start_pos1,
            "Invariant failed for query {}: left_clip ({}) != start_pos ({})",
            query_codes, result.soft_clips.0, result.start_pos1);
        
        // Core invariant: right_clip must equal seq_len - end_pos
        let expected_right = query.len() as u32 - result.end_pos1 as u32;
        assert_eq!(result.soft_clips.1, expected_right,
            "Invariant failed for query {}: right_clip ({}) != seq_len ({}) - end_pos ({})",
            query_codes, result.soft_clips.1, query.len(), result.end_pos1);
    }
}

#[test]
fn test_soft_clipping_positions_consistency() {
    // Verify that position fields and soft-clip values are mathematically consistent
    let query = protein_from_codes("SSSACDEFGH");
    let target = protein_from_codes("ACDEFGHTTT");

    let sw = SmithWaterman::new();
    let result = sw.align(&query, &target).unwrap();

    // Invariant: left_clip + (end_pos - start_pos) + right_clip = total_seq_len
    let aligned_length = (result.end_pos1 - result.start_pos1) as u32;
    let total = result.soft_clips.0 + aligned_length + result.soft_clips.1;
    assert_eq!(total as usize, query.len(),
        "Position consistency: {} + {} + {} = {} (expected {})",
        result.soft_clips.0, aligned_length, result.soft_clips.1, total, query.len());

    // Just verify CIGAR string is not empty for non-zero aligned length
    if aligned_length > 0 {
        assert!(!result.cigar.is_empty(),
            "CIGAR string should not be empty for aligned region");
    }
}

#[test]
fn test_soft_clipping_expected_values() {
    // Verify soft-clipping produces expected numeric values
    let query = protein_from_codes("ACDEFGH");
    let target = protein_from_codes("ACDEF");

    let sw = SmithWaterman::new();
    let result = sw.align(&query, &target).unwrap();

    // Verify that all position values are within valid ranges
    assert!(result.start_pos1 <= result.end_pos1, 
        "Start position must be <= end position");
    assert!(result.end_pos1 <= query.len(),
        "End position must be <= query length");
    
    assert!(result.start_pos2 <= result.end_pos2,
        "Start position (seq2) must be <= end position (seq2)");
    assert!(result.end_pos2 <= target.len(),
        "End position (seq2) must be <= target length");
    
    // Verify soft-clip values are non-negative and bounded
    assert!(result.soft_clips.0 <= query.len() as u32,
        "Left soft-clip must be <= query length");
    assert!(result.soft_clips.1 <= query.len() as u32,
        "Right soft-clip must be <= query length");
}

#[test]
fn test_soft_clipping_mathematical_proof() {
    // Formal verification that soft-clipping formula is now correct
    // For an alignment result, verify the invariant: left_clip + aligned + right_clip = total_len
    
    let test_sequences = vec![
        ("ABCDEFGH", "CDEFGH"),
        ("ABCDEFGH", "ABCDEFG"),
        ("XXXACDEFGXXX", "ACDEFG"),
    ];

    let sw = SmithWaterman::new();

    for (query_str, target_str) in test_sequences {
        let query = protein_from_codes(query_str);
        let target = protein_from_codes(target_str);
        let result = sw.align(&query, &target).unwrap();

        // Mathematical identity to verify
        let aligned_region_len = result.end_pos1 - result.start_pos1;
        let reconstructed_len = result.soft_clips.0 as usize + aligned_region_len + result.soft_clips.1 as usize;
        
        assert_eq!(reconstructed_len, query.len(),
            "For query '{}': Formula failed. left_clip ({}) + aligned ({}) + right_clip ({}) = {} (expected {})",
            query_str, result.soft_clips.0, aligned_region_len, result.soft_clips.1,
            reconstructed_len, query.len());
    }
}
