//! Example: Basic protein sequence alignment
//!
//! This example demonstrates how to perform local alignment between two protein sequences
//! using the Smith-Waterman algorithm with the default BLOSUM62 scoring matrix.

use omicsx::alignment::SmithWaterman;
use omicsx::protein::Protein;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create two protein sequences from FASTA-like strings
    let seq1 = Protein::from_string("MGLSDGEWQLVLNVWGKVEADIPGHGQ")?
        .with_id("Protein1".to_string())
        .with_description("Human hemoglobin subunit".to_string());

    let seq2 = Protein::from_string("MGHHEAELKPLAQSHATKHKIPVKYLEFS")?
        .with_id("Protein2".to_string())
        .with_description("Horse hemoglobin subunit".to_string());

    println!("Protein 1 ({}): {}", seq1.id().unwrap_or("Unknown"), seq1);
    println!("Protein 2 ({}): {}", seq2.id().unwrap_or("Unknown"), seq2);
    println!();

    // Create Smith-Waterman aligner with default settings
    let aligner = SmithWaterman::new();

    // Perform alignment
    let result = aligner.align(&seq1, &seq2)?;

    // Print results
    println!("=== Alignment Results ===");
    println!("Alignment Score: {}", result.score);
    println!("Identity: {:.2}%", result.identity());
    println!("Gaps: {}", result.gap_count());
    println!();
    println!("Alignment:");
    println!("{}", result.aligned_seq1);
    println!("{}", result.aligned_seq2);
    println!();
    println!("Position Range:");
    println!("  Seq1: {}..{}", result.start_pos1, result.end_pos1);
    println!("  Seq2: {}..{}", result.start_pos2, result.end_pos2);

    Ok(())
}
