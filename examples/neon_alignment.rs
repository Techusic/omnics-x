/// Example: NEON-accelerated sequence alignment on ARM
///
/// This example demonstrates how the NEON kernel accelerates alignment
/// on ARM architectures (aarch64). The same code works cross-platform,
/// with automatic fallback to scalar on non-ARM systems.
///
/// To use on ARM systems:
/// - AWS Graviton instances
/// - Apple Silicon Macs
/// - Raspberry Pi 4/5 with 64-bit OS
/// - ARM64 servers (e.g., Oracle Ampere)
///
/// Build and run:
/// ```bash
/// cargo run --example neon_alignment --release
/// ```
///
/// Cross-compile for ARM (from x86_64):
/// ```bash
/// rustup target add aarch64-unknown-linux-gnu
/// cargo build --example neon_alignment --target aarch64-unknown-linux-gnu --release
/// # Transfer binary to ARM system and run
/// ```

use omicsx::protein::{AminoAcid, Protein};
use omicsx::scoring::{ScoringMatrix, MatrixType};
use omicsx::alignment::SmithWaterman;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NEON-Optimized Alignment Example ===\n");

    // Create two protein sequences
    let seq1_str = "MVHLTPEEKS";
    let seq2_str = "MGHLTPEEKS";

    println!("Sequence 1: {}", seq1_str);
    println!("Sequence 2: {}\n", seq2_str);

    // Parse sequences
    let seq1 = Protein::from_string(seq1_str)?;
    let seq2 = Protein::from_string(seq2_str)?;

    // Create alignment with BLOSUM62 scoring matrix
    let matrix = ScoringMatrix::new(MatrixType::Blosum62)?;
    let aligner = SmithWaterman::with_matrix(matrix);

    // Align sequences
    let result = aligner.align(&seq1, &seq2)?;

    println!("Alignment Score: {}", result.score);
    println!("Identity: {:.1}%\n", result.identity());

    // Generate CIGAR string
    let mut result_with_cigar = result.clone();
    result_with_cigar.generate_cigar();
    println!("CIGAR: {}", result_with_cigar.cigar);

    // Information about execution
    if cfg!(target_arch = "aarch64") {
        println!("\n✓ Running on ARM64 with NEON acceleration");
    } else {
        println!("\n✓ Running on non-ARM system using scalar implementation");
        println!("  (For NEON acceleration, compile and run on aarch64 hardware)");
    }

    // Demonstrate batch processing capability
    println!("\n=== Batch Processing Example ===\n");

    use omicsx::alignment::batch::*;

    let reference_seq = "MVHLTPEEKS";
    let batch_aligner =
        BatchSmithWaterman::new(reference_seq, BatchConfig::new().with_threads(4))?;

    let queries = vec![
        BatchQuery {
            name: "query1".to_string(),
            sequence: "MGHLTPEEKS".to_string(),
        },
        BatchQuery {
            name: "query2".to_string(),
            sequence: "MVHLTPEEKS".to_string(),
        },
        BatchQuery {
            name: "query3".to_string(),
            sequence: "MVXLTPEEKS".to_string(),
        },
    ];

    let results = batch_aligner.align_batch(queries)?;

    println!("Processed {} queries:", results.len());
    for result in &results {
        println!(
            "  {}: score={}, identity={:.1}%",
            result.query_name,
            result.alignment.score,
            result.alignment.identity()
        );
    }

    // Filter by score threshold
    let high_scoring = BatchSmithWaterman::filter_by_score(&results, 40);
    println!("\nHigh-scoring results (score >= 40): {}", high_scoring.len());

    // Filter by identity threshold
    let high_identity = BatchSmithWaterman::filter_by_identity(&results, 80.0);
    println!("High-identity results (>= 80% identity): {}", high_identity.len());

    // Performance note
    println!("\n=== Performance Notes ===");
    println!("NEON Alignment Characteristics:");
    println!("  • 128-bit vectors: 4× int32 parallelism");
    println!("  • Anti-diagonal approach: Independent cell computation");
    println!("  • Expected speedup: 2-4× over scalar on ARM64");
    println!("  • Platforms: AWS Graviton, Apple Silicon, Ampere, Raspberry Pi 4/5");

    Ok(())
}
