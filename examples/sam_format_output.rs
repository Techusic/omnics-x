/// Example: Generate alignments in SAM format
///
/// This example demonstrates how to:
/// 1. Perform sequence alignment
/// 2. Generate SAM format records
/// 3. Output SAM header and alignment records

use omicsx::alignment::{SmithWaterman, SamHeader, SamRecord};
use omicsx::protein::Protein;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OMICS-SIMD: SAM Format Output Example");
    println!("=====================================\n");

    // Create sequences
    let seq1 = Protein::from_string("AGSGDSAFGCRESVLQSQFSKHSGFKSGQSAY")?;
    let seq2 = Protein::from_string("AGSGDSAF")?;

    // Perform alignment
    let aligner = SmithWaterman::new();
    let result = aligner.align(&seq1, &seq2)?;

    println!("Alignment Results:");
    println!("  Score: {}", result.score);
    println!("  Query:     {}", result.aligned_seq1);
    println!("  Reference: {}", result.aligned_seq2);
    println!("  CIGAR:     {}", result.cigar);
    println!("  Identity:  {:.2}%\n", result.identity());

    // Create SAM header
    let mut header = SamHeader::new("1.6");
    header.add_reference("human_protein", 350);
    header = header.with_program("omics-simd-0.1.0");

    println!("SAM Header:");
    for line in header.to_header_lines() {
        println!("  {}", line);
    }
    println!();

    // Create SAM record from alignment
    let sam_record = SamRecord::from_alignment(
        &result,
        "read001",                    // Query name
        "human_protein",              // Reference name
        result.start_pos2 as u32,    // Reference start position
    );

    println!("SAM Record:");
    println!("  {}\n", sam_record.to_sam_line());

    // Example: Multiple alignments generating multi-record SAM output
    println!("Example: Multiple Alignments in SAM Format");
    println!("==========================================\n");

    let query_sequences = vec![
        ("ref1", "ACGTACGTAC"),
        ("ref2", "TGCATGCA"),
        ("ref3", "ACATTACACA"),
    ];

    let reference = Protein::from_string("ACGTACGTACACATTACACA")?;

    for (ref_name, query_str) in query_sequences {
        let query = Protein::from_string(query_str)?;
        let result = aligner.align(&query, &reference)?;

        let sam = SamRecord::from_alignment(&result, ref_name, "target_ref", 0);
        println!("{}", sam.to_sam_line());
    }

    Ok(())
}
