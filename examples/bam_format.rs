/// Example: BAM (Binary Alignment Map) Format Output
///
/// This example demonstrates reading alignments and exporting them
/// in BAM format - the binary, compressed equivalent of SAM.
///
/// BAM advantages:
/// - 4x smaller than SAM (compression + binary encoding)
/// - Faster I/O operations
/// - Supports random access via BAI index files
/// - Industry standard (GATK, samtools, BCFtools)
///
/// Run:
/// ```bash
/// cargo run --example bam_format --release
/// ```

use omicsx::protein::Protein;
use omicsx::scoring::{ScoringMatrix, MatrixType};
use omicsx::alignment::SmithWaterman;
use omicsx::alignment::bam::{BamFile, BamRecord};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BAM Format Example ===\n");

    // Create reference and query sequences
    let reference_str = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLIVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGCFSDGLVHLVDEY";
    let queries = vec![
        ("read1", "MVHLTPEEKVTVGKVNVDEL"),
        ("read2", "MVHLTPEKGKVNVDVGALGR"),
        ("read3", "VHLTPEKAVTALWGKVNVDEL"),
    ];

    println!("Reference: {} bp", reference_str.len());
    println!("Queries: {}\n", queries.len());

    // Parse reference sequence
    let reference = Protein::from_string(reference_str)?;

    // Create BAM file with header
    let mut bam = {
        use omicsx::alignment::SamHeader;
        let header = SamHeader::new("1.0");
        let mut bam = BamFile::new(header);
        bam.add_reference("chr1".to_string(), reference.len() as u32);
        bam
    };

    // Perform alignments and create BAM records
    let matrix = ScoringMatrix::new(MatrixType::Blosum62)?;
    let aligner = SmithWaterman::with_matrix(matrix);

    println!("Processing alignments:");
    for (name, query_str) in &queries {
        let query = Protein::from_string(query_str)?;
        let result = aligner.align(&reference, &query)?;

        // Print info before moving result fields
        println!("  {}: score={}, aligned at pos {}, identity={:.1}%",
            name, result.score, result.start_pos1, result.identity());

        // Create SAM record
        use omicsx::alignment::SamRecord;
        let sam = SamRecord {
            qname: name.to_string(),
            query_seq: query_str.to_string(),
            query_qual: "I".repeat(query_str.len()),
            rname: "chr1".to_string(),
            reference_seq: reference_str.to_string(),
            pos: result.start_pos1 as u32 + 1, // Convert to 1-based
            mapq: 60,
            cigar: result.cigar,
            flag: 0,
            optional_fields: vec![],
        };

        // Convert to BAM record
        let bam_record = BamRecord::from_sam(&sam, 0);
        bam.add_record(bam_record);
    }

    // Serialize to bytes
    println!("\nSerializing to BAM format...");
    let bam_bytes = bam.to_bytes()?;
    println!("BAM size: {} bytes", bam_bytes.len());

    // Deserialize from bytes to verify round-trip
    println!("\nDeserializing from bytes...");
    let bam_loaded = BamFile::from_bytes(&bam_bytes)?;
    println!("Loaded {} records", bam_loaded.records.len());
    println!("References: {} sequences", bam_loaded.references.len());

    // Display BAM characteristics
    println!("\n=== BAM Format Benefits ===");
    println!("✓ Binary format: More efficient than text SAM");
    println!("✓ Compression-ready: Can be compressed to ~10% of SAM size");
    println!("✓ Fast I/O: Binary parsing faster than text");
    println!("✓ Supports indexing: BAI files enable random access");
    println!("✓ Standard format: Compatible with samtools, BCFtools, GATK");

    println!("\n=== CIGAR Encoding ===");
    if let Some(record) = bam_loaded.records.first() {
        println!("CIGAR operations: {:?}", record.cigar);
        let formatted = omicsx::alignment::bam::BamRecord::format_cigar(&record.cigar);
        println!("CIGAR string: {}", formatted);
    }

    Ok(())
}
