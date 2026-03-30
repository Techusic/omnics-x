/// St. Jude Omics Ecosystem Integration Example
///
/// This example demonstrates how to use the St. Jude bridge module to convert
/// between OMICS-SIMD internal types and St. Jude ecosystem formats.
///
/// # Use Cases
/// - Convert genomic data for St. Jude clinical pipelines
/// - Interoperate with St. Jude cancer research databases
/// - Integrate alignment results into St. Jude analysis workflows
/// - Support pediatric cancer genomics research

use omicsx::protein::Protein;
use omicsx::scoring::{ScoringMatrix, AffinePenalty, MatrixType};
use omicsx::futures::st_jude_bridge::{
    BridgeConfig, StJudeBridge, SequenceType,
};
use omicsx::futures::SeqRecord;

fn main() -> omicsx::Result<()> {
    println!("🏥 St. Jude Omics Ecosystem Integration Example\n");

    // ==================== Example 1: Protein Conversion ====================
    println!("📌 Example 1: Converting Protein Sequences");
    println!("─────────────────────────────────────────\n");

    // Create a protein (e.g., tumor suppressor p53)
    let protein = Protein::from_string("MVHLTPEEKS")?
        .with_id("TP53_HUMAN".to_string())
        .with_description("Tumor suppressor p53 (Fragment)".to_string());

    println!("OMICS-SIMD Protein:");
    println!("  ID: {}", protein.id().unwrap_or("N/A"));
    println!("  Sequence: {}", protein.sequence().iter().map(|aa| aa.to_code()).collect::<String>());
    println!("  Length: {}\n", protein.len());

    // Initialize bridge with default configuration
    let bridge = StJudeBridge::new(BridgeConfig::default());

    // Convert to St. Jude format
    let st_jude_seq = bridge.to_st_jude_sequence(&protein)?;

    println!("St. Jude Sequence:");
    println!("  ID: {}", st_jude_seq.id);
    println!("  Source DB: {:?}", st_jude_seq.source_db);
    println!("  Taxonomy: {:?}", st_jude_seq.taxonomy_id);
    println!("  Sequence Type: {:?}", st_jude_seq.sequence_type);
    println!("  Length: {}\n", st_jude_seq.len());

    // Convert back to OMICS-SIMD format
    let recovered = bridge.from_st_jude_sequence(&st_jude_seq)?;
    assert_eq!(protein.sequence(), recovered.sequence());
    println!("✓ Roundtrip conversion successful!\n");

    // ==================== Example 2: Clinical Metadata ====================
    println!("\n📌 Example 2: Adding Clinical Metadata");
    println!("───────────────────────────────────────\n");

    let mut clinical_seq = st_jude_seq.clone();

    // Add clinical flags for cancer research
    clinical_seq.add_clinical_flag("pathogenic".to_string());
    clinical_seq.add_clinical_flag("loss-of-function".to_string());
    clinical_seq.add_clinical_flag("pediatric-cancer".to_string());

    // Add metadata
    clinical_seq.metadata.insert("gene_name".to_string(), "TP53".to_string());
    clinical_seq.metadata.insert("disease".to_string(), "Li-Fraumeni Syndrome".to_string());
    clinical_seq.metadata.insert("chr".to_string(), "17".to_string());

    println!("Clinical Significance: {:?}", clinical_seq.is_clinically_significant());
    println!("Flags: {:?}", clinical_seq.clinical_flags);
    println!("Metadata: {:?}\n", clinical_seq.metadata);

    // ==================== Example 3: DNA Sequence Handling ====================
    println!("\n📌 Example 3: DNA Sequence Handling");
    println!("──────────────────────────────────\n");

    let dna_record = SeqRecord {
        id: "chr17:7571720-7590863".to_string(),
        description: Some("TP53 gene region".to_string()),
        sequence: "ACGTACGTACGTACGT".to_string(),
        quality: None,
    };

    let st_jude_dna = bridge.seq_record_to_st_jude(&dna_record)?;
    println!("DNA Sequence:");
    println!("  ID: {}", st_jude_dna.id);
    println!("  Type: {:?}", st_jude_dna.sequence_type);
    println!("  Length: {}", st_jude_dna.len());
    println!("  Encoded representation: {:?}\n", &st_jude_dna.sequence[..4.min(st_jude_dna.sequence.len())]);

    // Convert back to SeqRecord
    let recovered_dna = bridge.st_jude_to_seq_record(&st_jude_dna)?;
    assert_eq!(dna_record.sequence, recovered_dna.sequence);
    println!("✓ DNA sequence roundtrip successful!\n");

    // ==================== Example 4: Scoring Matrix Conversion ====================
    println!("\n📌 Example 4: Scoring Matrix Conversion");
    println!("─────────────────────────────────────\n");

    let omics_matrix = ScoringMatrix::new(MatrixType::Blosum62)?;
    let penalty = AffinePenalty::default_protein();

    let st_jude_matrix = bridge.to_st_jude_matrix(&omics_matrix, &penalty)?;

    println!("St. Jude Scoring Matrix:");
    println!("  Name: {}", st_jude_matrix.name);
    println!("  Size: {}x{}", st_jude_matrix.size, st_jude_matrix.size);
    println!("  Gap Open: {}", st_jude_matrix.gap_open);
    println!("  Gap Extend: {}", st_jude_matrix.gap_extend);
    println!("  Reference: {:?}\n", st_jude_matrix.reference);

    // Recover back to OMICS-SIMD format
    let _recovered_matrix = bridge.from_st_jude_matrix(&st_jude_matrix)?;
    println!("✓ Matrix conversion successful!\n");

    // ==================== Example 5: Custom Bridge Configuration ====================
    println!("\n📌 Example 5: Custom Bridge Configuration");
    println!("────────────────────────────────────────\n");

    let custom_config = BridgeConfig {
        include_coordinates: true,
        include_clinical: true,
        default_source_db: Some("ClinVar".to_string()),
        default_taxonomy_id: Some(9606), // Human
        validate_sequences: true,
    };

    let custom_bridge = StJudeBridge::new(custom_config);

    let custom_seq = custom_bridge.to_st_jude_sequence(&protein)?;
    println!("Custom Bridge Configuration:");
    println!("  Source DB: {:?}", custom_seq.source_db);
    println!("  Taxonomy ID: {:?}", custom_seq.taxonomy_id);
    println!("✓ Custom configuration applied!\n");

    // ==================== Example 6: Batch Processing ====================
    println!("\n📌 Example 6: Batch Processing Multiple Proteins");
    println!("──────────────────────────────────────────────\n");

    let proteins = vec![
        ("BRCA1", "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDIICKEFIGEDGCDF"),
        ("BRCA2", "MDLSALRPEAARALRPDEDRLSPLHSVYVDQWDWERVMGGYQQSTNSAAAE"),
        ("TP53", "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDI"),
    ];

    for (gene_name, seq) in proteins {
        match Protein::from_string(seq) {
            Ok(protein) => {
                let protein = protein.with_id(gene_name.to_string());
                let st_jude_seq = custom_bridge.to_st_jude_sequence(&protein)?;
                println!("  {} → St. Jude: ID={}, Length={}", 
                    gene_name, st_jude_seq.id, st_jude_seq.len());
            }
            Err(e) => {
                println!("  {} → Skipped: {}", gene_name, e);
            }
        }
    }
    println!();

    // ==================== Summary ====================
    println!("\n✨ Summary");
    println!("─────────");
    println!("✓ OMICS-SIMD ↔ St. Jude bidirectional conversions working");
    println!("✓ Clinical metadata support enabled");
    println!("✓ DNA/protein sequence handling integrated");
    println!("✓ Scoring matrices compatible");
    println!("✓ Batch processing demonstrated");
    println!("\n🏥 Ready for St. Jude pediatric cancer research integration!\n");

    Ok(())
}
