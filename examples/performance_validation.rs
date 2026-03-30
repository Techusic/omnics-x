/// Simple performance validation comparing scalar vs SIMD kernels
/// Avoids criterion overhead to get direct timing measurements

use omicsx::alignment::SmithWaterman;
use omicsx::protein::Protein;
use std::time::Instant;

fn main() {
    println!("OMICS-SIMD Performance Validation");
    println!("================================\n");

    // Test case 1: Small sequences (8 aa)
    println!("Test 1: Small Sequences (8 aa x 5 aa)");
    let small_seq1 = "AGSGDSAF";
    let small_seq2 = "AGSGD";
    let seq1 = Protein::from_string(small_seq1).unwrap();
    let seq2 = Protein::from_string(small_seq2).unwrap();

    // Scalar benchmark
    let scalar_sw = SmithWaterman::new().scalar_only();
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = scalar_sw.align(&seq1, &seq2);
    }
    let scalar_time = start.elapsed();
    println!("  Scalar (10k iterations): {:.2}ms", scalar_time.as_secs_f64() * 1000.0);
    println!("  Per alignment: {:.2}µs", scalar_time.as_secs_f64() * 1_000_000.0 / 10000.0);

    // SIMD benchmark
    let simd_sw = SmithWaterman::new().with_simd(true);
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = simd_sw.align(&seq1, &seq2);
    }
    let simd_time = start.elapsed();
    println!("  SIMD (10k iterations): {:.2}ms", simd_time.as_secs_f64() * 1000.0);
    println!("  Per alignment: {:.2}µs", simd_time.as_secs_f64() * 1_000_000.0 / 10000.0);
    
    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
    println!("  Speedup: {:.2}x\n", speedup);

    // Test case 2: Medium sequences (60 aa)
    println!("Test 2: Medium Sequences (~60 aa x ~55 aa)");
    let medium_seq1 = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRPQWQTFSTEPDLSAFSALGSDLWAYLB";
    let medium_seq2 = "MGHHEAELKPLAQSHATKHKIPVKYLEFISEAIIHVLHSRHGSQVLHSQEKE";
    let seq1 = Protein::from_string(medium_seq1).unwrap();
    let seq2 = Protein::from_string(medium_seq2).unwrap();

    // Scalar benchmark
    let scalar_sw = SmithWaterman::new().scalar_only();
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = scalar_sw.align(&seq1, &seq2);
    }
    let scalar_time = start.elapsed();
    println!("  Scalar (1k iterations): {:.2}ms", scalar_time.as_secs_f64() * 1000.0);
    println!("  Per alignment: {:.2}µs", scalar_time.as_secs_f64() * 1_000_000.0 / 1000.0);

    // SIMD benchmark
    let simd_sw = SmithWaterman::new().with_simd(true);
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = simd_sw.align(&seq1, &seq2);
    }
    let simd_time = start.elapsed();
    println!("  SIMD (1k iterations): {:.2}ms", simd_time.as_secs_f64() * 1000.0);
    println!("  Per alignment: {:.2}µs", simd_time.as_secs_f64() * 1_000_000.0 / 1000.0);
    
    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
    println!("  Speedup: {:.2}x\n", speedup);

    // Test case 3: Large sequences (180+ aa)
    println!("Test 3: Large Sequences (~180 aa x ~35 aa)");
    let large_seq1 = "MKTIIALSYIFCLVFADYKDDDKGSAGYQSGDYHKATNYNYSLTSYNNQDFVERDDDWGWKDHWLSIKGVSVSNSTITAPDLIQSPLGSCADVYNQLFPEGSWQKDISASKGVQTVLGSGIKKKDVPDHIGQEQIFGFPVNSRSKYQCYSVINGVV";
    let large_seq2 = "MKTIIALSYIFCLVFADYKDDDKGSAGYQSGD";
    let seq1 = Protein::from_string(large_seq1).unwrap();
    let seq2 = Protein::from_string(large_seq2).unwrap();

    // Scalar benchmark
    let scalar_sw = SmithWaterman::new().scalar_only();
    let start = Instant::now();
    for _ in 0..100 {
        let _ = scalar_sw.align(&seq1, &seq2);
    }
    let scalar_time = start.elapsed();
    println!("  Scalar (100 iterations): {:.2}ms", scalar_time.as_secs_f64() * 1000.0);
    println!("  Per alignment: {:.2}µs", scalar_time.as_secs_f64() * 1_000_000.0 / 100.0);

    // SIMD benchmark
    let simd_sw = SmithWaterman::new().with_simd(true);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = simd_sw.align(&seq1, &seq2);
    }
    let simd_time = start.elapsed();
    println!("  SIMD (100 iterations): {:.2}ms", simd_time.as_secs_f64() * 1000.0);
    println!("  Per alignment: {:.2}µs", simd_time.as_secs_f64() * 1_000_000.0 / 100.0);
    
    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
    println!("  Speedup: {:.2}x\n", speedup);

    println!("================================");
    println!("Performance validation complete!");
}
