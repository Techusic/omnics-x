use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use omicsx::alignment::SmithWaterman;
use omicsx::protein::Protein;

fn benchmark_smith_waterman_scalar_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("smith_waterman_comparison");
    
    // Test case 1: Small sequences (8 aa)
    let small_seq1_str = "AGSGDSAF";
    let small_seq2_str = "AGSGD";
    
    group.bench_function("scalar_small_sequences", |b| {
        b.iter(|| {
            let sw = SmithWaterman::new().scalar_only();
            let seq1 = Protein::from_string(black_box(small_seq1_str)).unwrap();
            let seq2 = Protein::from_string(black_box(small_seq2_str)).unwrap();
            let _ = sw.align(&seq1, &seq2);
        })
    });

    group.bench_function("simd_small_sequences", |b| {
        b.iter(|| {
            let sw = SmithWaterman::new().with_simd(true);
            let seq1 = Protein::from_string(black_box(small_seq1_str)).unwrap();
            let seq2 = Protein::from_string(black_box(small_seq2_str)).unwrap();
            let _ = sw.align(&seq1, &seq2);
        })
    });

    // Test case 2: Medium sequences (~60 aa - realistic protein domain)
    let medium_seq1_str = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRPQWQTFSTEPDLSAFSALGSDLWAYLB";
    let medium_seq2_str = "MGHHEAELKPLAQSHATKHKIPVKYLEFISEAIIHVLHSRHGSQVLHSQEKE";
    
    group.bench_function("scalar_medium_sequences", |b| {
        b.iter(|| {
            let sw = SmithWaterman::new().scalar_only();
            let seq1 = Protein::from_string(black_box(medium_seq1_str)).unwrap();
            let seq2 = Protein::from_string(black_box(medium_seq2_str)).unwrap();
            let _ = sw.align(&seq1, &seq2);
        })
    });

    group.bench_function("simd_medium_sequences", |b| {
        b.iter(|| {
            let sw = SmithWaterman::new().with_simd(true);
            let seq1 = Protein::from_string(black_box(medium_seq1_str)).unwrap();
            let seq2 = Protein::from_string(black_box(medium_seq2_str)).unwrap();
            let _ = sw.align(&seq1, &seq2);
        })
    });

    // Test case 3: Larger sequences (~200 aa - realistic enzyme)
    let large_seq1_str = "MKTIIALSYIFCLVFADYKDDDKGSAGYQSGDYHKATNYNYSLTSYNNQDFVERDDDWGWKDHWLSIKGVSVSNSTITAPDLIQSPLGSCADVYNQLFPEGSWQKDISASKGVQTVLGSGIKKKDVPDHIGQEQIFGFPVNSRSKYQCYSVINGVV";
    let large_seq2_str = "MKTIIALSYIFCLVFADYKDDDKGSAGYQSGD";
    
    group.bench_function("scalar_large_sequences", |b| {
        b.iter(|| {
            let sw = SmithWaterman::new().scalar_only();
            let seq1 = Protein::from_string(black_box(large_seq1_str)).unwrap();
            let seq2 = Protein::from_string(black_box(large_seq2_str)).unwrap();
            let _ = sw.align(&seq1, &seq2);
        })
    });

    group.bench_function("simd_large_sequences", |b| {
        b.iter(|| {
            let sw = SmithWaterman::new().with_simd(true);
            let seq1 = Protein::from_string(black_box(large_seq1_str)).unwrap();
            let seq2 = Protein::from_string(black_box(large_seq2_str)).unwrap();
            let _ = sw.align(&seq1, &seq2);
        })
    });

    group.finish();
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_analysis");
    
    // Benchmark with varying sequence lengths
    for seq_len in [10, 25, 50, 100, 200].iter() {
        let seq1_str = (0..*seq_len)
            .map(|i| match i % 10 {
                0 => 'A', 1 => 'G', 2 => 'S', 3 => 'D', 4 => 'E',
                5 => 'K', 6 => 'R', 7 => 'L', 8 => 'V', _ => 'I',
            })
            .collect::<String>();
        
        let seq2_str = (0..(*seq_len / 2))
            .map(|i| match i % 10 {
                0 => 'A', 1 => 'G', 2 => 'S', 3 => 'D', 4 => 'E',
                5 => 'K', 6 => 'R', 7 => 'L', 8 => 'V', _ => 'I',
            })
            .collect::<String>();
        
        group.bench_with_input(
            BenchmarkId::new("scalar_scaling", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    let sw = SmithWaterman::new().scalar_only();
                    let seq1 = Protein::from_string(black_box(&seq1_str)).unwrap();
                    let seq2 = Protein::from_string(black_box(&seq2_str)).unwrap();
                    let _ = sw.align(&seq1, &seq2);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_scaling", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    let sw = SmithWaterman::new().with_simd(true);
                    let seq1 = Protein::from_string(black_box(&seq1_str)).unwrap();
                    let seq2 = Protein::from_string(black_box(&seq2_str)).unwrap();
                    let _ = sw.align(&seq1, &seq2);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_smith_waterman_scalar_vs_simd, benchmark_scaling);
criterion_main!(benches);
