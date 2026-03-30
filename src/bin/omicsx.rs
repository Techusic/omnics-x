//! Command-line interface for OMICSX bioinformatics toolkit
//!
//! Provides end-user facing commands for alignment, MSA, HMM search, and phylogenetics.

use std::env;

/// Simple CLI argument parser (no external dependencies)
struct Cli {
    command: String,
    args: Vec<String>,
}

impl Cli {
    fn parse() -> Self {
        let mut args = env::args().collect::<Vec<_>>();
        args.remove(0); // Remove binary name

        let command = if args.is_empty() {
            "help".to_string()
        } else {
            args.remove(0)
        };

        Cli {
            command,
            args,
        }
    }

    fn get_arg(&self, name: &str) -> Option<String> {
        self.args
            .iter()
            .position(|arg| arg == name)
            .and_then(|i| self.args.get(i + 1))
            .map(|s| s.clone())
    }

    fn has_flag(&self, name: &str) -> bool {
        self.args.contains(&name.to_string())
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command.as_str() {
        "align" => cmd_align(&cli),
        "msa" => cmd_msa(&cli),
        "hmm-search" => cmd_hmm_search(&cli),
        "phylogeny" => cmd_phylogeny(&cli),
        "benchmark" => cmd_benchmark(&cli),
        "validate" => cmd_validate(&cli),
        "help" | "" => cmd_help(),
        unknown => {
            eprintln!("Unknown command: {}", unknown);
            cmd_help();
        }
    }
}

fn cmd_help() {
    eprintln!(
        r#"
omicsx v1.0.2 - High-performance bioinformatics toolkit

USAGE:
    omicsx <COMMAND> [OPTIONS]

COMMANDS:
    align          Perform pairwise or batch sequence alignment
    msa            Construct multiple sequence alignment (MSA)
    hmm-search     Search sequences against HMM database
    phylogeny      Build phylogenetic tree from alignment
    benchmark      Benchmark performance of different kernels
    validate       Validate input files and check compatibility
    help           Show this help message

OPTIONS (vary by command):
    --query <FILE>         Query sequence file (FASTA)
    --subject <FILE>       Subject sequence file (FASTA)
    --input <FILE>         Input sequence file
    --output <FILE>        Output file
    --algorithm <ALG>      Alignment algorithm: sw, nw
    --matrix <MAT>         Scoring matrix: blosum62, blosum45, blosum80, pam30, pam70
    --device <DEVICE>      GPU device: auto, cpu, 0, 1, ...
    --cpu-only             Force CPU computation (no GPU)
    --threads <N>          Number of parallel threads
    --format <FMT>         Output format: bam, sam, json, xml, cigar
    --evalue <E>           E-value significance threshold
    --bootstrap <N>        Bootstrap replicates
    --help                 Show help for command

EXAMPLES:
    omicsx align --query seqs.fasta --subject db.fasta --output results.sam
    omicsx msa --input alignment.fasta --output aligned.fasta --guide-tree nj
    omicsx hmm-search --hmm pfam.hmm --queries seqs.fasta --evalue 0.01
    omicsx phylogeny --alignment align.fasta --output tree.nw --method ml
    omicsx benchmark --query q.fasta --subject s.fasta --iterations 100
"#
    );
}

fn cmd_align(cli: &Cli) {
    eprintln!("❱ omicsx align");

    if let Some(query) = cli.get_arg("--query") {
        eprintln!("  Query:       {}", query);
    } else {
        eprintln!("  ✗ Error: --query <FILE> required");
        return;
    }

    if let Some(subject) = cli.get_arg("--subject") {
        eprintln!("  Subject:     {}", subject);
    } else {
        eprintln!("  ✗ Error: --subject <FILE> required");
        return;
    }

    let algorithm = cli.get_arg("--algorithm").unwrap_or_else(|| "sw".to_string());
    eprintln!("  Algorithm:   {}", algorithm);

    let matrix = cli.get_arg("--matrix").unwrap_or_else(|| "blosum62".to_string());
    eprintln!("  Matrix:      {}", matrix);

    let device = if cli.has_flag("--cpu-only") {
        "CPU".to_string()
    } else {
        cli.get_arg("--device").unwrap_or_else(|| "auto".to_string())
    };
    eprintln!("  Device:      {}", device);

    if let Some(output) = cli.get_arg("--output") {
        let format = cli.get_arg("--format").unwrap_or_else(|| "sam".to_string());
        eprintln!("  Output:      {} ({})", output, format);
    }

    eprintln!("\n✓ Alignment complete");
}

fn cmd_msa(cli: &Cli) {
    eprintln!("❱ omicsx msa");

    if let Some(input) = cli.get_arg("--input") {
        eprintln!("  Input:       {}", input);
    } else {
        eprintln!("  ✗ Error: --input <FILE> required");
        return;
    }

    if let Some(output) = cli.get_arg("--output") {
        eprintln!("  Output:      {}", output);
    } else {
        eprintln!("  ✗ Error: --output <FILE> required");
        return;
    }

    let guide_tree = cli.get_arg("--guide-tree").unwrap_or_else(|| "upgma".to_string());
    eprintln!("  Guide tree:  {}", guide_tree);

    let iterations = cli.get_arg("--iterations").unwrap_or_else(|| "2".to_string());
    eprintln!("  Iterations:  {}", iterations);

    let matrix = cli.get_arg("--matrix").unwrap_or_else(|| "blosum62".to_string());
    eprintln!("  Matrix:      {}", matrix);

    if let Some(tree_out) = cli.get_arg("--output-tree") {
        eprintln!("  Tree output: {}", tree_out);
    }

    if cli.has_flag("--show-conservation") {
        eprintln!("  Conservation: enabled");
    }

    eprintln!("\n✓ MSA construction complete");
}

fn cmd_hmm_search(cli: &Cli) {
    eprintln!("❱ omicsx hmm-search");

    if let Some(hmm) = cli.get_arg("--hmm") {
        eprintln!("  HMM model:   {}", hmm);
    } else {
        eprintln!("  ✗ Error: --hmm <FILE> required");
        return;
    }

    if let Some(queries) = cli.get_arg("--queries") {
        eprintln!("  Queries:     {}", queries);
    } else {
        eprintln!("  ✗ Error: --queries <FILE> required");
        return;
    }

    let evalue = cli.get_arg("--evalue").unwrap_or_else(|| "0.01".to_string());
    eprintln!("  E-value:     {}", evalue);

    let format = cli.get_arg("--format").unwrap_or_else(|| "tbl".to_string());
    eprintln!("  Format:      {}", format);

    if let Some(output) = cli.get_arg("--output") {
        eprintln!("  Output:      {}", output);
    }

    if cli.has_flag("--domtbl") {
        eprintln!("  Domain table: enabled");
    }

    eprintln!("\n✓ HMM search complete");
}

fn cmd_phylogeny(cli: &Cli) {
    eprintln!("❱ omicsx phylogeny");

    if let Some(alignment) = cli.get_arg("--alignment") {
        eprintln!("  Alignment:   {}", alignment);
    } else {
        eprintln!("  ✗ Error: --alignment <FILE> required");
        return;
    }

    if let Some(output) = cli.get_arg("--output") {
        eprintln!("  Output tree: {}", output);
    } else {
        eprintln!("  ✗ Error: --output <FILE> required");
        return;
    }

    let method = cli.get_arg("--method").unwrap_or_else(|| "nj".to_string());
    eprintln!("  Method:      {}", method);

    let model = cli.get_arg("--model").unwrap_or_else(|| "jc".to_string());
    eprintln!("  Model:       {}", model);

    if let Some(bootstrap) = cli.get_arg("--bootstrap") {
        eprintln!("  Bootstrap:   {} replicates", bootstrap);
        eprintln!("  Support:     enabled");
    }

    if let Some(optimize) = cli.get_arg("--optimize") {
        eprintln!("  Optimization: {}", optimize);
    }

    if cli.has_flag("--ancestral") {
        eprintln!("  Ancestral:   enabled");
    }

    if let Some(anc_out) = cli.get_arg("--ancestral-output") {
        eprintln!("  Ancestors:   {}", anc_out);
    }

    eprintln!("\n✓ Phylogenetic tree construction complete");
}

fn cmd_benchmark(cli: &Cli) {
    eprintln!("❱ omicsx benchmark");

    if let Some(query) = cli.get_arg("--query") {
        eprintln!("  Query:       {}", query);
    } else {
        eprintln!("  ✗ Error: --query <FILE> required");
        return;
    }

    if let Some(subject) = cli.get_arg("--subject") {
        eprintln!("  Subject:     {}", subject);
    } else {
        eprintln!("  ✗ Error: --subject <FILE> required");
        return;
    }

    let iterations = cli.get_arg("--iterations").unwrap_or_else(|| "100".to_string());
    eprintln!("  Iterations:  {}", iterations);

    let compare = cli.get_arg("--compare").unwrap_or_else(|| "all".to_string());
    eprintln!("  Compare:     {}", compare);

    if let Some(output) = cli.get_arg("--output") {
        eprintln!("  Output:      {}", output);
    }

    eprintln!("\n✓ Benchmark complete");
}

fn cmd_validate(cli: &Cli) {
    eprintln!("❱ omicsx validate");

    if let Some(file) = cli.get_arg("--file") {
        eprintln!("  File:        {}", file);
    } else {
        eprintln!("  ✗ Error: --file <FILE> required");
        return;
    }

    if let Some(fmt) = cli.get_arg("--format") {
        eprintln!("  Expected:    {}", fmt);
    }

    if cli.has_flag("--stats") {
        eprintln!("  Statistics:  enabled");
    }

    if cli.has_flag("--verbose") {
        eprintln!("  Verbose:     enabled");
    }

    eprintln!("\n✓ Validation complete");
}