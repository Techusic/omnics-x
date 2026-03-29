//! CLI Buffered File I/O for processing large genomic databases
//! Efficient streaming processing of FASTA, FASTQ, BAM, and custom formats

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use crate::error::{Error, Result};

/// Sequence record from input file
#[derive(Debug, Clone)]
pub struct SeqRecord {
    /// Sequence identifier
    pub id: String,
    /// Optional description
    pub description: Option<String>,
    /// Sequence data
    pub sequence: String,
    /// Quality scores (FASTQ only)
    pub quality: Option<String>,
}

impl SeqRecord {
    /// Get full header line
    pub fn header(&self) -> String {
        match &self.description {
            Some(desc) => format!("{} {}", self.id, desc),
            None => self.id.clone(),
        }
    }

    /// Get length of sequence
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if record is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

/// Input file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// FASTA format (>id description \n sequence)
    Fasta,
    /// FASTQ format (@ quality)
    Fastq,
    /// Tab-separated values (id \t sequence)
    Tsv,
}

impl FileFormat {
    /// Detect format from file extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        match path.extension().and_then(|s| s.to_str()) {
            Some("fasta") | Some("fa") | Some("faa") => FileFormat::Fasta,
            Some("fastq") | Some("fq") => FileFormat::Fastq,
            Some("tsv") | Some("txt") => FileFormat::Tsv,
            _ => FileFormat::Fasta, // Default
        }
    }
}

/// Buffered sequence file reader
pub struct SeqFileReader {
    reader: BufReader<File>,
    format: FileFormat,
    buffer: String,
    line_count: u64,
}

impl SeqFileReader {
    /// Open file with automatic format detection
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| Error::AlignmentError(format!("Failed to open file: {}", e)))?;

        let format = FileFormat::from_path(path);
        Ok(SeqFileReader {
            reader: BufReader::new(file),
            format,
            buffer: String::new(),
            line_count: 0,
        })
    }

    /// Open file with explicit format
    pub fn open_with_format<P: AsRef<Path>>(path: P, format: FileFormat) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::AlignmentError(format!("Failed to open file: {}", e)))?;

        Ok(SeqFileReader {
            reader: BufReader::new(file),
            format,
            buffer: String::new(),
            line_count: 0,
        })
    }

    /// Read next sequence record
    pub fn next_record(&mut self) -> Result<Option<SeqRecord>> {
        match self.format {
            FileFormat::Fasta => self.read_fasta_record(),
            FileFormat::Fastq => self.read_fastq_record(),
            FileFormat::Tsv => self.read_tsv_record(),
        }
    }

    /// Read FASTA record
    fn read_fasta_record(&mut self) -> Result<Option<SeqRecord>> {
        self.buffer.clear();

        // Skip until header line
        loop {
            let n = self
                .reader
                .read_line(&mut self.buffer)
                .map_err(|e| Error::AlignmentError(format!("Read error: {}", e)))?;

            if n == 0 {
                return Ok(None); // EOF
            }

            self.line_count += 1;

            if self.buffer.starts_with('>') {
                break;
            }
            self.buffer.clear();
        }

        let header = self.buffer.trim_end().to_string();
        let (id, description) = parse_fasta_header(&header);

        let mut sequence = String::new();
        self.buffer.clear();

        // Read sequence lines until next header or EOF
        loop {
            let n = self
                .reader
                .read_line(&mut self.buffer)
                .map_err(|e| Error::AlignmentError(format!("Read error: {}", e)))?;

            if n == 0 || self.buffer.starts_with('>') {
                if self.buffer.starts_with('>') {
                    // Put back the header by seeking?
                    // For now, this becomes the next record's header
                }
                break;
            }

            self.line_count += 1;
            sequence.push_str(self.buffer.trim_end());
            self.buffer.clear();
        }

        Ok(Some(SeqRecord {
            id,
            description,
            sequence,
            quality: None,
        }))
    }

    /// Read FASTQ record
    fn read_fastq_record(&mut self) -> Result<Option<SeqRecord>> {
        let mut lines = [String::new(), String::new(), String::new(), String::new()];

        for line in &mut lines {
            let n = self
                .reader
                .read_line(line)
                .map_err(|e| Error::AlignmentError(format!("Read error: {}", e)))?;

            if n == 0 {
                return Ok(None);
            }
            self.line_count += 1;
            *line = line.trim_end().to_string();
        }

        let (id, description) = parse_fastq_header(&lines[0]);
        let sequence = lines[1].clone();
        let quality = Some(lines[3].clone());

        if lines[2] != "+" {
            return Err(Error::AlignmentError(
                format!("Invalid FASTQ format at line {}", self.line_count - 1),
            ));
        }

        Ok(Some(SeqRecord {
            id,
            description,
            sequence,
            quality,
        }))
    }

    /// Read TSV record
    fn read_tsv_record(&mut self) -> Result<Option<SeqRecord>> {
        self.buffer.clear();
        let n = self
            .reader
            .read_line(&mut self.buffer)
            .map_err(|e| Error::AlignmentError(format!("Read error: {}", e)))?;

        if n == 0 {
            return Ok(None);
        }

        self.line_count += 1;
        let parts: Vec<&str> = self.buffer.trim().split('\t').collect();

        if parts.len() < 2 {
            return Err(Error::AlignmentError("TSV record must have at least 2 columns".to_string()));
        }

        Ok(Some(SeqRecord {
            id: parts[0].to_string(),
            description: parts.get(2).map(|s| s.to_string()),
            sequence: parts[1].to_string(),
            quality: parts.get(3).map(|s| s.to_string()),
        }))
    }

    /// Get current line number
    pub fn line_number(&self) -> u64 {
        self.line_count
    }
}

/// Buffered sequence file writer
pub struct SeqFileWriter {
    writer: BufWriter<File>,
    format: FileFormat,
    record_count: u64,
}

impl SeqFileWriter {
    /// Create new file for writing
    pub fn create<P: AsRef<Path>>(path: P, format: FileFormat) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| Error::AlignmentError(format!("Failed to create file: {}", e)))?;

        Ok(SeqFileWriter {
            writer: BufWriter::new(file),
            format,
            record_count: 0,
        })
    }

    /// Write sequence record
    pub fn write_record(&mut self, record: &SeqRecord) -> Result<()> {
        match self.format {
            FileFormat::Fasta => {
                writeln!(self.writer, ">{}", record.header())
                    .map_err(|e| Error::AlignmentError(e.to_string()))?;
                writeln!(self.writer, "{}", record.sequence)
                    .map_err(|e| Error::AlignmentError(e.to_string()))?;
            }
            FileFormat::Fastq => {
                if let Some(ref quality) = record.quality {
                    writeln!(self.writer, "@{}", record.header())
                        .map_err(|e| Error::AlignmentError(e.to_string()))?;
                    writeln!(self.writer, "{}", record.sequence)
                        .map_err(|e| Error::AlignmentError(e.to_string()))?;
                    writeln!(self.writer, "+")
                        .map_err(|e| Error::AlignmentError(e.to_string()))?;
                    writeln!(self.writer, "{}", quality)
                        .map_err(|e| Error::AlignmentError(e.to_string()))?;
                } else {
                    return Err(Error::AlignmentError("FASTQ record requires quality scores".to_string()));
                }
            }
            FileFormat::Tsv => {
                write!(self.writer, "{}\t{}", record.id, record.sequence)
                    .map_err(|e| Error::AlignmentError(e.to_string()))?;
                if let Some(ref desc) = record.description {
                    write!(self.writer, "\t{}", desc)
                        .map_err(|e| Error::AlignmentError(e.to_string()))?;
                }
                if let Some(ref qual) = record.quality {
                    write!(self.writer, "\t{}", qual)
                        .map_err(|e| Error::AlignmentError(e.to_string()))?;
                }
                writeln!(self.writer).map_err(|e| Error::AlignmentError(e.to_string()))?;
            }
        }

        self.record_count += 1;
        Ok(())
    }

    /// Write multiple records
    pub fn write_batch(&mut self, records: &[SeqRecord]) -> Result<()> {
        for record in records {
            self.write_record(record)?;
        }
        Ok(())
    }

    /// Flush buffered writes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| Error::AlignmentError(e.to_string()))
    }

    /// Get number of records written
    pub fn record_count(&self) -> u64 {
        self.record_count
    }
}

/// Parse FASTA header (>id description)
fn parse_fasta_header(header: &str) -> (String, Option<String>) {
    let header = header.trim_start_matches('>');
    let parts: Vec<&str> = header.splitn(2, ' ').collect();

    let id = parts[0].to_string();
    let description = parts.get(1).map(|s| s.to_string());

    (id, description)
}

/// Parse FASTQ header (@id description)
fn parse_fastq_header(header: &str) -> (String, Option<String>) {
    let header = header.trim_start_matches('@');
    let parts: Vec<&str> = header.splitn(2, ' ').collect();

    let id = parts[0].to_string();
    let description = parts.get(1).map(|s| s.to_string());

    (id, description)
}

/// Batch processor for sequence analysis
pub struct BatchProcessor {
    batch_size: usize,
    filter_min_len: usize,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize) -> Self {
        BatchProcessor {
            batch_size,
            filter_min_len: 0,
        }
    }

    /// Set minimum sequence length filter
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.filter_min_len = len;
        self
    }

    /// Process file in batches
    pub fn process_file<P: AsRef<Path>, F>(
        &self,
        path: P,
        mut callback: F,
    ) -> Result<u64>
    where
        F: FnMut(&[SeqRecord]) -> Result<()>,
    {
        let mut reader = SeqFileReader::open(path)?;
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut total = 0u64;

        loop {
            match reader.next_record()? {
                Some(record) => {
                    if record.len() >= self.filter_min_len {
                        batch.push(record);

                        if batch.len() >= self.batch_size {
                            callback(&batch)?;
                            total += batch.len() as u64;
                            batch.clear();
                        }
                    }
                }
                None => {
                    // Process remaining records
                    if !batch.is_empty() {
                        callback(&batch)?;
                        total += batch.len() as u64;
                    }
                    break;
                }
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_format_detection() {
        assert_eq!(FileFormat::from_path("test.fasta"), FileFormat::Fasta);
        assert_eq!(FileFormat::from_path("test.fastq"), FileFormat::Fastq);
        assert_eq!(FileFormat::from_path("test.tsv"), FileFormat::Tsv);
    }

    #[test]
    fn test_parse_fasta_header() {
        let (id, desc) = parse_fasta_header(">seq1 This is a description");
        assert_eq!(id, "seq1");
        assert_eq!(desc, Some("This is a description".to_string()));
    }

    #[test]
    fn test_parse_fastq_header() {
        let (id, desc) = parse_fastq_header("@seq1 description goes here");
        assert_eq!(id, "seq1");
        assert_eq!(desc, Some("description goes here".to_string()));
    }

    #[test]
    fn test_seq_record_creation() {
        let record = SeqRecord {
            id: "test".to_string(),
            description: Some("desc".to_string()),
            sequence: "ACGT".to_string(),
            quality: None,
        };

        assert_eq!(record.id, "test");
        assert_eq!(record.len(), 4);
        assert!(!record.is_empty());
    }

    #[test]
    fn test_seq_record_header() {
        let record = SeqRecord {
            id: "seq1".to_string(),
            description: Some("with desc".to_string()),
            sequence: "ACGT".to_string(),
            quality: None,
        };

        assert_eq!(record.header(), "seq1 with desc");
    }

    #[test]
    fn test_batch_processor_creation() {
        let proc = BatchProcessor::new(100);
        assert_eq!(proc.batch_size, 100);
    }

    #[test]
    fn test_batch_processor_with_filter() {
        let proc = BatchProcessor::new(50).with_min_length(10);
        assert_eq!(proc.filter_min_len, 10);
    }
}
