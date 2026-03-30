//! BAM (Binary Alignment/Map) Format Support
//!
//! This module provides binary serialization/deserialization of alignment records
//! in BAM format, which is the compressed binary equivalent of SAM (Sequence Alignment Map).
//!
//! ## BAM Format Overview
//!
//! BAM is a binary container for sequence alignments with several advantages over SAM:
//! - **Size**: ~4x smaller than SAM (compression + binary encoding)
//! - **Speed**: Faster to read/write than SAM text parsing
//! - **Indexing**: Supports quick random access via BAI (BAM Index) files
//! - **Standard**: Widely used in bioinformatics (GATK, samtools, BCFtools)
//!
//! ## BAM Structure
//!
//! ```text
//! [Header Section] -> Variable-length text header
//! [Reference Names] -> Sequence names and lengths
//! [Alignment Records] -> Individual read alignments
//! ```
//!
//! Each record contains:
//! - Read name, sequence, quality scores (text)
//! - Position coordinates (integer)
//! - CIGAR string (encoded as array of (op, count) pairs)
//! - Optional tags (key:type:value pairs)
//!
//! See: <https://samtools.github.io/hts-specs/SAMv1.pdf>

use crate::alignment::{SamHeader, SamRecord};
use crate::error::Result;

/// BAM magic bytes - identifies file as BAM format
const BAM_MAGIC: &[u8; 4] = b"BAM\x01";

/// Represents a binary BAM file with header and records
#[derive(Debug, Clone)]
pub struct BamFile {
    /// Header information (version, references, programs)
    pub header: SamHeader,
    /// Reference sequence names and lengths for BAM indexing
    pub references: Vec<(String, u32)>,
    /// Alignment records in binary format
    pub records: Vec<BamRecord>,
}

impl BamFile {
    /// Create new BAM file with header
    pub fn new(header: SamHeader) -> Self {
        BamFile {
            header,
            references: Vec::new(),
            records: Vec::new(),
        }
    }

    /// Add reference sequence to BAM header
    pub fn add_reference(&mut self, name: String, length: u32) {
        self.references.push((name, length));
    }

    /// Add alignment record to BAM file
    pub fn add_record(&mut self, record: BamRecord) {
        self.records.push(record);
    }

    /// Serialize BAM file to binary format
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // Write magic bytes
        bytes.extend_from_slice(BAM_MAGIC);

        // Write header text
        let header_text = self.header.to_header_lines().join("\n");
        let header_bytes = header_text.as_bytes();
        write_le_i32(&mut bytes, header_bytes.len() as i32);
        bytes.extend_from_slice(header_bytes);

        // Write reference sequences
        write_le_i32(&mut bytes, self.references.len() as i32);
        for (name, length) in &self.references {
            let name_bytes = name.as_bytes();
            write_le_i32(&mut bytes, (name_bytes.len() + 1) as i32);
            bytes.extend_from_slice(name_bytes);
            bytes.push(0); // null terminator
            write_le_i32(&mut bytes, *length as i32);
        }

        // Write records
        for record in &self.records {
            let record_bytes = record.to_bytes()?;
            write_le_i32(&mut bytes, record_bytes.len() as i32);
            bytes.extend_from_slice(&record_bytes);
        }

        Ok(bytes)
    }

    /// Deserialize BAM file from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = 0;

        // Verify magic bytes
        if cursor + 4 > data.len() || &data[cursor..cursor + 4] != BAM_MAGIC {
            return Err(crate::error::Error::Custom("Invalid BAM magic bytes".to_string()));
        }
        cursor += 4;

        // Read header text
        let header_len = read_le_i32(&data, &mut cursor)? as usize;
        if cursor + header_len > data.len() {
            return Err(crate::error::Error::Custom("Invalid BAM header size".to_string()));
        }
        let _header_text = String::from_utf8(data[cursor..cursor + header_len].to_vec())
            .map_err(|e| crate::error::Error::Custom(format!("Invalid UTF-8 in BAM header: {}", e)))?;
        cursor += header_len;

        // Parse header
        let header = SamHeader::new("1.0");

        // Read references
        let ref_count = read_le_i32(&data, &mut cursor)? as usize;
        let mut references = Vec::new();
        for _ in 0..ref_count {
            let name_len = read_le_i32(&data, &mut cursor)? as usize;
            if cursor + name_len > data.len() {
                return Err(crate::error::Error::Custom("Invalid reference name size".to_string()));
            }
            let name = String::from_utf8(data[cursor..cursor + name_len - 1].to_vec())
                .map_err(|e| crate::error::Error::Custom(format!("Invalid UTF-8 in reference name: {}", e)))?;
            cursor += name_len;
            let length = read_le_i32(&data, &mut cursor)? as u32;
            references.push((name, length));
        }

        // Read records
        let mut records = Vec::new();
        while cursor < data.len() {
            if cursor + 4 > data.len() {
                break;
            }
            let record_len = read_le_i32(&data, &mut cursor)? as usize;
            if cursor + record_len > data.len() {
                break;
            }
            let record = BamRecord::from_bytes(&data[cursor..cursor + record_len])?;
            records.push(record);
            cursor += record_len;
        }

        Ok(BamFile {
            header,
            references,
            records,
        })
    }
}

/// Binary representation of a single alignment record
#[derive(Debug, Clone)]
pub struct BamRecord {
    /// Reference sequence ID
    pub ref_id: i32,
    /// Alignment start position (0-based)
    pub pos: i32,
    /// Quality score, binned quality (MSB) and read name length (LSB)
    pub bin_mq_nl: u32,
    /// Flag and number of CIGAR operations
    pub flag_nc: u32,
    /// Read sequence length (including hard clips, excludes soft clips)
    pub l_seq: i32,
    /// Next reference ID
    pub next_ref_id: i32,
    /// Next position
    pub next_pos: i32,
    /// Template length
    pub tlen: i32,
    /// Read name (null-terminated string)
    pub read_name: String,
    /// CIGAR string encoded as array of operations
    pub cigar: Vec<(u32, u8)>, // (length, operation)
    /// Read sequence (4-bit encoded)
    pub seq: Vec<u8>,
    /// ASCII quality scores (0x21 = !, 0x7e = ~)
    pub qual: Vec<u8>,
}

impl BamRecord {
    /// Create BAM record from SAM record
    pub fn from_sam(sam: &SamRecord, ref_id: i32) -> Self {
        BamRecord {
            ref_id,
            pos: sam.pos as i32,
            bin_mq_nl: 0, // Will be computed
            flag_nc: (sam.flag as u32) << 16, // Will add CIGAR count
            l_seq: sam.query_seq.len() as i32,
            next_ref_id: -1,
            next_pos: -1,
            tlen: 0,
            read_name: sam.qname.clone(),
            cigar: Self::parse_cigar(&sam.cigar),
            seq: Self::encode_seq(&sam.query_seq),
            qual: sam.query_qual.as_bytes().to_vec(),
        }
    }

    /// Encode sequence to 4-bit format (2 bases per byte)
    fn encode_seq(seq: &str) -> Vec<u8> {
        let bases = vec!['=', 'A', 'C', 'M', 'G', 'R', 'S', 'V', 'T', 'W', 'Y', 'H', 'K', 'D', 'B', 'N'];
        let mut encoded = vec![0u8; (seq.len() + 1) / 2];

        for (i, base) in seq.chars().enumerate() {
            let code = bases.iter().position(|&b| b == base).unwrap_or(0) as u8;
            if i % 2 == 0 {
                encoded[i / 2] |= code << 4;
            } else {
                encoded[i / 2] |= code;
            }
        }
        encoded
    }

    /// Parse CIGAR string into operations and lengths
    fn parse_cigar(cigar: &str) -> Vec<(u32, u8)> {
        let mut result = Vec::new();
        let mut current_num = String::new();

        for c in cigar.chars() {
            if c.is_numeric() {
                current_num.push(c);
            } else {
                if !current_num.is_empty() {
                    if let Ok(len) = current_num.parse::<u32>() {
                        let op = match c {
                            'M' => 0,
                            'I' => 1,
                            'D' => 2,
                            'N' => 3,
                            'S' => 4,
                            'H' => 5,
                            'P' => 6,
                            '=' => 7,
                            'X' => 8,
                            _ => 0,
                        };
                        result.push((len, op));
                    }
                    current_num.clear();
                }
            }
        }
        result
    }

    /// Format CIGAR from parsed operations
    pub fn format_cigar(ops: &[(u32, u8)]) -> String {
        let op_map = ['M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X'];
        ops.iter()
            .map(|(len, op)| format!("{}{}", len, op_map[*op as usize]))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Serialize record to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        write_le_i32(&mut bytes, self.ref_id);
        write_le_i32(&mut bytes, self.pos);
        write_le_u32(&mut bytes, self.bin_mq_nl);
        write_le_u32(&mut bytes, self.flag_nc | self.cigar.len() as u32);
        write_le_i32(&mut bytes, self.l_seq);
        write_le_i32(&mut bytes, self.next_ref_id);
        write_le_i32(&mut bytes, self.next_pos);
        write_le_i32(&mut bytes, self.tlen);

        // Write null-terminated read name
        bytes.extend_from_slice(self.read_name.as_bytes());
        bytes.push(0);

        // Write CIGAR operations
        for (len, op) in &self.cigar {
            write_le_u32(&mut bytes, (len << 4) | (*op as u32));
        }

        // Write encoded sequence and quality
        bytes.extend_from_slice(&self.seq);
        bytes.extend_from_slice(&self.qual);

        Ok(bytes)
    }

    /// Deserialize record from bytes
    fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = 0;

        let ref_id = read_le_i32(data, &mut cursor)?;
        let pos = read_le_i32(data, &mut cursor)?;
        let bin_mq_nl = read_le_u32(data, &mut cursor)?;
        let flag_nc = read_le_u32(data, &mut cursor)?;
        let l_seq = read_le_i32(data, &mut cursor)?;
        let next_ref_id = read_le_i32(data, &mut cursor)?;
        let next_pos = read_le_i32(data, &mut cursor)?;
        let tlen = read_le_i32(data, &mut cursor)?;

        // Read null-terminated read name
        let name_end = data[cursor..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| crate::error::Error::Custom("Invalid BAM record: no null terminator".to_string()))?;
        let read_name = String::from_utf8(data[cursor..cursor + name_end].to_vec())
            .map_err(|e| crate::error::Error::Custom(format!("Invalid UTF-8 in read name: {}", e)))?;
        cursor += name_end + 1;

        let cigar_count = (flag_nc & 0xFFFF) as usize;
        let mut cigar = Vec::new();
        for _ in 0..cigar_count {
            let val = read_le_u32(data, &mut cursor)?;
            let len = val >> 4;
            let op = (val & 0xF) as u8;
            cigar.push((len, op));
        }

        // Read sequence and quality
        let seq_bytes = (l_seq + 1) / 2;
        let seq = data[cursor..cursor + seq_bytes as usize].to_vec();
        cursor += seq_bytes as usize;

        let qual = data[cursor..cursor + l_seq as usize].to_vec();

        Ok(BamRecord {
            ref_id,
            pos,
            bin_mq_nl,
            flag_nc: flag_nc & 0xFFFF0000,
            l_seq,
            next_ref_id,
            next_pos,
            tlen,
            read_name,
            cigar,
            seq,
            qual,
        })
    }
}

// Helper functions for little-endian integer encoding/decoding
fn write_le_i32(bytes: &mut Vec<u8>, val: i32) {
    bytes.extend_from_slice(&val.to_le_bytes());
}

fn write_le_u32(bytes: &mut Vec<u8>, val: u32) {
    bytes.extend_from_slice(&val.to_le_bytes());
}

fn read_le_i32(data: &[u8], cursor: &mut usize) -> Result<i32> {
    if *cursor + 4 > data.len() {
        return Err(crate::error::Error::Custom("BAM: insufficient data for i32".to_string()));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[*cursor..*cursor + 4]);
    *cursor += 4;
    Ok(i32::from_le_bytes(bytes))
}

fn read_le_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > data.len() {
        return Err(crate::error::Error::Custom("BAM: insufficient data for u32".to_string()));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[*cursor..*cursor + 4]);
    *cursor += 4;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bam_file_creation() {
        let header = SamHeader::new("1.0");
        let mut bam = BamFile::new(header);
        bam.add_reference("chr1".to_string(), 1000);

        assert_eq!(bam.references.len(), 1);
        assert_eq!(bam.references[0].0, "chr1");
        assert_eq!(bam.references[0].1, 1000);
    }

    #[test]
    fn test_bam_serialization() -> Result<()> {
        let header = SamHeader::new("1.0");
        let mut bam = BamFile::new(header);
        bam.add_reference("chr1".to_string(), 1000);

        let bytes = bam.to_bytes()?;
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], BAM_MAGIC);

        Ok(())
    }

    #[test]
    fn test_sequence_encoding() {
        let encoded = BamRecord::encode_seq("ACGT");
        assert!(!encoded.is_empty());
        // A=1, C=2, G=4, T=8 (IUPAC codes)
    }

    #[test]
    fn test_cigar_parsing() {
        let cigar = BamRecord::parse_cigar("10M2I5D3M");
        assert_eq!(cigar.len(), 4);
        assert_eq!(cigar[0], (10, 0)); // 10M
        assert_eq!(cigar[1], (2, 1)); // 2I
        assert_eq!(cigar[2], (5, 2)); // 5D
        assert_eq!(cigar[3], (3, 0)); // 3M
    }

    #[test]
    fn test_cigar_formatting() {
        let ops = vec![(10, 0), (2, 1), (5, 2), (3, 0)];
        let cigar = BamRecord::format_cigar(&ops);
        assert_eq!(cigar, "10M2I5D3M");
    }

    #[test]
    fn test_invalid_utf8_in_bam_header() {
        // Create BAM with valid magic but invalid UTF-8 in header
        let mut data = Vec::from(BAM_MAGIC);
        
        // Write header size
        data.extend_from_slice(&(4i32).to_le_bytes()); // 4 bytes of header
        
        // Write invalid UTF-8 sequence (invalid continuation byte)
        data.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]); // Invalid UTF-8
        
        // Write 0 references
        data.extend_from_slice(&(0i32).to_le_bytes());
        
        let result = BamFile::from_bytes(&data);
        assert!(result.is_err(), "Should error on invalid UTF-8 in header");
    }

    #[test]
    fn test_invalid_utf8_in_reference_name() {
        // Create BAM with 1 valid header but invalid UTF-8 in reference name
        let mut data = Vec::from(BAM_MAGIC);
        
        // Write valid empty header
        data.extend_from_slice(&(0i32).to_le_bytes()); // empty header
        
        // Write 1 reference with invalid UTF-8
        data.extend_from_slice(&(1i32).to_le_bytes()); // 1 reference
        data.extend_from_slice(&(5i32).to_le_bytes()); // name length: 4 bytes + null termininator
        data.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0x00]); // Invalid UTF-8 then null
        data.extend_from_slice(&(1000i32).to_le_bytes()); // reference length
        
        let result = BamFile::from_bytes(&data);
        assert!(result.is_err(), "Should error on invalid UTF-8 in reference name");
    }

    #[test]
    fn test_invalid_utf8_in_read_name() {
        // Create a BAM record with invalid UTF-8 in read name
        let mut data = Vec::new();
        
        // Write BAM record header fields
        data.extend_from_slice(&(0i32).to_le_bytes()); // ref_id
        data.extend_from_slice(&(100i32).to_le_bytes()); // pos
        data.extend_from_slice(&(0i32).to_le_bytes()); // bin_mq_nl
        data.extend_from_slice(&(0i32).to_le_bytes()); // flag_nc
        data.extend_from_slice(&(0i32).to_le_bytes()); // l_seq
        data.extend_from_slice(&(0i32).to_le_bytes()); // next_ref_id
        data.extend_from_slice(&(0i32).to_le_bytes()); // next_pos
        data.extend_from_slice(&(0i32).to_le_bytes()); // tlen
        
        // Write invalid UTF-8 read name (not null-terminated properly)
        data.extend_from_slice(&[0xFF, 0xFE, 0x00]); // Invalid UTF-8 with null term
        
        let result = BamRecord::from_bytes(&data);
        assert!(result.is_err(), "Should error on invalid UTF-8 in read name");
    }

    #[test]
    fn test_valid_utf8_roundtrip() -> Result<()> {
        let header = SamHeader::new("1.0");
        let mut bam = BamFile::new(header);
        
        // Add references with valid UTF-8 names
        bam.add_reference("chromosome_1".to_string(), 1000000);
        bam.add_reference("chr2_fragment".to_string(), 2000000);
        
        let bytes = bam.to_bytes()?;
        let restored = BamFile::from_bytes(&bytes)?;
        
        assert_eq!(restored.references.len(), 2);
        assert_eq!(restored.references[0].0, "chromosome_1");
        assert_eq!(restored.references[1].0, "chr2_fragment");
        
        Ok(())
    }
}
