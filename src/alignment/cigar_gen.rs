//! Production-Grade CIGAR String Generation from DP Backtracking
//!
//! Implements complete trace-back algorithms to generate accurate SAM/BAM formatted CIGAR strings
//! with full MDNP operation support. This ensures compatibility with standard bioinformatics tools.
//!
//! # CIGAR Operations
//! - M: Sequence match (can be mismatch)
//! - I: Insertion to reference
//! - D: Deletion from reference
//! - N: Skipped region from reference
//! - S: Soft clipping (clipped sequences present in SEQ)
//! - H: Hard clipping (clipped sequences NOT present in SEQ)
//! - =: Sequence match
//! - X: Sequence mismatch
//! - P: Padding (silent deletion from padded reference)

use std::fmt;
use serde::{Deserialize, Serialize};

/// CIGAR operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CigarOp {
    /// Alignment match (M)
    Match,
    /// Insertion to reference (I)
    Insertion,
    /// Deletion from reference (D)
    Deletion,
    /// Skipped region from reference (N)
    Skip,
    /// Soft clipping (S)
    SoftClip,
    /// Hard clipping (H)
    HardClip,
    /// Sequence match (=)
    SeqMatch,
    /// Sequence mismatch (X)
    SeqMismatch,
    /// Padding (P)
    Pad,
}

impl fmt::Display for CigarOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CigarOp::Match => 'M',
                CigarOp::Insertion => 'I',
                CigarOp::Deletion => 'D',
                CigarOp::Skip => 'N',
                CigarOp::SoftClip => 'S',
                CigarOp::HardClip => 'H',
                CigarOp::SeqMatch => '=',
                CigarOp::SeqMismatch => 'X',
                CigarOp::Pad => 'P',
            }
        )
    }
}

/// Parsed CIGAR element
#[derive(Debug, Clone, Copy)]
pub struct CigarElement {
    pub op: CigarOp,
    pub len: u32,
}

impl fmt::Display for CigarElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.len, self.op)
    }
}

/// Full CIGAR string with parsed elements
#[derive(Debug, Clone)]
pub struct CigarString {
    /// Parsed CIGAR elements
    pub elements: Vec<CigarElement>,
}

impl CigarString {
    /// Create CIGAR string from operation sequence
    pub fn from_ops(ops: &[(CigarOp, u32)]) -> Self {
        let mut elements = Vec::new();

        // Merge consecutive operations
        if ops.is_empty() {
            return CigarString { elements };
        }

        let (mut current_op, mut current_len) = ops[0];

        for &(op, len) in &ops[1..] {
            if op == current_op {
                current_len += len;
            } else {
                elements.push(CigarElement {
                    op: current_op,
                    len: current_len,
                });
                current_op = op;
                current_len = len;
            }
        }

        elements.push(CigarElement {
            op: current_op,
            len: current_len,
        });

        CigarString { elements }
    }

    /// Parse SAM/BAM CIGAR string format
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut elements = Vec::new();
        let mut current_len = String::new();

        for ch in s.chars() {
            match ch {
                '0'..='9' => current_len.push(ch),
                'M' | 'I' | 'D' | 'N' | 'S' | 'H' | '=' | 'X' | 'P' => {
                    let len = current_len
                        .parse::<u32>()
                        .map_err(|_| format!("Invalid CIGAR length: {}", current_len))?;

                    let op = match ch {
                        'M' => CigarOp::Match,
                        'I' => CigarOp::Insertion,
                        'D' => CigarOp::Deletion,
                        'N' => CigarOp::Skip,
                        'S' => CigarOp::SoftClip,
                        'H' => CigarOp::HardClip,
                        '=' => CigarOp::SeqMatch,
                        'X' => CigarOp::SeqMismatch,
                        'P' => CigarOp::Pad,
                        _ => unreachable!(),
                    };

                    elements.push(CigarElement { op, len });
                    current_len.clear();
                }
                _ => return Err(format!("Invalid CIGAR character: {}", ch)),
            }
        }

        if !current_len.is_empty() {
            return Err("CIGAR string ended with incomplete operation".to_string());
        }

        Ok(CigarString { elements })
    }

    /// Convert to SAM format string
    pub fn to_string_sam(&self) -> String {
        self.elements
            .iter()
            .map(|elem| elem.to_string())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Calculate query consumed length
    pub fn query_len(&self) -> u32 {
        self.elements
            .iter()
            .map(|elem| match elem.op {
                CigarOp::Match
                | CigarOp::Insertion
                | CigarOp::SoftClip
                | CigarOp::SeqMatch
                | CigarOp::SeqMismatch => elem.len,
                _ => 0,
            })
            .sum()
    }

    /// Calculate reference consumed length
    pub fn reference_len(&self) -> u32 {
        self.elements
            .iter()
            .map(|elem| match elem.op {
                CigarOp::Match
                | CigarOp::Deletion
                | CigarOp::Skip
                | CigarOp::SeqMatch
                | CigarOp::SeqMismatch => elem.len,
                _ => 0,
            })
            .sum()
    }
}

impl fmt::Display for CigarString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string_sam())
    }
}

/// Backtracking matrix for DP alignment
#[derive(Debug, Clone, Copy)]
pub enum BacktrackOp {
    /// Diagonal (match/mismatch)
    Diagonal,
    /// Up (insertion)
    Up,
    /// Left (deletion)
    Left,
}

/// Trace back from DP table to generate CIGAR
pub fn traceback_to_cigar(
    backtrack_matrix: &[Vec<BacktrackOp>],
    query: &[u8],
    reference: &[u8],
    mut query_pos: usize,
    mut ref_pos: usize,
) -> CigarString {
    let mut ops = Vec::new();

    // Trace back from end to start
    while query_pos > 0 || ref_pos > 0 {
        let op = if query_pos > 0 && ref_pos > 0 {
            backtrack_matrix[query_pos][ref_pos]
        } else if query_pos > 0 {
            BacktrackOp::Up
        } else {
            BacktrackOp::Left
        };

        match op {
            BacktrackOp::Diagonal => {
                // Check if match or mismatch
                let cigar_op = if query[query_pos - 1] == reference[ref_pos - 1] {
                    CigarOp::SeqMatch
                } else {
                    CigarOp::SeqMismatch
                };

                ops.push((cigar_op, 1));
                query_pos -= 1;
                ref_pos -= 1;
            }
            BacktrackOp::Up => {
                // Insertion in query
                ops.push((CigarOp::Insertion, 1));
                query_pos -= 1;
            }
            BacktrackOp::Left => {
                // Deletion from query
                ops.push((CigarOp::Deletion, 1));
                ref_pos -= 1;
            }
        }
    }

    ops.reverse();
    CigarString::from_ops(&ops)
}

/// Generate CIGAR from HMM Viterbi path
pub fn cigar_from_hmm_path(path: &[u8], length: usize) -> CigarString {
    let mut ops = Vec::new();
    let mut current_op = None;
    let mut current_len = 0u32;

    for &state in path {
        let new_op = match state {
            0 => CigarOp::SeqMatch,  // Match state
            1 => CigarOp::Insertion, // Insert state
            2 => CigarOp::Deletion,  // Delete state
            _ => CigarOp::Match,
        };

        if let Some(op) = current_op {
            if op == new_op {
                current_len += 1;
            } else {
                ops.push((op, current_len));
                current_op = Some(new_op);
                current_len = 1;
            }
        } else {
            current_op = Some(new_op);
            current_len = 1;
        }
    }

    if let Some(op) = current_op {
        ops.push((op, current_len));
    }

    CigarString::from_ops(&ops)
}

/// Calculate query alignment position from CIGAR
pub fn query_position_from_cigar(cigar: &CigarString, bp: u32) -> Option<u32> {
    let mut pos = 0;

    for elem in &cigar.elements {
        let elem_len = elem.len;

        match elem.op {
            CigarOp::Match | CigarOp::Insertion | CigarOp::SoftClip | CigarOp::SeqMatch | CigarOp::SeqMismatch => {
                if pos + elem_len >= bp {
                    return Some(pos + (bp - pos).min(elem_len));
                }
                pos += elem_len;
            }
            _ => {}
        }
    }

    None
}

/// Calculate reference alignment position from CIGAR
pub fn reference_position_from_cigar(cigar: &CigarString, bp: u32) -> Option<u32> {
    let mut pos = 0;

    for elem in &cigar.elements {
        let elem_len = elem.len;

        match elem.op {
            CigarOp::Match | CigarOp::Deletion | CigarOp::Skip | CigarOp::SeqMatch | CigarOp::SeqMismatch => {
                if pos + elem_len >= bp {
                    return Some(pos + (bp - pos).min(elem_len));
                }
                pos += elem_len;
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cigar_parsing() {
        let cigar_str = "10M2I5D";
        let cigar = CigarString::parse(cigar_str).unwrap();
        
        assert_eq!(cigar.elements.len(), 3);
        assert_eq!(cigar.elements[0].len, 10);
        assert_eq!(cigar.elements[1].op, CigarOp::Insertion);
    }

    #[test]
    fn test_cigar_lengths() {
        let ops = vec![
            (CigarOp::Match, 10),
            (CigarOp::Insertion, 2),
            (CigarOp::Deletion, 5),
        ];
        let cigar = CigarString::from_ops(&ops);

        assert_eq!(cigar.query_len(), 12);
        assert_eq!(cigar.reference_len(), 15);
    }

    #[test]
    fn test_cigar_generation() {
        let ops = vec![(CigarOp::Match, 5), (CigarOp::Insertion, 2)];
        let cigar = CigarString::from_ops(&ops);
        
        assert_eq!(cigar.to_string_sam(), "5M2I");
    }

    #[test]
    fn test_hmm_cigar_generation() {
        let path = vec![0, 0, 0, 1, 1, 0, 0, 2];
        let cigar = cigar_from_hmm_path(&path, 8);
        
        assert!(!cigar.elements.is_empty());
    }
}
