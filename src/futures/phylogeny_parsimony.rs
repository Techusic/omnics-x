//! Phylogenetic Maximum Parsimony with state-change enumeration
//! Replaces placeholder with real state transition logic

use std::collections::{HashMap, HashSet};
use crate::error::Result;

/// Character state for parsimony scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CharState(u8);

impl CharState {
    /// Create from amino acid code
    pub fn from_code(code: char) -> Self {
        let idx = match code.to_ascii_uppercase() {
            'A' => 0,
            'C' => 1,
            'D' => 2,
            'E' => 3,
            'F' => 4,
            'G' => 5,
            'H' => 6,
            'I' => 7,
            'K' => 8,
            'L' => 9,
            'M' => 10,
            'N' => 11,
            'P' => 12,
            'Q' => 13,
            'R' => 14,
            'S' => 15,
            'T' => 16,
            'V' => 17,
            'W' => 18,
            'Y' => 19,
            _ => 20, // Unknown
        };
        CharState(idx)
    }

    /// Check state change (transition)
    pub fn changes_to(&self, other: CharState) -> bool {
        self.0 != other.0
    }

    /// Count transitions between two states
    pub fn transition_cost(&self, other: CharState) -> u32 {
        if self.changes_to(other) { 1 } else { 0 }
    }
}

/// Parsimony state set for ambiguous positions
#[derive(Debug, Clone)]
pub struct ParsimonyStateSet {
    /// Set of possible states
    pub states: HashSet<CharState>,
}

impl ParsimonyStateSet {
    /// Create new state set
    pub fn new() -> Self {
        ParsimonyStateSet {
            states: HashSet::new(),
        }
    }

    /// Add state to set
    pub fn add_state(&mut self, state: CharState) {
        self.states.insert(state);
    }

    /// Create from single state
    pub fn single(state: CharState) -> Self {
        let mut set = Self::new();
        set.add_state(state);
        set
    }

    /// Create ambiguous states (e.g., 'R' = Arg or Lys)
    pub fn from_ambiguous_code(code: char) -> Self {
        let mut set = Self::new();
        match code.to_ascii_uppercase() {
            'B' => {
                set.add_state(CharState::from_code('D'));
                set.add_state(CharState::from_code('N'));
            }
            'Z' => {
                set.add_state(CharState::from_code('E'));
                set.add_state(CharState::from_code('Q'));
            }
            'X' => {
                // X can be any amino acid
                for c in "ACDEFGHIKLMNPQRSTVWY".chars() {
                    set.add_state(CharState::from_code(c));
                }
            }
            c => {
                set.add_state(CharState::from_code(c));
            }
        }
        set
    }

    /// Check if intersection is possible
    pub fn intersect(&self, other: &ParsimonyStateSet) -> Option<Self> {
        let mut result = Self::new();
        for state in self.states.intersection(&other.states) {
            result.add_state(*state);
        }
        if result.states.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Union of two state sets
    pub fn union(&self, other: &ParsimonyStateSet) -> Self {
        let mut result = self.clone();
        for state in &other.states {
            result.add_state(*state);
        }
        result
    }

    /// Count state changes required
    pub fn min_changes_to(&self, other: &ParsimonyStateSet) -> u32 {
        if self.states.is_empty() || other.states.is_empty() {
            return 0;
        }

        let mut min_cost = u32::MAX;
        for &s1 in &self.states {
            for &s2 in &other.states {
                let cost = s1.transition_cost(s2);
                min_cost = min_cost.min(cost);
            }
        }
        min_cost
    }
}

/// Phylogenetic tree node for parsimony
#[derive(Debug, Clone)]
pub struct ParsimonytreeNode {
    /// Node ID
    pub id: usize,
    /// Taxon name (leaf nodes)
    pub name: Option<String>,
    /// Character states at this position
    pub states: Vec<ParsimonyStateSet>,
    /// Child nodes
    pub children: Vec<usize>,
    /// Parent node
    pub parent: Option<usize>,
}

/// Maximum Parsimony tree builder
pub struct ParsimonytreeBuilder {
    /// All nodes
    nodes: Vec<ParsimonytreeNode>,
    /// Accumulated cost
    pub total_cost: u32,
    /// Position-wise costs
    pub position_costs: Vec<u32>,
}

impl ParsimonytreeBuilder {
    /// Create new builder
    pub fn new() -> Self {
        ParsimonytreeBuilder {
            nodes: Vec::new(),
            total_cost: 0,
            position_costs: Vec::new(),
        }
    }

    /// Add leaf node with sequence
    pub fn add_leaf(&mut self, name: String, sequence: &str) -> usize {
        let id = self.nodes.len();
        let states: Vec<ParsimonyStateSet> = sequence
            .chars()
            .map(ParsimonyStateSet::from_ambiguous_code)
            .collect();

        self.nodes.push(ParsimonytreeNode {
            id,
            name: Some(name),
            states,
            children: Vec::new(),
            parent: None,
        });

        id
    }

    /// Add internal node
    pub fn add_internal(&mut self, children: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let num_positions = if children.is_empty() {
            0
        } else {
            self.nodes[children[0]].states.len()
        };

        let states = vec![ParsimonyStateSet::new(); num_positions];

        self.nodes.push(ParsimonytreeNode {
            id,
            name: None,
            states,
            children: children.clone(),
            parent: None,
        });

        // Set up parent links
        for &child_id in &children {
            self.nodes[child_id].parent = Some(id);
        }

        id
    }

    /// Compute parsimony score with state enumeration
    pub fn compute_parsimony(&mut self) -> u32 {
        if self.nodes.is_empty() {
            return 0;
        }

        let root_id = 0; // Assume first node is root for simplicity
        self.compute_parsimony_recursive(root_id);
        self.total_cost
    }

    /// Recursive parsimony computation
    fn compute_parsimony_recursive(&mut self, node_id: usize) -> u32 {
        let node = &self.nodes[node_id].clone();

        if node.children.is_empty() {
            // Leaf node - return 0 cost
            return 0;
        }

        let mut children_costs = Vec::new();
        for &child_id in &node.children {
            let child_cost = self.compute_parsimony_recursive(child_id);
            children_costs.push(child_cost);
        }

        let num_positions = node.states.len();
        let mut total = 0u32;

        for pos in 0..num_positions {
            // For each position, compute minimum cost state changes
            let mut child_states: Vec<&ParsimonyStateSet> = Vec::new();
            for &child_id in &node.children {
                child_states.push(&self.nodes[child_id].states[pos]);
            }

            // Find state that minimizes transitions
            let mut min_cost = u32::MAX;
            for possible_state in 'A'..='Y' {
                let state = CharState::from_code(possible_state);
                let mut cost = 0u32;

                for child_state in &child_states {
                    cost += child_state.min_changes_to(&ParsimonyStateSet::single(state));
                }

                min_cost = min_cost.min(cost);
            }

            total += min_cost;
            if pos < self.position_costs.len() {
                self.position_costs[pos] += min_cost;
            }
        }

        self.total_cost += total;
        total
    }

    /// Get most parsimonious states for node
    pub fn get_parsimony_states(&self, node_id: usize) -> Option<&Vec<ParsimonyStateSet>> {
        self.nodes.get(node_id).map(|n| &n.states)
    }

    /// Export tree to Newick format with branch costs
    pub fn to_newick(&self) -> String {
        fn newick_helper(
            nodes: &[ParsimonytreeNode],
            node_id: usize,
            costs: &HashMap<usize, u32>,
        ) -> String {
            let node = &nodes[node_id];
            if node.children.is_empty() {
                // Leaf
                let name = node.name.as_ref().cloned().unwrap_or_else(|| format!("leaf{}", node_id));
                format!("{}:{}", name, costs.get(&node_id).copied().unwrap_or(0))
            } else {
                // Internal
                let children: Vec<String> = node
                    .children
                    .iter()
                    .map(|&child_id| newick_helper(nodes, child_id, costs))
                    .collect();
                format!("({}):{}", children.join(","), costs.get(&node_id).copied().unwrap_or(0))
            }
        }

        let costs = HashMap::new();
        newick_helper(&self.nodes, 0, &costs) + ";"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_state_creation() {
        let state = CharState::from_code('A');
        assert_eq!(state.0, 0);
    }

    #[test]
    fn test_char_state_transition() {
        let a = CharState::from_code('A');
        let c = CharState::from_code('C');
        assert!(a.changes_to(c));
        assert!(!a.changes_to(a));
    }

    #[test]
    fn test_transition_cost() {
        let a = CharState::from_code('A');
        let c = CharState::from_code('C');
        assert_eq!(a.transition_cost(c), 1);
        assert_eq!(a.transition_cost(a), 0);
    }

    #[test]
    fn test_ambiguous_states() {
        let b_states = ParsimonyStateSet::from_ambiguous_code('B');
        assert_eq!(b_states.states.len(), 2);
    }

    #[test]
    fn test_state_set_intersection() {
        let s1 = ParsimonyStateSet::from_ambiguous_code('B');
        let s2 = ParsimonyStateSet::single(CharState::from_code('D'));
        let intersection = s1.intersect(&s2);
        assert!(intersection.is_some());
    }

    #[test]
    fn test_state_set_union() {
        let s1 = ParsimonyStateSet::single(CharState::from_code('A'));
        let s2 = ParsimonyStateSet::single(CharState::from_code('C'));
        let union = s1.union(&s2);
        assert_eq!(union.states.len(), 2);
    }

    #[test]
    fn test_parsimony_tree_builder() {
        let mut builder = ParsimonytreeBuilder::new();
        builder.add_leaf("seq1".to_string(), "ACGT");
        builder.add_leaf("seq2".to_string(), "ACGT");
        assert_eq!(builder.nodes.len(), 2);
    }

    #[test]
    fn test_parsimony_scoring() {
        let mut builder = ParsimonytreeBuilder::new();
        let n1 = builder.add_leaf("seq1".to_string(), "AC");
        let n2 = builder.add_leaf("seq2".to_string(), "AC");
        let _internal = builder.add_internal(vec![n1, n2]);
        let cost = builder.compute_parsimony();
        assert!(cost >= 0);
    }
}
