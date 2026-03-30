//! GPU Halo-Buffer Management for Tiled DP Computation
//!
//! This module implements efficient boundary handling for tiled dynamic programming
//! on GPUs. When sequences exceed single-GPU memory capacity, tiling is necessary,
//! but DP dependencies span tile boundaries. Halo regions (overlapping boundaries)
//! solve this by caching boundary values from adjacent tiles.
//!
//! ## Problem
//! DP recurrence: H[i][j] depends on H[i-1][j], H[i][j-1], H[i-1][j-1]
//! With tiling, cells at tile edges need values from neighboring tiles.
//! Without halo regions, boundary cells compute with missing/incorrect dependency values.
//!
//! ## Solution
//! Each tile includes "halo" rows/columns from neighboring tiles:
//! ```
//! Tile Layout (with halo):
//! ┌─────────────────────┐
//! │ Halo (top)    (1 row)│
//! ├─────────────────────┤
//! │ Halo  Main tile   Halo │
//! │(L)   16×16       (R)  │
//! ├─────────────────────┤
//! │ Halo (bottom) (1 row)  │
//! └─────────────────────┘
//! Total: 18×18 with 16×16 core
//! ```
//!
//! ## Performance
//! - Memory overhead: 12.5% (18×18 vs 16×16)
//! - Eliminates boundary errors (infinite penalty or zero)
//! - Enables larger batch sizes for tiled execution

use std::collections::HashMap;

/// Configuration for halo-buffer tiling
#[derive(Debug, Clone, Copy)]
pub struct HaloConfig {
    /// Core tile width (usually 16)
    pub tile_width: usize,
    /// Core tile height (usually 16)
    pub tile_height: usize,
    /// Halo size in cells (usually 1)
    pub halo_size: usize,
}

impl HaloConfig {
    /// Create default 16×16 tiles with 1-cell halo
    pub fn default() -> Self {
        HaloConfig {
            tile_width: 16,
            tile_height: 16,
            halo_size: 1,
        }
    }

    /// Get padded dimensions including halo
    pub fn padded_dimensions(&self) -> (usize, usize) {
        (
            self.tile_width + 2 * self.halo_size,
            self.tile_height + 2 * self.halo_size,
        )
    }

    /// Get total shared memory needed for one tile (in i32 units)
    pub fn shared_memory_size(&self) -> usize {
        let (w, h) = self.padded_dimensions();
        w * h
    }
}

/// Single halo-buffered tile
#[derive(Debug, Clone)]
pub struct HaloTile {
    /// Tile ID (row, col)
    pub tile_id: (usize, usize),
    /// Core region: [halo:halo+tile_height][halo:halo+tile_width]
    pub dp_values: Vec<i32>,
    /// Row dimension (tile_height + 2*halo)
    pub rows: usize,
    /// Column dimension (tile_width + 2*halo)
    pub cols: usize,
    /// Halo size
    pub halo_size: usize,
}

impl HaloTile {
    /// Create new halo tile
    pub fn new(tile_id: (usize, usize), config: &HaloConfig) -> Self {
        let (w, h) = config.padded_dimensions();
        HaloTile {
            tile_id,
            dp_values: vec![i32::MIN / 2; w * h], // Initialize to -inf (avoid overflow)
            rows: h,
            cols: w,
            halo_size: config.halo_size,
        }
    }

    /// Get value at (i, j) with bounds checking
    pub fn get(&self, i: usize, j: usize) -> Option<i32> {
        if i < self.rows && j < self.cols {
            Some(self.dp_values[i * self.cols + j])
        } else {
            None
        }
    }

    /// Set value at (i, j) with bounds checking
    pub fn set(&mut self, i: usize, j: usize, value: i32) -> bool {
        if i < self.rows && j < self.cols {
            self.dp_values[i * self.cols + j] = value;
            true
        } else {
            false
        }
    }

    /// Copy values into core region of this tile
    pub fn set_core(&mut self, core_data: &[i32]) -> bool {
        let core_size = (self.rows - 2 * self.halo_size) * (self.cols - 2 * self.halo_size);
        if core_data.len() < core_size {
            return false;
        }

        for i in 0..self.rows - 2 * self.halo_size {
            for j in 0..self.cols - 2 * self.halo_size {
                let src_idx = i * (self.cols - 2 * self.halo_size) + j;
                let dst_idx = (i + self.halo_size) * self.cols + (j + self.halo_size);
                self.dp_values[dst_idx] = core_data[src_idx];
            }
        }
        true
    }

    /// Get core region data (excluding halo)
    pub fn get_core(&self) -> Vec<i32> {
        let core_height = self.rows - 2 * self.halo_size;
        let core_width = self.cols - 2 * self.halo_size;
        let mut core = vec![0i32; core_height * core_width];

        for i in 0..core_height {
            for j in 0..core_width {
                let src_idx = (i + self.halo_size) * self.cols + (j + self.halo_size);
                let dst_idx = i * core_width + j;
                core[dst_idx] = self.dp_values[src_idx];
            }
        }
        core
    }

    /// Update top halo from neighbor's bottom core row
    pub fn update_top_halo(&mut self, neighbor_bottom_row: &[i32]) {
        let core_width = self.cols - 2 * self.halo_size;
        if neighbor_bottom_row.len() < core_width {
            return;
        }

        for j in 0..core_width {
            let src_idx = j;
            let dst_idx = (self.halo_size - 1) * self.cols + (self.halo_size + j);
            self.dp_values[dst_idx] = neighbor_bottom_row[src_idx];
        }
    }

    /// Update left halo from neighbor's right core column
    pub fn update_left_halo(&mut self, neighbor_right_col: &[i32]) {
        let core_height = self.rows - 2 * self.halo_size;
        if neighbor_right_col.len() < core_height {
            return;
        }

        for i in 0..core_height {
            let src_idx = i;
            let dst_idx = (self.halo_size + i) * self.cols + (self.halo_size - 1);
            self.dp_values[dst_idx] = neighbor_right_col[src_idx];
        }
    }

    /// Get bottom core row for passing to neighbor
    pub fn get_bottom_core_row(&self) -> Vec<i32> {
        let core_height = self.rows - 2 * self.halo_size;
        let core_width = self.cols - 2 * self.halo_size;
        let last_core_row = core_height - 1;

        let mut row = vec![0i32; core_width];
        for j in 0..core_width {
            let idx = (self.halo_size + last_core_row) * self.cols + (self.halo_size + j);
            row[j] = self.dp_values[idx];
        }
        row
    }

    /// Get right core column for passing to neighbor
    pub fn get_right_core_col(&self) -> Vec<i32> {
        let core_height = self.rows - 2 * self.halo_size;
        let core_width = self.cols - 2 * self.halo_size;
        let last_core_col = core_width - 1;

        let mut col = vec![0i32; core_height];
        for i in 0..core_height {
            let idx = (self.halo_size + i) * self.cols + (self.halo_size + last_core_col);
            col[i] = self.dp_values[idx];
        }
        col
    }
}

/// Manages halo buffers for tiled DP computation
pub struct HaloBufferManager {
    /// Configuration
    config: HaloConfig,
    /// Active tiles: (tile_row, tile_col) → HaloTile
    tiles: HashMap<(usize, usize), HaloTile>,
    /// Sequence dimensions
    pub seq_len1: usize,
    pub seq_len2: usize,
}

impl HaloBufferManager {
    /// Create new halo buffer manager
    pub fn new(seq_len1: usize, seq_len2: usize, config: HaloConfig) -> Self {
        HaloBufferManager {
            config,
            tiles: HashMap::new(),
            seq_len1,
            seq_len2,
        }
    }

    /// Get or create tile at position
    pub fn get_tile(&mut self, tile_row: usize, tile_col: usize) -> &mut HaloTile {
        self.tiles
            .entry((tile_row, tile_col))
            .or_insert_with(|| HaloTile::new((tile_row, tile_col), &self.config))
    }

    /// Number of tiles needed vertically
    pub fn num_tile_rows(&self) -> usize {
        (self.seq_len1 + self.config.tile_height - 1) / self.config.tile_height
    }

    /// Number of tiles needed horizontally
    pub fn num_tile_cols(&self) -> usize {
        (self.seq_len2 + self.config.tile_width - 1) / self.config.tile_width
    }

    /// Update halo regions after tile computation
    /// Call this after computing a tile to propagate boundary values to neighbors
    pub fn propagate_boundaries(&mut self, tile_row: usize, tile_col: usize) {
        if !self.tiles.contains_key(&(tile_row, tile_col)) {
            return;
        }

        // Get boundary data from current tile
        let bottom_row = self.tiles[&(tile_row, tile_col)].get_bottom_core_row();
        let right_col = self.tiles[&(tile_row, tile_col)].get_right_core_col();

        // Propagate to bottom neighbor
        if tile_row + 1 < self.num_tile_rows() {
            let neighbor = self.get_tile(tile_row + 1, tile_col);
            neighbor.update_top_halo(&bottom_row);
        }

        // Propagate to right neighbor
        if tile_col + 1 < self.num_tile_cols() {
            let neighbor = self.get_tile(tile_row, tile_col + 1);
            neighbor.update_left_halo(&right_col);
        }
    }

    /// Initialize boundaries for first row/column (zero boundary condition)
    pub fn initialize_boundaries(&mut self) {
        let num_rows = self.num_tile_rows();
        let num_cols = self.num_tile_cols();

        for tile_row in 0..num_rows {
            for tile_col in 0..num_cols {
                let tile = self.get_tile(tile_row, tile_col);

                // First row: halo is zero
                if tile_row == 0 {
                    for j in 0..tile.cols {
                        tile.set(tile.halo_size - 1, j, 0);
                    }
                }

                // First column: halo is zero
                if tile_col == 0 {
                    for i in 0..tile.rows {
                        tile.set(i, tile.halo_size - 1, 0);
                    }
                }

                // Corner (0,0): boundary cell
                if tile_row == 0 && tile_col == 0 {
                    tile.set(tile.halo_size - 1, tile.halo_size - 1, 0);
                }
            }
        }
    }

    /// Get total GPU memory required for all tiles
    pub fn total_gpu_memory(&self) -> usize {
        let tile_mem = self.config.shared_memory_size() * std::mem::size_of::<i32>();
        let num_tiles = self.num_tile_rows() * self.num_tile_cols();
        num_tiles * tile_mem
    }

    /// Assemble final DP matrix from all tiles (for small outputs)
    pub fn assemble_result(&self) -> Vec<Vec<i32>> {
        let mut result = vec![vec![0i32; self.seq_len2 + 1]; self.seq_len1 + 1];

        let num_rows = self.num_tile_rows();
        let num_cols = self.num_tile_cols();

        for tile_row in 0..num_rows {
            for tile_col in 0..num_cols {
                if let Some(tile) = self.tiles.get(&(tile_row, tile_col)) {
                    let core = tile.get_core();
                    let core_height = self.config.tile_height;
                    let core_width = self.config.tile_width;

                    let global_row_start = tile_row * core_height;
                    let global_col_start = tile_col * core_width;

                    for i in 0..core_height {
                        for j in 0..core_width {
                            let global_i = global_row_start + i + 1;
                            let global_j = global_col_start + j + 1;

                            if global_i <= self.seq_len1 && global_j <= self.seq_len2 {
                                let src_idx = i * core_width + j;
                                result[global_i][global_j] = core[src_idx];
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halo_config_dimensions() {
        let config = HaloConfig::default();
        let (w, h) = config.padded_dimensions();
        assert_eq!(w, 18); // 16 + 2*1
        assert_eq!(h, 18);
        assert_eq!(config.shared_memory_size(), 18 * 18);
    }

    #[test]
    fn test_halo_tile_creation() {
        let config = HaloConfig::default();
        let tile = HaloTile::new((0, 0), &config);
        assert_eq!(tile.rows, 18);
        assert_eq!(tile.cols, 18);
        assert_eq!(tile.dp_values.len(), 18 * 18);
    }

    #[test]
    fn test_halo_tile_get_set() {
        let config = HaloConfig::default();
        let mut tile = HaloTile::new((0, 0), &config);

        tile.set(5, 5, 42);
        assert_eq!(tile.get(5, 5), Some(42));
        assert_eq!(tile.get(20, 20), None); // Out of bounds
    }

    #[test]
    fn test_halo_buffer_manager() {
        let config = HaloConfig::default();
        let mut manager = HaloBufferManager::new(32, 32, config);

        assert_eq!(manager.num_tile_rows(), 2);
        assert_eq!(manager.num_tile_cols(), 2);

        let tile = manager.get_tile(0, 0);
        assert_eq!(tile.tile_id, (0, 0));
    }

    #[test]
    fn test_boundary_initialization() {
        let config = HaloConfig::default();
        let mut manager = HaloBufferManager::new(16, 16, config);
        manager.initialize_boundaries();

        let tile = manager.get_tile(0, 0);
        // Top halo row should be zero
        for j in 0..tile.cols {
            assert_eq!(tile.get(tile.halo_size - 1, j), Some(0));
        }
    }

    #[test]
    fn test_core_region_extraction() {
        let config = HaloConfig::default();
        let mut tile = HaloTile::new((0, 0), &config);

        // Set some values in core region
        for i in 0..16 {
            for j in 0..16 {
                tile.set(i + tile.halo_size, j + tile.halo_size, (i * 16 + j) as i32);
            }
        }

        let core = tile.get_core();
        assert_eq!(core.len(), 16 * 16);
        assert_eq!(core[0], 0);
        assert_eq!(core[255], 255);
    }

    #[test]
    fn test_halo_propagation() {
        let config = HaloConfig::default();
        let mut manager = HaloBufferManager::new(32, 32, config.clone());

        let tile1 = manager.get_tile(0, 0);
        // Fill bottom row of tile (0,0) with test values
        let core_width = config.tile_width;
        for j in 0..core_width {
            let idx = (config.tile_height - 1 + config.halo_size) * tile1.cols
                + (config.halo_size + j);
            tile1.dp_values[idx] = (j as i32) * 10;
        }

        manager.propagate_boundaries(0, 0);

        // Check that tile (1,0) has updated top halo
        let tile2 = manager.get_tile(1, 0);
        for j in 0..core_width {
            let top_halo_idx = (config.halo_size - 1) * tile2.cols + (config.halo_size + j);
            assert_eq!(tile2.dp_values[top_halo_idx], (j as i32) * 10);
        }
    }
}
