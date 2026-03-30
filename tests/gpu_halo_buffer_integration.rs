//! GPU Halo Buffer Integration Tests
//!
//! Comprehensive test suite demonstrating halo-buffer tiled DP computation.
//! Tests cover boundary handling, halo propagation, and result accuracy.

use omicsx::alignment::{HaloBufferManager, HaloConfig, GpuTilingStrategy, TilingProfile};
use omicsx::scoring::{ScoringMatrix, MatrixType};

#[test]
fn test_halo_single_tile_no_propagation() {
    // Simple case: single 16x16 tile with halo
    let config = HaloConfig::default();
    let mut manager = HaloBufferManager::new(16, 16, config);

    // Initialize boundaries (first row/col = 0)
    manager.initialize_boundaries();

    let tile = manager.get_tile(0, 0);
    assert_eq!(tile.rows, 18);
    assert_eq!(tile.cols, 18);

    // Check boundary initialization
    let halo_idx = tile.halo_size - 1;
    for j in 0..18 {
        assert_eq!(tile.get(halo_idx, j), Some(0), "Top halo should be 0");
    }
    for i in 0..18 {
        assert_eq!(tile.get(i, halo_idx), Some(0), "Left halo should be 0");
    }
}

#[test]
fn test_halo_two_tile_vertical_propagation() {
    // Two tiles stacked vertically
    let config = HaloConfig::default();
    let mut manager = HaloBufferManager::new(32, 16, config);

    manager.initialize_boundaries();

    // Simulate filling bottom row of tile (0,0)
    let tile1 = manager.get_tile(0, 0);
    let core_width = config.tile_width;
    let halo = config.halo_size;

    // Fill bottom core row with test pattern: 10, 20, 30, ..., 160
    for j in 0..core_width {
        let row_idx = halo + config.tile_height - 1;
        let col_idx = halo + j;
        tile1.dp_values[row_idx * tile1.cols + col_idx] = ((j + 1) * 10) as i32;
    }

    // Propagate boundaries
    manager.propagate_boundaries(0, 0);

    // Check that tile (1,0) received the values in its top halo
    let tile2 = manager.get_tile(1, 0);
    let top_halo_row = halo - 1;

    for j in 0..core_width {
        let col_idx = halo + j;
        let received_value = tile2.dp_values[top_halo_row * tile2.cols + col_idx];
        let expected = ((j + 1) * 10) as i32;
        assert_eq!(
            received_value, expected,
            "Row {} Col {}: expected {}, got {}",
            top_halo_row, col_idx, expected, received_value
        );
    }
}

#[test]
fn test_halo_two_tile_horizontal_propagation() {
    // Two tiles side by side
    let config = HaloConfig::default();
    let mut manager = HaloBufferManager::new(16, 32, config);

    manager.initialize_boundaries();

    // Simulate filling right column of tile (0,0)
    let tile1 = manager.get_tile(0, 0);
    let core_height = config.tile_height;
    let halo = config.halo_size;

    // Fill right core column with test pattern: 5, 10, 15, ..., 80
    for i in 0..core_height {
        let row_idx = halo + i;
        let col_idx = halo + config.tile_width - 1;
        tile1.dp_values[row_idx * tile1.cols + col_idx] = ((i + 1) * 5) as i32;
    }

    // Propagate boundaries
    manager.propagate_boundaries(0, 0);

    // Check that tile (0,1) received the values in its left halo
    let tile2 = manager.get_tile(0, 1);
    let left_halo_col = halo - 1;

    for i in 0..core_height {
        let row_idx = halo + i;
        let received_value = tile2.dp_values[row_idx * tile2.cols + left_halo_col];
        let expected = ((i + 1) * 5) as i32;
        assert_eq!(
            received_value, expected,
            "Row {} Col {}: expected {}, got {}",
            row_idx, left_halo_col, expected, received_value
        );
    }
}

#[test]
fn test_halo_four_tile_grid() {
    // 2x2 grid of tiles (32x32 total)
    let config = HaloConfig::default();
    let mut manager = HaloBufferManager::new(32, 32, config);

    assert_eq!(manager.num_tile_rows(), 2);
    assert_eq!(manager.num_tile_cols(), 2);

    manager.initialize_boundaries();

    let halo = config.halo_size;
    let tile_size = config.tile_width;

    // Fill all four tiles with gradients
    for tr in 0..2 {
        for tc in 0..2 {
            let tile = manager.get_tile(tr, tc);

            // Set core values to pattern: value = (tr * 100 + tc * 10 + i * 1 + j * 0.1)
            for i in 0..tile_size {
                for j in 0..tile_size {
                    let row_idx = halo + i;
                    let col_idx = halo + j;
                    let value = (tr as i32 * 1000 + tc as i32 * 100 + i as i32);
                    tile.dp_values[row_idx * tile.cols + col_idx] = value;
                }
            }
        }
    }

    // Propagate boundaries sequentially (raster scan)
    for tr in 0..2 {
        for tc in 0..2 {
            manager.propagate_boundaries(tr, tc);
        }
    }

    // Verify interior boundaries are consistent
    // Bottom-right of tile (0,0) should equal top-left interior of tile (1,0)
    let bottom_row_0_0 = manager.get_tile(0, 0).get_bottom_core_row();
    
    let top_halo_1_0: Vec<i32> = (0..tile_size)
        .map(|j| {
            let col_idx = halo + j;
            let tile_1_0 = manager.get_tile(1, 0);
            tile_1_0.dp_values[(halo - 1) * tile_1_0.cols + col_idx]
        })
        .collect();

    for j in 0..tile_size {
        assert_eq!(
            bottom_row_0_0[j], top_halo_1_0[j],
            "Horizontal boundary mismatch at column {}",
            j
        );
    }
}

#[test]
fn test_halo_result_assembly() {
    // Assemble 4x4 result from 2x2 tile grid
    let config = HaloConfig::default();
    let mut manager = HaloBufferManager::new(32, 32, config);

    manager.initialize_boundaries();

    // Fill all tiles with simple pattern: value = i + j
    let halo = config.halo_size;
    let tile_size = config.tile_width;

    for tr in 0..2 {
        for tc in 0..2 {
            let tile = manager.get_tile(tr, tc);

            for i in 0..tile_size {
                for j in 0..tile_size {
                    let row_idx = halo + i;
                    let col_idx = halo + j;
                    let global_i = tr * tile_size + i;
                    let global_j = tc * tile_size + j;

                    let value = (global_i as i32 + global_j as i32) % 100;
                    tile.dp_values[row_idx * tile.cols + col_idx] = value;
                }
            }
        }
    }

    // Assemble result
    let result = manager.assemble_result();

    // Verify dimensions
    assert_eq!(result.len(), 33); // 32x32 + 1 for (0,0)
    assert_eq!(result[0].len(), 33);

    // Verify core values match pattern
    for i in 1..=32 {
        for j in 1..=32 {
            let expected = ((i - 1 + j - 1) as i32) % 100;
            assert_eq!(
                result[i][j], expected,
                "Mismatch at [{},{}]: expected {}, got {}",
                i, j, expected, result[i][j]
            );
        }
    }
}

#[test]
fn test_tiling_profile_memory_overhead() {
    // Verify memory overhead is reasonable
    // For 16x16 core with 1-cell halo (18x18 total):
    // Overhead = (18² - 16²) / 16² = (324 - 256) / 256 = 26.56%
    let config = HaloConfig::default();
    let core_size = config.tile_width * config.tile_height;
    let padded_size = (config.tile_width + 2 * config.halo_size)
        * (config.tile_height + 2 * config.halo_size);

    let overhead = (padded_size as f64 - core_size as f64) / core_size as f64;
    
    // Overhead should be below 50% for reasonable configurations  
    assert!(
        overhead < 0.50,
        "Memory overhead {:.2}% should be < 50%",
        overhead * 100.0
    );
}

#[test]
fn test_gpu_tiling_strategy_large_sequence() {
    let matrix = ScoringMatrix::new(MatrixType::Blosum62).unwrap();
    let profile = TilingProfile::a100();

    // 100k bp sequences 
    let strategy =
        GpuTilingStrategy::new(100_000, 100_000, matrix, -11, -1, profile).unwrap();

    let num_tiles = strategy.num_tiles();
    assert!(num_tiles > 0);

    // Verify memory requirement calculation
    let mem_bytes = strategy.gpu_memory_requirement();
    assert!(mem_bytes > 0, "Memory requirement should be positive");
}

#[test]
fn test_gpu_tiling_strategy_tile_grid() {
    let matrix = ScoringMatrix::new(MatrixType::Blosum62).unwrap();
    let profile = TilingProfile::conservative();

    // 64x64 sequences with 16x16 tiles = 4x4 grid
    let strategy = GpuTilingStrategy::new(64, 64, matrix, -11, -1, profile).unwrap();

    let tiles = strategy.tiles_in_order();
    assert_eq!(tiles.len(), 16); // 4x4 = 16 tiles

    // Verify raster scan order
    let expected_order = vec![
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ];

    for (i, &tile) in tiles.iter().enumerate() {
        assert_eq!(tile, expected_order[i], "Tile {} order mismatch", i);
    }
}

#[test]
fn test_halo_edge_case_single_cell() {
    // Edge case: 1x1 tile (just halo boundaries)
    let config = HaloConfig {
        tile_width: 1,
        tile_height: 1,
        halo_size: 1,
    };

    let mut manager = HaloBufferManager::new(1, 1, config.clone());
    manager.initialize_boundaries();

    let tile = manager.get_tile(0, 0);
    assert_eq!(tile.rows, 3); // 1 + 2
    assert_eq!(tile.cols, 3); // 1 + 2

    // Verify boundary initialization
    // Note: boundaries are initialized to 0 by initialize_boundaries()
    // Interior corners are initialized to i32::MIN/2 (uncomputed cells)
    assert_eq!(tile.get(0, 0), Some(0)); // corner set by initialization
    assert_eq!(tile.get(0, 1), Some(0)); // top edge
    assert_eq!(tile.get(1, 0), Some(0)); // left edge
}

#[test]
fn test_halo_custom_configuration() {
    // Test with custom sizes (4x4 tiles with 2-cell halo)
    let config = HaloConfig {
        tile_width: 4,
        tile_height: 4,
        halo_size: 2,
    };

    let mut manager = HaloBufferManager::new(8, 8, config.clone());
    manager.initialize_boundaries();

    let (w, h) = config.padded_dimensions();
    assert_eq!(w, 8); // 4 + 2*2
    assert_eq!(h, 8);

    let tile = manager.get_tile(0, 0);
    assert_eq!(tile.rows, 8);
    assert_eq!(tile.cols, 8);

    // Core region is 4x4
    let core = tile.get_core();
    assert_eq!(core.len(), 16);
}

#[test]
fn test_halo_boundary_diagonal_propagation() {
    // Test diagonal neighbor interaction (right-bottom corner)
    let config = HaloConfig::default();
    let mut manager = HaloBufferManager::new(32, 32, config.clone());

    manager.initialize_boundaries();

    let halo = config.halo_size;
    let tile_size = config.tile_width;

    // Tile (0,0): set corner value at bottom-right
    let tile_0_0 = manager.get_tile(0, 0);
    let corner_row = halo + tile_size - 1;
    let corner_col = halo + tile_size - 1;
    tile_0_0.dp_values[corner_row * tile_0_0.cols + corner_col] = 999;

    manager.propagate_boundaries(0, 0);

    // Tile (1,1) should NOT receive this corner value (no diagonal propagation)
    let tile_1_1 = manager.get_tile(1, 1);
    let top_left_core = tile_1_1.dp_values[(halo) * tile_1_1.cols + (halo)];
    assert_ne!(
        top_left_core, 999,
        "Diagonal propagation should not occur"
    );
}

#[test]
fn test_tiling_profile_selection() {
    // Test profile selection for different GPU models
    let v100 = TilingProfile::v100();
    let a100 = TilingProfile::a100();
    let rtx = TilingProfile::rtx3090();
    let conservative = TilingProfile::conservative();

    // A100 should be most aggressive
    assert!(a100.batch_size >= v100.batch_size);
    assert!(a100.batch_size >= rtx.batch_size);
    assert!(a100.batch_size >= conservative.batch_size);

    // Conservative should work everywhere
    assert!(conservative.tile_size <= 16);
    assert!(conservative.shared_memory_limit <= 48);
}
