/// Actual CUDA kernel implementations compiled and executed at runtime
/// Uses cudarc + NVRTC for JIT compilation

pub const SMITH_WATERMAN_KERNEL: &str = r#"
extern "C" __global__ void smith_waterman_kernel(
    const unsigned char* seq1,      // Query sequence
    const unsigned char* seq2,      // Subject sequence
    const int* scoring_matrix,      // 24x24 scoring matrix (row-major)
    int* dp_matrix,                 // Output DP table
    int* traceback,                 // Output traceback
    int len1,                       // Query length
    int len2,                       // Subject length
    int gap_open,                   // Gap open penalty
    int gap_extend,                 // Gap extend penalty
    int* max_score,                 // Output: max score found
    int* max_i,                     // Output: row of max
    int* max_j                      // Output: col of max
) {
    // Grid-stride loop pattern for 2D DP
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= len1 || j >= len2) return;
    
    // Shared memory for scoring matrix (optimized access)
    __shared__ int smatrix[24 * 24];
    if (threadIdx.x < 24 && threadIdx.y < 24) {
        smatrix[threadIdx.y * 24 + threadIdx.x] = 
            scoring_matrix[threadIdx.y * 24 + threadIdx.x];
    }
    __syncthreads();
    
    int idx = i * len2 + j;
    int score = 0;
    int trace = 0;  // 0=stop, 1=diag, 2=up, 3=left
    
    if (i == 0 && j == 0) {
        dp_matrix[idx] = 0;
        traceback[idx] = 0;
        return;
    }
    
    // Get amino acids (0-19 standard IUPAC)
    int aa1 = seq1[i - 1];
    int aa2 = seq2[j - 1];
    
    // Match/mismatch score
    int match_score = (i > 0 && j > 0) 
        ? dp_matrix[(i-1) * len2 + (j-1)] + smatrix[aa1 * 24 + aa2]
        : smatrix[aa1 * 24 + aa2];
    
    // Deletion score (gap in seq2)
    int del_score = (i > 0)
        ? dp_matrix[(i-1) * len2 + j] + gap_extend
        : 0;
    
    // Insertion score (gap in seq1)
    int ins_score = (j > 0)
        ? dp_matrix[i * len2 + (j-1)] + gap_extend
        : 0;
    
    // Local alignment: can start fresh from 0
    score = max({0, match_score, del_score, ins_score});
    
    if (score == 0) trace = 0;
    else if (score == match_score) trace = 1;
    else if (score == del_score) trace = 2;
    else trace = 3;
    
    dp_matrix[idx] = score;
    traceback[idx] = trace;
    
    // Update global maximum with atomic operation
    if (score > 0) {
        atomicMax(max_score, score);
        if (score == *max_score) {
            atomicExch(max_i, i);
            atomicExch(max_j, j);
        }
    }
}
"#;

pub const NEEDLEMAN_WUNSCH_KERNEL: &str = r#"
extern "C" __global__ void needleman_wunsch_kernel(
    const unsigned char* seq1,
    const unsigned char* seq2,
    const int* scoring_matrix,
    int* dp_matrix,
    int* traceback,
    int len1,
    int len2,
    int gap_open,
    int gap_extend
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > len1 || j > len2) return;
    
    // Shared memory for scoring matrix
    __shared__ int smatrix[24 * 24];
    if (threadIdx.x < 24 && threadIdx.y < 24) {
        smatrix[threadIdx.y * 24 + threadIdx.x] = 
            scoring_matrix[threadIdx.y * 24 + threadIdx.x];
    }
    __syncthreads();
    
    int idx = i * (len2 + 1) + j;
    int trace = 0;
    
    // Initialize boundaries
    if (i == 0) {
        dp_matrix[idx] = j * gap_extend;
        traceback[idx] = 3;  // left
        return;
    }
    if (j == 0) {
        dp_matrix[idx] = i * gap_extend;
        traceback[idx] = 2;  // up
        return;
    }
    
    // Get amino acids
    int aa1 = seq1[i - 1];
    int aa2 = seq2[j - 1];
    
    // Compute DP recurrence
    int match = dp_matrix[(i-1) * (len2 + 1) + (j-1)] + 
                smatrix[aa1 * 24 + aa2];
    int del = dp_matrix[(i-1) * (len2 + 1) + j] + gap_extend;
    int ins = dp_matrix[i * (len2 + 1) + (j-1)] + gap_extend;
    
    int score = max({match, del, ins});
    
    if (score == match) trace = 1;
    else if (score == del) trace = 2;
    else trace = 3;
    
    dp_matrix[idx] = score;
    traceback[idx] = trace;
}
"#;

pub const VITERBI_HMM_KERNEL: &str = r#"
extern "C" __global__ void viterbi_forward_kernel(
    const unsigned char* sequence,  // Encoded amino acids
    float* dp_m,                    // Match state DP table (output)
    float* dp_i,                    // Insert state DP table
    float* dp_d,                    // Delete state DP table
    const float* transitions,       // m×3 transition matrix
    const float* emissions,         // 20×m emission matrix
    int n,                          // Sequence length
    int m                           // HMM length
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int state = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos >= n || state >= m) return;
    
    int aa = sequence[pos];
    
    // Shared memory for efficiency
    __shared__ float trans[512];     // Max 128 states × 3
    __shared__ float emis[512];
    
    if (threadIdx.x + threadIdx.y * blockDim.x < m * 3) {
        trans[threadIdx.x + threadIdx.y * blockDim.x] = 
            transitions[(threadIdx.x + threadIdx.y * blockDim.x)];
    }
    if (threadIdx.x + threadIdx.y * blockDim.x < 20 * m) {
        emis[threadIdx.x + threadIdx.y * blockDim.x] = 
            emissions[(threadIdx.x + threadIdx.y * blockDim.x)];
    }
    __syncthreads();
    
    float emit_score = emis[aa * m + state];
    
    if (pos == 0) {
        dp_m[state] = emit_score;
        dp_i[state] = -1e6f;
        dp_d[state] = -1e6f;
        return;
    }
    
    // Previous states
    float prev_m = dp_m[(pos-1) * m + state];
    float prev_i = dp_i[(pos-1) * m + state];
    float prev_d = dp_d[(pos-1) * m + state];
    
    // Viterbi recurrence
    float best_m = max({prev_m, prev_i, prev_d}) + 
                   trans[state * 3 + 0] + emit_score;
    float best_i = max({prev_m, prev_i, prev_d}) + 
                   trans[state * 3 + 1] + emit_score;
    float best_d = max({prev_m, prev_i, prev_d}) + 
                   trans[state * 3 + 2];
    
    dp_m[pos * m + state] = best_m;
    dp_i[pos * m + state] = best_i;
    dp_d[pos * m + state] = best_d;
}
"#;
