/* patched_ass2.c  -- Modified for fully distributed memory with virtual padding
   Key changes:
   - NO padding matrices allocated - uses virtual padding accessor
   - Matrix generation parallelized and KEPT distributed across MPI ranks
   - NO gathering of input matrix - each rank works with its portion directly
   - Padding calculated inside convolution function
   - Maximum memory efficiency across all processes
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdarg.h>
#include "mpi.h"
#include <omp.h>
#include <unistd.h>
#include <time.h>

int print = 0;
int record = 0;

void safe_abort(MPI_Comm comm, int code) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "[Rank %d] Aborting...\n", rank);
    MPI_Abort(comm, code);
}

float **alloc_matrix_contiguous(int rows, int cols) {
    float *data = calloc((size_t)rows * cols, sizeof(float));
    if (!data) return NULL;
    float **arr = malloc(rows * sizeof(float*));
    if (!arr) { free(data); return NULL; }
    for (int i = 0; i < rows; ++i)
        arr[i] = data + (size_t)i * cols;
    return arr;
}

void printif(const char *format, ...) {
    if (print) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

float **allocate_2d(int h, int w) {
    if (h <= 0 || w <= 0) return NULL;
    float *data = calloc((size_t)h * w, sizeof(float));
    if (!data) return NULL;
    float **mat = malloc((size_t)h * sizeof(float *));
    if (!mat) { free(data); return NULL; }
    for (int i = 0; i < h; ++i) {
        mat[i] = data + (size_t)i * w;
    }
    return mat;
}

void free_2d(float **mat) {
    if (!mat) return;
    free(mat[0]);
    free(mat);
}

bool is_row_zero(float **mat, int row, int w, float epsilon) {
    for (int j = 0; j < w; ++j)
        if (fabs(mat[row][j]) > epsilon)
            return false;
    return true;
}

bool is_col_zero(float **mat, int h, int col, float epsilon) {
    for (int i = 0; i < h; ++i)
        if (fabs(mat[i][col]) > epsilon)
            return false;
    return true;
}

float **remove_zero_padding(float **input, int in_h, int in_w, int *out_h, int *out_w) {
    const float epsilon = 1e-6f;
    int top = 0, bottom = in_h - 1;
    int left = 0, right = in_w - 1;

    while (top <= bottom && is_row_zero(input, top, in_w, epsilon)) top++;
    while (bottom >= top && is_row_zero(input, bottom, in_w, epsilon)) bottom--;
    while (left <= right && is_col_zero(input, in_h, left, epsilon)) left++;
    while (right >= left && is_col_zero(input, in_h, right, epsilon)) right++;

    *out_h = bottom - top + 1;
    *out_w = right - left + 1;

    if (*out_h <= 0 || *out_w <= 0)
        return NULL;

    float **trimmed = allocate_2d(*out_h, *out_w);
    for (int i = 0; i < *out_h; ++i)
        memcpy(trimmed[i], &input[top + i][left], (*out_w) * sizeof(float));

    return trimmed;
}

float **read_matrix(const char *filename, int *rows, int *cols) {
    printif("Reading matrix from %s\n", filename);
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("File open failed");
        return NULL;
    }

    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return NULL;
    }

    float **matrix = alloc_matrix_contiguous(*rows, *cols);
    if (!matrix) { fclose(fp); return NULL; }

    for (int i = 0; i < *rows; i++){
        for (int j = 0; j < *cols; j++){
            if (fscanf(fp, "%f", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading value at [%d][%d]\n", i, j);
                fclose(fp);
                free_2d(matrix);
                return NULL;
            }
        }
    }   

    fclose(fp);
    return matrix;
}

void print_matrix(float **matrix, int rows, int cols, int dp) {
    printif("Matrix (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printif("%.*f ",dp, matrix[i][j]);
        printif("\n");
    }
}

void save_matrix(const char *filename, float **matrix2d, int rows, int cols, int dp) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("File open failed");
        return;
    }

    fprintf(fp, "%d %d\n", rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            fprintf(fp, "%.*f ", dp, matrix2d[i][j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
}

/* Parallel random matrix generation - returns distributed portions */
float **generate_random_matrix_parallel(int rows, int cols, int *local_rows, int *local_start_row, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    if (rows <= 0 || cols <= 0) return NULL;
    
    // Synchronize random seed across all ranks
    unsigned int seed;
    if (rank == 0) {
        seed = (unsigned int)time(NULL);
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, comm);
    
    // Each rank generates a portion of rows
    int rows_per_proc = rows / size;
    int remainder = rows % size;
    *local_start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    *local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    printif("[Rank %d] Generating %d rows (from row %d)\n", rank, *local_rows, *local_start_row);
    
    // Allocate local portion
    float **local_matrix = allocate_2d(*local_rows, cols);
    if (!local_matrix) return NULL;
    
    // Generate random values in parallel with OpenMP
    #pragma omp parallel
    {
        unsigned int thread_seed = seed + (*local_start_row) * 1000 + omp_get_thread_num();
        
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < *local_rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Use thread-safe random generation
                local_matrix[i][j] = (float)rand_r(&thread_seed) / RAND_MAX;
            }
        }
    }
    
    return local_matrix;
}

/* Original sequential version for compatibility */
float **generate_random_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    float **matrix = allocate_2d(rows, cols);
    if (!matrix) return NULL;
    float *data = matrix[0];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[(size_t)i * cols + j] = (float)rand() / RAND_MAX;
        }
    }
    return matrix;
}

void free_matrix(float **matrix, int rows) {
    (void) rows;
    if (!matrix) return;
    free(matrix[0]);
    free(matrix);
}

/* Distribute matrix from rank 0 to all other ranks */
void distribute_matrix(float **full_matrix, int total_rows, int cols, 
                      float ***local_matrix, int *local_rows, int *local_start_row,
                      MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int rows_per_proc = total_rows / size;
    int remainder = total_rows % size;
    
    *local_start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    *local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Handle case where we have more ranks than rows
    if (*local_rows <= 0) {
        *local_rows = 0;
        *local_start_row = 0;
        *local_matrix = NULL;
        printif("[Rank %d] No rows assigned (matrix has %d rows, %d ranks)\n", rank, total_rows, size);
        return;
    }
    
    if (rank == 0) {
        printif("Distributing matrix of size %dx%d to %d ranks\n", total_rows, cols, size);
    }
    
    // Allocate local portion
    *local_matrix = allocate_2d(*local_rows, cols);
    if (!(*local_matrix)) {
        fprintf(stderr, "[Rank %d] Failed to allocate local matrix portion\n", rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    
    if (rank == 0) {
        if (!full_matrix) {
            fprintf(stderr, "[Rank 0] Error: full_matrix is NULL in distribute_matrix\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
        
        // Copy rank 0's portion
        for (int i = 0; i < *local_rows; i++) {
            memcpy((*local_matrix)[i], full_matrix[*local_start_row + i], cols * sizeof(float));
        }
        
        // Send to other ranks
        for (int r = 1; r < size; r++) {
            int r_start = r * rows_per_proc + (r < remainder ? r : remainder);
            int r_rows = rows_per_proc + (r < remainder ? 1 : 0);
            
            // Skip ranks with no rows
            if (r_rows <= 0) continue;
            
            for (int i = 0; i < r_rows; i++) {
                MPI_Send(full_matrix[r_start + i], cols, MPI_FLOAT, r, 100 + i, comm);
            }
        }
        printif("[Rank 0] Distributed matrix to all ranks\n");
    } else {
        // Receive from rank 0 only if we have rows assigned
        if (*local_rows > 0) {
            for (int i = 0; i < *local_rows; i++) {
                MPI_Recv((*local_matrix)[i], cols, MPI_FLOAT, 0, 100 + i, comm, MPI_STATUS_IGNORE);
            }
            printif("[Rank %d] Received local matrix with %d rows (from row %d)\n", 
                    rank, *local_rows, *local_start_row);
        }
    }
}

void conv2d_stride_2d_MPI_OMP(float **input2d, int local_in_rows, int local_start_row,
                              int total_in_h, int in_w,
                              float **kernel, int k_h, int k_w,
                              int stride_h, int stride_w,
                              int *out_h, int *out_w,
                              float ***output, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0)
        printif("[Rank %d] Starting conv2d_stride_2d_MPI_OMP with %d processes (VIRTUAL PADDING)...\n", rank, size);

    double start_time = MPI_Wtime();

    // Calculate padding based on kernel size (same/valid padding)
    int pad_top = (k_h - 1) / 2;
    int pad_bottom = (k_h - 1) - pad_top;
    int pad_left = (k_w - 1) / 2;
    int pad_right = (k_w - 1) - pad_left;
    
    // Virtual padded dimensions
    int padded_h = total_in_h + pad_top + pad_bottom;
    int padded_w = in_w + pad_left + pad_right;
    
    if (rank == 0) {
        printif("[Rank %d] Original: %dx%d, Virtual padding: top=%d bottom=%d left=%d right=%d, Padded: %dx%d\n",
                rank, total_in_h, in_w, pad_top, pad_bottom, pad_left, pad_right, padded_h, padded_w);
        printif("[Rank %d] Memory saved by virtual padding: %.2f MB\n",
                rank, (padded_h * padded_w - total_in_h * in_w) * sizeof(float) / (1024.0 * 1024.0));
    }

    // 1. Compute output dimensions based on padded size
    *out_h = (padded_h - k_h) / stride_h + 1;
    *out_w = (padded_w - k_w) / stride_w + 1;

    // 2. Flatten and broadcast kernel
    size_t kernel_size = (size_t)k_h * k_w;
    float *kernel1d = (float*)malloc(kernel_size * sizeof(float));
    if (!kernel1d) MPI_Abort(comm, EXIT_FAILURE);
    if (rank == 0) {
        for (int i = 0; i < k_h; ++i)
            for (int j = 0; j < k_w; ++j)
                kernel1d[i*k_w + j] = kernel[i][j];
    }
    MPI_Bcast(kernel1d, (int)kernel_size, MPI_FLOAT, 0, comm);

    // 3. Compute local output rows for each rank
    int rows_per_proc = (*out_h) / size;
    int remainder = (*out_h) % size;
    int local_out_start = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int local_out_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int local_out_end = local_out_start + local_out_rows;

    // 4. Determine what input rows this rank needs from OTHER ranks
    // Map output rows to padded input rows
    int padded_input_start = local_out_start * stride_h;
    int padded_input_end = (local_out_end - 1) * stride_h + k_h - 1;
    
    // Convert to original coordinates
    int needed_orig_start = padded_input_start - pad_top;
    int needed_orig_end = padded_input_end - pad_top;
    
    // Clamp to valid range
    if (needed_orig_start < 0) needed_orig_start = 0;
    if (needed_orig_end >= total_in_h) needed_orig_end = total_in_h - 1;
    
    int needed_rows = needed_orig_end - needed_orig_start + 1;

    printif("[Rank %d] Needs rows %d to %d (%d total) for output rows %d to %d\n",
            rank, needed_orig_start, needed_orig_end, needed_rows, local_out_start, local_out_end - 1);

    // 5. Use MPI_Alltoallv to exchange data
    // First, determine what each rank needs from every other rank
    int *send_counts = (int*)calloc(size, sizeof(int));
    int *send_displs = (int*)calloc(size, sizeof(int));
    int *recv_counts = (int*)calloc(size, sizeof(int));
    int *recv_displs = (int*)calloc(size, sizeof(int));
    
    if (!send_counts || !send_displs || !recv_counts || !recv_displs) MPI_Abort(comm, EXIT_FAILURE);
    
    // Calculate what I need to receive from each rank
    for (int r = 0; r < size; r++) {
        int r_rows_per_proc = total_in_h / size;
        int r_remainder = total_in_h % size;
        int r_start = r * r_rows_per_proc + (r < r_remainder ? r : r_remainder);
        int r_rows = r_rows_per_proc + (r < r_remainder ? 1 : 0);
        int r_end = r_start + r_rows - 1;
        
        // Check overlap with needed range
        int overlap_start = (needed_orig_start > r_start) ? needed_orig_start : r_start;
        int overlap_end = (needed_orig_end < r_end) ? needed_orig_end : r_end;
        
        if (overlap_end >= overlap_start && r_rows > 0) {
            recv_counts[r] = (overlap_end - overlap_start + 1) * in_w;
            recv_displs[r] = (overlap_start - needed_orig_start) * in_w;
        } else {
            recv_counts[r] = 0;
            recv_displs[r] = 0;
        }
    }
    
    // Use MPI_Alltoall to exchange how much each rank wants from me
    MPI_Alltoall(recv_counts, 1, MPI_INT, send_counts, 1, MPI_INT, comm);
    
    // Calculate send displacements based on what others need from me
    int total_send = 0;
    for (int r = 0; r < size; r++) {
        if (send_counts[r] > 0) {
            // Figure out which of my rows to send
            // Others are asking for rows in terms of global indices
            // Need to map back through the recv_counts logic
            int r_out_rows_per_proc = (*out_h) / size;
            int r_out_remainder = (*out_h) % size;
            int r_out_start = r * r_out_rows_per_proc + (r < r_out_remainder ? r : r_out_remainder);
            int r_out_end = r_out_start + (r_out_rows_per_proc + (r < r_out_remainder ? 1 : 0)) - 1;
            
            int r_pad_start = r_out_start * stride_h;
            int r_pad_end = r_out_end * stride_h + k_h - 1;
            
            int r_need_start = r_pad_start - pad_top;
            int r_need_end = r_pad_end - pad_top;
            
            if (r_need_start < 0) r_need_start = 0;
            if (r_need_end >= total_in_h) r_need_end = total_in_h - 1;
            
            // Overlap with what I have
            int overlap_start = (r_need_start > local_start_row) ? r_need_start : local_start_row;
            int overlap_end = (r_need_end < local_start_row + local_in_rows - 1) ? r_need_end : (local_start_row + local_in_rows - 1);
            
            if (overlap_end >= overlap_start) {
                send_displs[r] = (overlap_start - local_start_row) * in_w;
            }
        }
        total_send += send_counts[r];
    }
    
    // Allocate buffers
    float *full_needed_data = (float*)malloc((size_t)needed_rows * in_w * sizeof(float));
    if (!full_needed_data) MPI_Abort(comm, EXIT_FAILURE);
    
    // Prepare send buffer (just use input2d directly as it's contiguous)
    float *send_buf = (local_in_rows > 0) ? input2d[0] : NULL;
    
    printif("[Rank %d] Exchanging data via Alltoallv (sending %d floats, receiving %d floats)\n",
            rank, total_send, needed_rows * in_w);
    
    // Perform the all-to-all exchange
    MPI_Alltoallv(send_buf, send_counts, send_displs, MPI_FLOAT,
                  full_needed_data, recv_counts, recv_displs, MPI_FLOAT,
                  comm);
    
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);

    printif("[Rank %d] Data exchange complete\n", rank);

    // 6. Allocate local output buffer
    float *local_output = (float*)calloc((size_t)local_out_rows * (*out_w), sizeof(float));
    if (!local_output && local_out_rows > 0) MPI_Abort(comm, EXIT_FAILURE);
    
    double local_start = MPI_Wtime();
    
    // 7. Perform convolution with VIRTUAL PADDING using OpenMP
    #pragma omp parallel 
    {
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < local_out_rows; i++) {
            for (int j = 0; j < *out_w; j++) {
                float sum = 0.0f;
                int global_out_i = local_out_start + i;
                
                // Map to padded coordinates
                int padded_i = global_out_i * stride_h;
                int padded_j = j * stride_w;

                #pragma omp loop collapse(2) reduction(+:sum) 
                for (int ki = 0; ki < k_h; ki++) {
                    for (int kj = 0; kj < k_w; kj++) {
                        // Padded coordinates
                        int pi = padded_i + ki;
                        int pj = padded_j + kj;
                        
                        // Convert to original coordinates
                        int orig_i = pi - pad_top;
                        int orig_j = pj - pad_left;
                        
                        float val;
                        // Check bounds and apply virtual padding
                        if (orig_i < 0 || orig_i >= total_in_h || orig_j < 0 || orig_j >= in_w) {
                            val = 0.0f;  // Virtual padding
                        } else {
                            // Access from full_needed_data
                            int needed_i = orig_i - needed_orig_start;
                            if (needed_i >= 0 && needed_i < needed_rows) {
                                val = full_needed_data[needed_i * in_w + orig_j];
                            } else {
                                val = 0.0f;
                            }
                        }
                        
                        sum += val * kernel1d[ki*k_w + kj];
                    }
                }
                local_output[i*(*out_w) + j] = sum;
            }
        }
    }
    
    double local_end = MPI_Wtime();
    
    // 8. Prepare for gathering results
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    int offset = 0;
    for (int r = 0; r < size; r++) {
        int r_out_rows = rows_per_proc + (r < remainder ? 1 : 0);
        sendcounts[r] = r_out_rows * (*out_w);
        displs[r] = offset;
        offset += sendcounts[r];
    }

    float *output1d = NULL;
    if (rank == 0) {
        output1d = (float*)malloc((size_t)(*out_h) * (*out_w) * sizeof(float));
        if (!output1d) MPI_Abort(comm, EXIT_FAILURE);
    }

    // 9. Chunked Gatherv to handle large data
    const size_t CHUNK_BYTES = 128 * 1024 * 1024; // 128MB
    const size_t CHUNK_FLOATS = CHUNK_BYTES / sizeof(float);
    size_t total_elems = (size_t)(*out_h) * (*out_w);
    
    size_t rank_global_start = (size_t) displs[rank];
    size_t rank_global_end = rank_global_start + (size_t) sendcounts[rank];

    size_t offset2 = 0;
    int *chunk_recvcounts = malloc(size * sizeof(int));
    int *chunk_recvdispls = malloc(size * sizeof(int));
    if(!chunk_recvcounts || !chunk_recvdispls) MPI_Abort(comm, EXIT_FAILURE);

    while (offset2 < total_elems) {
        size_t remaining = total_elems - offset2;
        size_t chunk = remaining < CHUNK_FLOATS ? remaining : CHUNK_FLOATS;

        for (int r = 0; r < size; ++r) {
            size_t r_start = (size_t) displs[r];
            size_t r_end   = r_start + (size_t) sendcounts[r];

            size_t ov_start = (r_start > offset2) ? r_start : offset2;
            size_t ov_end   = (r_end < offset2 + chunk) ? r_end : (offset2 + chunk);

            if (ov_end > ov_start) {
                size_t count = ov_end - ov_start;
                chunk_recvcounts[r] = (int) count;
                chunk_recvdispls[r] = (int) (ov_start - offset2);
            } else {
                chunk_recvcounts[r] = 0;
                chunk_recvdispls[r] = 0;
            }
        }

        size_t my_ov_start = (rank_global_start > offset2) ? rank_global_start : offset2;
        size_t my_ov_end   = (rank_global_end < offset2 + chunk) ? rank_global_end : (offset2 + chunk);

        int my_sendcount = 0;
        const float *my_sendptr = NULL;
        if (my_ov_end > my_ov_start) {
            my_sendcount = (int) (my_ov_end - my_ov_start);
            size_t local_off = my_ov_start - rank_global_start;
            my_sendptr = local_output + local_off;
        } else {
            my_sendcount = 0;
            my_sendptr = local_output;
        }

        float *root_chunk_ptr = NULL;
        if (rank == 0) {
            root_chunk_ptr = output1d + offset2;
        }

        MPI_Gatherv((void*) my_sendptr, my_sendcount, MPI_FLOAT,
                    root_chunk_ptr, chunk_recvcounts, chunk_recvdispls, MPI_FLOAT,
                    0, comm);

        offset2 += chunk;
    }

    free(chunk_recvcounts);
    free(chunk_recvdispls);

    double end_time = MPI_Wtime();
    
    // 10. Build 2D view on root
    if (rank == 0 && output1d) {
        double local_compute_time = local_end - local_start;
        double total_time = end_time - start_time;
        float **view = (float**)malloc((*out_h) * sizeof(float*));
        view[0] = output1d;
        for (int i = 1; i < *out_h; i++)
            view[i] = view[0] + i*(*out_w);
        *output = view;
        
        if (record) {
            FILE *fp = fopen("conv2d_log.csv", "a");
            if (fp) {
                fseek(fp, 0, SEEK_END);
                if (ftell(fp) == 0)
                    fprintf(fp, "in_h,in_w,k_h,k_w,stride_h,stride_w,out_h,out_w,num_procs,num_threads,local_time_s,total_time_s,mode\n");
                fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,MPI_FULLY_DISTRIBUTED\n",
                        total_in_h, in_w, k_h, k_w, stride_h, stride_w,
                        *out_h, *out_w, size, omp_get_max_threads(),
                        local_compute_time, total_time);
                fclose(fp);
            }
        }
    }

    // 11. Cleanup
    free(full_needed_data);
    free(local_output);
    free(kernel1d);
    free(sendcounts);
    free(displs);
}

bool arrays_equal(float** arr1, float** arr2, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (arr1[i][j] != arr2[i][j]) {
                printif("Mismatch at (%d,%d): %f != %f\n", i, j, arr1[i][j], arr2[i][j]);
                return false;
            }
        }
    }
    return true;
}

void free_matrix_contiguous(float **mat) {
    if (!mat) return;
    free(mat[0]);
    free(mat);
}

int main(int argc, char *argv[]) {
    int height = 0, width = 0, kheight = 0, kwidth = 0, swidth = 1, sheight = 1;
    char *save = "", *ksave = "", *osave = "", *ffile = "f1.txt", *gfile = "g1.txt", *ofile = "o1.txt";
    
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-H") == 0) {
            height = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-W") == 0) {
            width = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-kH") == 0) {
            kheight = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-kW") == 0) {
            kwidth = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-o") == 0) {
            osave = argv[i + 1];
        } else if (strcmp(argv[i], "-f") == 0) {
            save = argv[i + 1];
        } else if (strcmp(argv[i], "-g") == 0) {
            ksave = argv[i + 1];
        } else if (strcmp(argv[i], "-ff") == 0) {
            ffile = argv[i + 1];
        } else if (strcmp(argv[i], "-fg") == 0) {
            gfile = argv[i + 1];
        } else if (strcmp(argv[i], "-sH") == 0) {
            sheight = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-sW") == 0) {
            swidth = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-p") == 0) {
            print = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            record = 1;
        }
    }
    
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
      
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printif("[Rank %d] Started (total %d ranks)\n", rank, size);

    /* Create kernel Matrix */
    float **kmatrix = NULL;
    int local_k_rows = 0, local_k_start = 0;
    
    // Determine if we're generating or reading
    int k_generating = (kheight > 0 && kwidth > 0) ? 1 : 0;
    
    if (k_generating) {
        // Generate kernel in parallel across all ranks
        kmatrix = generate_random_matrix_parallel(kheight, kwidth, &local_k_rows, &local_k_start, MPI_COMM_WORLD);
        if (!kmatrix) {
            printif("[Rank %d] Kernel memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Broadcast dimensions
        MPI_Bcast(&kheight, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&kwidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Gather full kernel on rank 0 for saving and broadcasting
        if (rank == 0) {
            float **full_kernel = allocate_2d(kheight, kwidth);
            // Copy rank 0's portion
            for (int i = 0; i < local_k_rows; i++) {
                memcpy(full_kernel[local_k_start + i], kmatrix[i], kwidth * sizeof(float));
            }
            // Receive from others
            for (int r = 1; r < size; r++) {
                int r_rows_per_proc = kheight / size;
                int r_remainder = kheight % size;
                int r_start = r * r_rows_per_proc + (r < r_remainder ? r : r_remainder);
                int r_rows = r_rows_per_proc + (r < r_remainder ? 1 : 0);
                for (int i = 0; i < r_rows; i++) {
                    MPI_Recv(full_kernel[r_start + i], kwidth, MPI_FLOAT, r, i + 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            // Free distributed and use full
            free_matrix(kmatrix, local_k_rows);
            kmatrix = full_kernel;
        } else {
            // Send to rank 0
            for (int i = 0; i < local_k_rows; i++) {
                MPI_Send(kmatrix[i], kwidth, MPI_FLOAT, 0, i + 2000, MPI_COMM_WORLD);
            }
            // Free after sending
            free_matrix(kmatrix, local_k_rows);
            kmatrix = NULL;
        }
    } else {
        // Read from file (only rank 0)
        if (rank == 0) {
            kmatrix = read_matrix(gfile, &kheight, &kwidth);
            if (!kmatrix) {          
                printif("Error reading kernel matrix.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        // Broadcast kernel dimensions to all ranks
        MPI_Bcast(&kheight, 1, MPI_INT, 0, MPI_COMM_WORLD); 
        MPI_Bcast(&kwidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // Ensure all ranks are synchronized
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Create Feature Matrix - stays distributed! */
    float **local_matrix = NULL;
    int local_rows = 0, local_start_row = 0;
    float **full_matrix_rank0 = NULL;  // Only for saving on rank 0
    
    // Determine if we're generating or reading
    int generating = (height > 0 && width > 0) ? 1 : 0;
    
    if (generating) {
        // Generate matrix in parallel - KEEP DISTRIBUTED
        local_matrix = generate_random_matrix_parallel(height, width, &local_rows, &local_start_row, MPI_COMM_WORLD);
        if (!local_matrix) {
            printif("[Rank %d] Feature matrix memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Broadcast dimensions to all ranks
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Determine if we need to gather for saving
        int need_gather = (strlen(save) > 0) ? 1 : 0;
        
        // If saving is requested, gather on rank 0
        if (need_gather) {
            if (rank == 0) {
                full_matrix_rank0 = allocate_2d(height, width);
                // Copy rank 0's portion
                for (int i = 0; i < local_rows; i++) {
                    memcpy(full_matrix_rank0[local_start_row + i], local_matrix[i], width * sizeof(float));
                }
                // Receive from others
                for (int r = 1; r < size; r++) {
                    int r_rows_per_proc = height / size;
                    int r_remainder = height % size;
                    int r_start = r * r_rows_per_proc + (r < r_remainder ? r : r_remainder);
                    int r_rows = r_rows_per_proc + (r < r_remainder ? 1 : 0);
                    for (int i = 0; i < r_rows; i++) {
                        MPI_Recv(full_matrix_rank0[r_start + i], width, MPI_FLOAT, r, i + 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            } else {
                // Send to rank 0 for saving
                for (int i = 0; i < local_rows; i++) {
                    MPI_Send(local_matrix[i], width, MPI_FLOAT, 0, i + 1000, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        // Read from file - rank 0 reads, then distributes
        float **temp_matrix = NULL;
        
        if (rank == 0) {
            temp_matrix = read_matrix(ffile, &height, &width);
            if (!temp_matrix) {          
                printif("Error reading feature matrix.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            print_matrix(temp_matrix, height, width, 3);
        }
        
        // Broadcast dimensions to all ranks
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Check if we have enough rows for all ranks
        if (height < size && rank == 0) {
            printif("WARNING: Matrix has %d rows but %d ranks. Some ranks will have 0 rows.\n", height, size);
        }
        
        // Distribute to all ranks
        if (rank == 0) {
            distribute_matrix(temp_matrix, height, width, &local_matrix, &local_rows, &local_start_row, MPI_COMM_WORLD);
            
            // Keep full matrix on rank 0 for saving if needed
            if (strlen(save) > 0) {
                full_matrix_rank0 = temp_matrix;
            } else {
                free_matrix(temp_matrix, height);
            }
        } else {
            distribute_matrix(NULL, height, width, &local_matrix, &local_rows, &local_start_row, MPI_COMM_WORLD);
        }
    }
    
    // Ensure all ranks have synchronized
    MPI_Barrier(MPI_COMM_WORLD);
    
    omp_set_nested(1);
    
    printif("[Rank %d] Starting convolution with stride %d,%d (FULLY DISTRIBUTED, NO GATHERING)\n", 
            rank, sheight, swidth);
    printif("[Rank %d] Local matrix portion: %d rows (from row %d to %d)\n",
            rank, local_rows, local_start_row, local_start_row + local_rows - 1);
    
    int out_height, out_width;
    float **out_matrix = NULL;
    
    // Pass distributed local portion directly to convolution
    conv2d_stride_2d_MPI_OMP(local_matrix, local_rows, local_start_row,
                             height, width, kmatrix, kheight, kwidth,
                             sheight, swidth, &out_height, &out_width, 
                             &out_matrix, MPI_COMM_WORLD);
    
    printif("[Rank %d] Convolution done\n", rank);
    
    if (rank == 0) {
        printif("Output Matrix size: %dx%d\n", out_height, out_width);
    }
    
    // Free local matrix portions - each rank frees its own
    if (local_matrix) {
        free_matrix(local_matrix, local_rows);
    }
    
    MPI_Finalize();
    
    if (rank == 0) {
        // Save outputs if requested
        if (strlen(save) > 0 && full_matrix_rank0) {
            save_matrix(save, full_matrix_rank0, height, width, 3);
            free_matrix(full_matrix_rank0, height);
        }
        if (strlen(ksave) > 0 && kmatrix) {
            save_matrix(ksave, kmatrix, kheight, kwidth, 3);
        }
        if (strlen(osave) > 0 && out_matrix) {
            save_matrix(osave, out_matrix, out_height, out_width, 3);
        }
        
        // Cleanup
        if (kmatrix) free_matrix(kmatrix, kheight);
        if (out_matrix) free_matrix_contiguous(out_matrix);
    }
    
    return 0;
}