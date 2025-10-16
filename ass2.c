/* patched_ass2.c  -- minimal edits from original ass2.c to fix memory issues.
   Key fixes:
   - allocate_2d now allocates a single contiguous block and sets row pointers.
   - generate_random_matrix simplified to fill contiguous storage.
   - free_2d updated to free contiguous block.
   (Other logic unchanged.)
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

/* --- CHANGED: allocate_2d now allocates a single contiguous block and returns
      row pointers into it. This makes it safe to free with free(matrix[0]); free(matrix);
      which matches how pad_matrix() and other code free matrices. --- */
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

/* Helper: Free 2D matrix (now assumes contiguous allocation) */
void free_2d(float **mat) {
    if (!mat) return;
    free(mat[0]); // contiguous block
    free(mat);    // row pointers
}

/* Helper: Check if a row is all zeros */
bool is_row_zero(float **mat, int row, int w, float epsilon) {
    for (int j = 0; j < w; ++j)
        if (fabs(mat[row][j]) > epsilon)
            return false;
    return true;
}

/* Helper: Check if a column is all zeros */
bool is_col_zero(float **mat, int h, int col, float epsilon) {
    for (int i = 0; i < h; ++i)
        if (fabs(mat[i][col]) > epsilon)
            return false;
    return true;
}

/* Main function to remove all-zero rows and columns */
float **remove_zero_padding(float **input, int in_h, int in_w, int *out_h, int *out_w) {
    const float epsilon = 1e-6f;  // threshold for float zero

    // Identify non-zero row and column bounds
    int top = 0, bottom = in_h - 1;
    int left = 0, right = in_w - 1;

    while (top <= bottom && is_row_zero(input, top, in_w, epsilon)) top++;
    while (bottom >= top && is_row_zero(input, bottom, in_w, epsilon)) bottom--;
    while (left <= right && is_col_zero(input, in_h, left, epsilon)) left++;
    while (right >= left && is_col_zero(input, in_h, right, epsilon)) right--;

    // New size
    *out_h = bottom - top + 1;
    *out_w = right - left + 1;

    if (*out_h <= 0 || *out_w <= 0)
        return NULL;  // all zeros

    // Allocate new matrix
    float **trimmed = allocate_2d(*out_h, *out_w);

    // Copy values
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

    // Read matrix dimensions
    if (fscanf(fp, "%d %d", rows, cols) != 2) {
        fclose(fp);
        return NULL;
    }

    // Allocate matrix
    float **matrix = alloc_matrix_contiguous(*rows, *cols);
    if (!matrix) { fclose(fp); return NULL; }

    // Read matrix data
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

    // Write matrix dimensions
    fprintf(fp, "%d %d\n", rows, cols);

    // Write matrix data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            fprintf(fp, "%.*f ", dp,matrix2d[i][j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
}

/* --- CHANGED: generate_random_matrix now assumes allocate_2d's contiguous layout --- */
float **generate_random_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    float **matrix = allocate_2d(rows, cols);
    if (!matrix) return NULL;
    float *data = matrix[0];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[(size_t)i * cols + j] = (float)rand() / RAND_MAX; // random float in [0,1)
        }
    }
    return matrix;
}

/* Keep consistent name that frees contiguous block */
void free_matrix(float **matrix, int rows) {
    (void) rows;
    if (!matrix) return;
    free(matrix[0]);
    free(matrix);
}

float **pad_matrix(float **matrix, int rows, int cols,
                               int pad_top, int pad_bottom,
                               int pad_left, int pad_right)
{
    int new_rows = rows + pad_top + pad_bottom;
    int new_cols = cols + pad_left + pad_right;

    printif("Padding matrix from %dx%d to %dx%d (top=%d, bottom=%d, left=%d, right=%d)\n",
            rows, cols, new_rows, new_cols, pad_top, pad_bottom, pad_left, pad_right);

    // Allocate single contiguous block for padded data
    float *data = (float *)calloc((size_t)new_rows * new_cols, sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed in pad_matrix_free_source\n");
        return NULL;
    }

    float **padded = (float **)malloc(new_rows * sizeof(float *));
    if (!padded) {
        free(data);
        fprintf(stderr, "Error: Memory allocation failed for row pointers\n");
        return NULL;
    }

    // Assign row pointers
    for (int i = 0; i < new_rows; ++i)
        padded[i] = data + (size_t)i * new_cols;

    // Copy and free source rows as soon as they’re done
    for (int i = 0; i < rows; ++i) {
        memcpy(&padded[i + pad_top][pad_left], matrix[i], cols * sizeof(float));
    }

    // Free the pointer array of the original matrix (assumes contiguous original)
    if (matrix) {
        free(matrix[0]);
        free(matrix);
    }

    printif("Padding complete using %.2f MB total.\n",
            (new_rows * (size_t)new_cols * sizeof(float)) / (1024.0 * 1024.0));

    return padded;
}

/* (The rest of the code is unchanged; for brevity I omit the large conv function bodies here in the snippet)
   In your copy keep the existing conv2d_stride_2d_MPI_OMP, conv helpers and main() with no further changes,
   because they already rely on contiguous buffers. */





// float** conv2d_stride_2d(float **input2d, int in_h, int in_w,
//                          float **kernel, int k_h, int k_w,
//                          int stride_h, int stride_w,
//                          int *out_h, int *out_w) {
//     //printif("Starting conv2d_stride_2d...\n");
//     // Convert input2d to 1D array
//     //printif("in_h=%d in_w=%d sizeof(float)=%zu total=%zu\n",
//     //    in_h, in_w, sizeof(float), (size_t)in_h * in_w * sizeof(float));
//     size_t total = (size_t)in_h * (size_t)in_w * sizeof(float);
//     float *input = malloc(total);
//     //printif("Input dimensions: %dx%d\n", in_h, in_w);
//     // print_matrix(input2d, in_h, in_w,3);
//     //printif("Input array allocated.\n");
//     if (!input) return NULL;
//     for (int i = 0; i < in_h; ++i){
//         for (int j = 0; j < in_w; ++j){
//             //printif("Copying input2d[%d][%d] =  to input[%d]\n", i, j, i * in_w + j);
//             input[i * in_w + j] = input2d[i][j];
//             //printif("input");
//         }
//     }
//     free_matrix(input2d, in_h);
//     //printif("Input copy done.\n");
//     // *out_h = ceil((in_h - k_h) / stride_h) + 1;
//     // *out_w = ceil((in_w - k_w) / stride_w) + 1;
//     // *out_h = (int)ceil((float)in_h / stride_h);
//     // *out_w = (int)ceil((float)in_w / stride_w);
//     // *out_h = (int)ceil((float)(in_h - k_h + 1) / stride_h);
//     // *out_w = (int)ceil((float)(in_w - k_w + 1) / stride_w);
//     *out_h = (in_h - k_h) / stride_h + 1;   // integer division -> floor((in_h-k)/stride)+1
//     *out_w = (in_w - k_w) / stride_w + 1;   
//     // while (( (*out_h - 1) * stride_h + k_h ) > in_h) (*out_h)--;
//     // while (( (*out_w - 1) * stride_w + k_w ) > in_w) (*out_w)--;
//     //printif("Output dimensions: %dx%d\n", *out_h, *out_w);
//     //printif("Output from conv2d_stride_2d before reshaping:\n");
//     float **output = (float**)malloc((*out_h) * sizeof(float*));
//     float *output1d = (float*)malloc((*out_h) * (*out_w) * sizeof(float));
//     if (!output) { free(input); return NULL; }
//     for (int i = 0; i < *out_h; ++i) {
//         output[i] = (float*)malloc((*out_w) * sizeof(float));
//         if (!output[i]) {
//             for (int k = 0; k < i; ++k) free(output[k]);
//             free(output);
//             free(input);
//             return NULL;
//         }
//     }

//     double start = omp_get_wtime();
//     //printif("Output from conv2d_stride_2d before reshaping:\n");
//     for (int i = 0; i < *out_h; ++i) {
//         for (int j = 0; j < *out_w; ++j) {
//             float sum = 0.0f;
//             int in_i = i * stride_h;
//             int in_j = j * stride_w;
//             for (int ki = 0; ki < k_h; ++ki) {
//                 for (int kj = 0; kj < k_w; ++kj) {
//                     int ii = in_i + ki;
//                     int jj = in_j + kj;
//                     if (ii < in_h && jj < in_w) {   // boundary check!
//                         sum += input[ii * in_w + jj] * kernel[ki][kj];
//                     }
//                 }
//             }
//             output1d[i*(*out_w)+j] = sum;
//         }
//     }
//     double end = omp_get_wtime();
//     double local_compute_time = end - start;
//     double total_time = end - start;
//     //printif("Output from conv2d_stride_2d before reshaping:\n");
//     for (int i = 0; i < *out_h; ++i) {
//         for (int j = 0; j < *out_w; ++j) {
//             output[i][j] = output1d[i*(*out_w)+j];
//         }
//     }
//     if(record)
//         {
//             const char *filename = "conv2d_log.csv";
//             FILE *fp = fopen(filename, "a");  // open in append mode
//             if (fp == NULL) {
//                 fprintf(stderr, "Error: Could not open %s for writing.\n", filename);
//             } else {
//                 // If file is empty, write header
//                 fseek(fp, 0, SEEK_END);
//                 long size = ftell(fp);
//                 if (size == 0) {
//                     fprintf(fp, "in_h,in_w,k_h,k_w,stride_h,stride_w,out_h,out_w,num_procs,num_threads,local_time_s,total_time_s,type\n");
//                 }

//                 fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%ld,%d,%.6f,%.6f,none\n",
//                         in_h, in_w, k_h, k_w, stride_h, stride_w, *out_h, *out_w,
//                         size, 0,
//                         local_compute_time, total_time);

//                 fclose(fp);
//             printif("[Rank 0] Logged results to %s\n", filename);
//             }
//         }
//     free(input);
//     return output;
// }
// float** conv2d_stride_2d_MPI(float **input2d, int in_h, int in_w,
//                              float **kernel, int k_h, int k_w,
//                              int stride_h, int stride_w,
//                              int *out_h, int *out_w)
// {
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // if (rank == 0)
//         //printif("Starting conv2d_stride_2d_MPI with %d processes...\n", size);

//     // Flatten input
//     float *input = (float*)malloc(in_h * in_w * sizeof(float));
//     for (int i = 0; i < in_h; ++i)
//         for (int j = 0; j < in_w; ++j)
//             input[i * in_w + j] = input2d[i][j];

//     // Compute output size
//     *out_h = (in_h - k_h) / stride_h + 1;
//     *out_w = (in_w - k_w) / stride_w + 1;

//     // Determine workload for each process
//     int rows_per_proc = (*out_h) / size;
//     int remainder = (*out_h) % size;

//     int *sendcounts = (int*)malloc(size * sizeof(int));
//     int *displs     = (int*)malloc(size * sizeof(int));
//     int offset = 0;
//     for (int i = 0; i < size; i++) {
//         int rows = rows_per_proc + (i < remainder ? 1 : 0);
//         sendcounts[i] = rows * (*out_w);
//         displs[i] = offset;
//         offset += sendcounts[i];
//     }

//     // Allocate output buffers
//     int local_rows = sendcounts[rank] / (*out_w);
//     float *local_output = (float*)calloc(local_rows * (*out_w), sizeof(float));

//     // Each process computes its assigned output rows
//     int start_row = 0;
//     for (int i = 0; i < rank; i++)
//         start_row += sendcounts[i] / (*out_w);

//     for (int i = 0; i < local_rows; i++) {
//         int global_i = start_row + i;
//         for (int j = 0; j < *out_w; j++) {
//             float sum = 0.0f;
//             int in_i = global_i * stride_h;
//             int in_j = j * stride_w;
//             for (int ki = 0; ki < k_h; ki++)
//                 for (int kj = 0; kj < k_w; kj++) {
//                     int ii = in_i + ki;
//                     int jj = in_j + kj;
//                     if (ii < in_h && jj < in_w)
//                         sum += input[ii * in_w + jj] * kernel[ki][kj];
//                 }
//             local_output[i * (*out_w) + j] = sum;
//         }
//     }

//     // Gather all partial outputs on rank 0
//     float *output1d = NULL;
//     if (rank == 0)
//         output1d = (float*)malloc((*out_h) * (*out_w) * sizeof(float));

//     MPI_Gatherv(local_output, sendcounts[rank], MPI_FLOAT,
//                 output1d, sendcounts, displs, MPI_FLOAT,
//                 0, MPI_COMM_WORLD);

//     float **output2d = NULL;
//     if (rank == 0) {
//         output2d = (float**)malloc((*out_h) * sizeof(float*));
//         for (int i = 0; i < *out_h; i++) {
//             output2d[i] = (float*)malloc((*out_w) * sizeof(float));
//             for (int j = 0; j < *out_w; j++)
//                 output2d[i][j] = output1d[i * (*out_w) + j];
//         }
//     }

//     free(input);
//     free(local_output);
//     free(sendcounts);
//     free(displs);
//     if (rank == 0)
//         free(output1d);

//     return output2d;
// }


void conv2d_stride_2d_MPI_OMP(float **input2d, int in_h, int in_w,
                              float **kernel, int k_h, int k_w,
                              int stride_h, int stride_w,
                              int *out_h, int *out_w,
                              float ***output, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0)
        printif("[Rank %d] Starting conv2d_stride_2d_MPI_OMP with %d processes...\n", rank, size);

    double start_time = MPI_Wtime();

    // 1. Compute output dimensions
    *out_h = (in_h - k_h) / stride_h + 1;
    *out_w = (in_w - k_w) / stride_w + 1;

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

    // 4. Compute exact input rows needed (including kernel)
    int input_start_row = local_out_start * stride_h;
    int input_end_row = (local_out_end - 1) * stride_h + k_h - 1;
    if (input_end_row >= in_h) input_end_row = in_h - 1;
    int local_in_rows = input_end_row - input_start_row + 1;

    // 5. Allocate local input buffer
    float *local_input1d = (float*)malloc((size_t)local_in_rows * in_w * sizeof(float));
    if (!local_input1d) MPI_Abort(comm, EXIT_FAILURE);

    // 6. Copy input rows from rank 0
    if (rank == 0) {
        // copy local slice for rank 0
        for (int i = 0; i < local_in_rows; i++)
            memcpy(&local_input1d[i*in_w], input2d[input_start_row + i], in_w * sizeof(float));

        // send slices to other ranks
        for (int r = 1; r < size; r++) {
            int r_out_start = r * rows_per_proc + (r < remainder ? r : remainder);
            int r_out_rows = rows_per_proc + (r < remainder ? 1 : 0);
            int r_out_end = r_out_start + r_out_rows;

            int r_input_start = r_out_start * stride_h;
            int r_input_end = (r_out_end - 1) * stride_h + k_h - 1;
            if (r_input_end >= in_h) r_input_end = in_h - 1;
            int r_in_rows = r_input_end - r_input_start + 1;

            for (int i = 0; i < r_in_rows; i++)
                MPI_Send(input2d[r_input_start + i], in_w, MPI_FLOAT, r, 0, comm);
        }
    } else {
        // receive slice for this rank
        for (int i = 0; i < local_in_rows; i++)
            MPI_Recv(&local_input1d[i*in_w], in_w, MPI_FLOAT, 0, 0, comm, MPI_STATUS_IGNORE);
    }

    printif("[Rank %d] local_input1d rows=%d (input_start_row=%d, input_end_row=%d)\n",
            rank, local_in_rows, input_start_row, input_end_row);

    // 7. Allocate local output buffer
    float *local_output = (float*)calloc((size_t)local_out_rows * (*out_w), sizeof(float));
    if (!local_output && local_out_rows > 0) MPI_Abort(comm, EXIT_FAILURE);
    double local_start = MPI_Wtime();
    // 8. Perform convolution
        
    #pragma omp parallel 
    {

        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < local_out_rows; i++) {
            for (int j = 0; j < *out_w; j++) {
                float sum = 0.0f;
                int global_i = local_out_start + i;
                int in_i_local = global_i*stride_h - input_start_row;

                #pragma omp loop collapse(2) reduction(+:sum) 
                for (int ki = 0; ki < k_h; ki++) {
                    for (int kj = 0; kj < k_w; kj++) {
                        int ii = in_i_local + ki;
                        int jj = j*stride_w + kj;
                        sum += local_input1d[ii*in_w + jj] * kernel1d[ki*k_w + kj];
                    }
                }
                local_output[i*(*out_w) + j] = sum;
            }
        }
    }
    // 9. Gather results to root
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
    double local_end = MPI_Wtime();
    // MPI_Gatherv(local_output, local_out_rows * (*out_w), MPI_FLOAT,
    //             output1d, sendcounts, displs, MPI_FLOAT,
    //             0, comm);
    /**
     * Safe Version of gather that keeps transmissions under 2GB
     * by breaking the gather into multiple smaller gathers if needed.
     */
     // Constants: tune this to be safely < 2GB in bytes
    const size_t CHUNK_BYTES = 128 * 1024 * 1024; // 128MB
    const size_t CHUNK_FLOATS = CHUNK_BYTES / sizeof(float);

    size_t total_elems = (size_t)(*out_h) * (*out_w); // total elements in output1d

    // compute global interval owned by this rank (from original displs/sendcounts)
    size_t rank_global_start = (size_t) displs[rank];
    size_t rank_global_end = rank_global_start + (size_t) sendcounts[rank]; // one past end

    size_t offset2 = 0;
    int *chunk_recvcounts = malloc(size * sizeof(int));
    int *chunk_recvdispls = malloc(size * sizeof(int));
    if(!chunk_recvcounts || !chunk_recvdispls) MPI_Abort(comm, EXIT_FAILURE);

    while (offset2 < total_elems) {
        size_t remaining = total_elems - offset2;
        size_t chunk = remaining < CHUNK_FLOATS ? remaining : CHUNK_FLOATS;

        // Compute, for each rank r, how many elements of [offset, offset+chunk) come from r,
        // and where they go inside the chunk buffer (displacement relative to chunk start).
        for (int r = 0; r < size; ++r) {
            size_t r_start = (size_t) displs[r];
            size_t r_end   = r_start + (size_t) sendcounts[r];

            // overlap of [r_start, r_end) and [offset, offset+chunk)
            size_t ov_start = (r_start > offset2) ? r_start : offset2;
            size_t ov_end   = (r_end < offset2 + chunk) ? r_end : (offset2 + chunk);

            if (ov_end > ov_start) {
                size_t count = ov_end - ov_start;
                // recvcounts: how many elements to receive from rank r INTO this chunk
                chunk_recvcounts[r] = (int) count;
                // recvdispls: where in the root's chunk buffer to place data from rank r
                chunk_recvdispls[r] = (int) (ov_start - offset2); // relative to chunk start
            } else {
                chunk_recvcounts[r] = 0;
                chunk_recvdispls[r] = 0; // value unused when count==0
            }
        }

        // Now compute what THIS rank must send for this chunk
        // compute overlap with this rank's interval [rank_global_start, rank_global_end)
        size_t my_ov_start = (rank_global_start > offset2) ? rank_global_start : offset2;
        size_t my_ov_end   = (rank_global_end < offset2 + chunk) ? rank_global_end : (offset2 + chunk);

        int my_sendcount = 0;
        const float *my_sendptr = NULL;
        if (my_ov_end > my_ov_start) {
            my_sendcount = (int) (my_ov_end - my_ov_start);
            // local offset inside local_output: (my_ov_start - rank_global_start)
            size_t local_off = my_ov_start - rank_global_start;
            my_sendptr = local_output + local_off;
        } else {
            my_sendcount = 0;
            // MPI allows sending NULL when count==0 on many implementations,
            // but to be safe pass a valid pointer when count==0:
            my_sendptr = local_output; // won't be used because count==0
        }

        // root pointer for this chunk inside the big output buffer
        float *root_chunk_ptr = NULL;
        if (rank == 0) {
            root_chunk_ptr = output1d + offset2;
        }

        // Perform the gather for this chunk. Every process calls this.
        MPI_Gatherv((void*) my_sendptr, my_sendcount, MPI_FLOAT,
                    root_chunk_ptr, chunk_recvcounts, chunk_recvdispls, MPI_FLOAT,
                    0, comm);

        offset2 += chunk;
    }

    free(chunk_recvcounts);
    free(chunk_recvdispls);



    double end_time = MPI_Wtime();
    // printif("[Rank %d] Total time: %.4f s\n", rank, end_time - start_time);
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
                fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,MPI_BCAST\n",
                        in_h, in_w, k_h, k_w, stride_h, stride_w,
                        *out_h, *out_w, size, omp_get_max_threads(),
                        local_compute_time, total_time);
                fclose(fp);
            }
        }
    }
 

    // 11. Cleanup
    free(local_input1d);
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
    // free the contiguous data block pointed to by mat[0]
    free(mat[0]);
    // free the array of row pointers
    free(mat);
}

int main(int argc, char *argv[]) {
    int height = 0, width = 0,kheight = 0, kwidth = 0,swidth=1,sheight=1;
    char *save="", *ksave="", *osave="" , *ffile="f1.txt", *gfile="g1.txt", *ofile="o1.txt";
    // bool pre_padded=false;
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
        }else if (strcmp(argv[i], "-fg") == 0) {
            gfile = argv[i + 1];
        } else if (strcmp(argv[i], "-sH") == 0) {
            sheight = atoi(argv[i + 1]);
        }else if (strcmp(argv[i], "-sW") == 0) {
            swidth = atoi(argv[i + 1]);
        }else if (strcmp(argv[i], "-p") == 0) {
            print=1;
        }else if (strcmp(argv[i], "-r") == 0) {
            record=1;
        }
    }
        int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
      
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes


    /* Create kernel Matrix*/
    float **kmatrix;
    if( rank==0){
        if (kheight > 0 && kwidth > 0) {
            kmatrix = generate_random_matrix(kheight, kwidth);
            if (!kmatrix) {
               printif("Memory allocation failed\n");
                return 1;
            }
            // print_matrix(kmatrix, kheight, kwidth);
        } else {
            kmatrix =read_matrix(gfile, &kheight, &kwidth);
            if (!kmatrix) {          
               printif("Error reading matrix.\n");
                return 1;
            }
            // print_matrix(kmatrix, kheight, kwidth);
        }
    } else {
        kmatrix = NULL; // other ranks do not hold the full kernel
    }
    /* Create Feature Matrix*/
    float **matrix;
    float **padded_matrix;
    float padh, padw;
    int oh, ow;
    
    if( rank==0){
        if (height > 0 && width > 0) {
            matrix = generate_random_matrix(height, width);
            if (!matrix) {
               printif("Memory allocation failed\n");
                return 1;
            }
            // print_matrix(matrix, height, width);
        } else {
            matrix =read_matrix(ffile, &height, &width);
            if (!matrix) {          
               printif("Error reading matrix.\n");
                return 1;
            }
            print_matrix(matrix, height, width,3);
        }

       
        int pad_top    = (kheight - 1) / 2;
        int pad_bottom = (kheight - 1) - pad_top;
        int pad_left   = (kwidth - 1) / 2;
        int pad_right  = (kwidth - 1) - pad_left;

        oh = height + pad_top + pad_bottom;
        ow = width + pad_left + pad_right;

        padded_matrix = pad_matrix(matrix, height, width,
                                        pad_top, pad_bottom, pad_left, pad_right);
        // free_matrix(matrix, height); 
    } else {
       padded_matrix = NULL; // other ranks do not hold the full feature 
    }
    MPI_Bcast(&oh, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kheight, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&kwidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
   
    
    
    // if(pre_padded){
    //    printif("Input matrix is already padded, skipping padding step.\n");
    //     padh=0;
    //     padw=0;
       
    //     matrix=remove_zero_padding(matrix, height, width, &height, &width);
    //      oh=height;
    //     ow=width;
    //     print_matrix(matrix, height, width,3);
    // } else{
       /* Pad feature Matrix*/
        // padh=((float)kheight-1)/2;
        // padw=((float)kwidth-1)/2;
        // oh=height+2*(int)padh;
        // ow=width+2*(int)padw;
    // }

    // int pad_top    = (kheight - 1) / 2;
    // int pad_bottom = (kheight - 1) - pad_top;
    // int pad_left   = (kwidth - 1) / 2;
    // int pad_right  = (kwidth - 1) - pad_left;

    // oh = height + pad_top + pad_bottom;
    // ow = width + pad_left + pad_right;

    // float **padded_matrix = pad_matrix(matrix, height, width,
    //                                 pad_top, pad_bottom, pad_left, pad_right);
    //printif("Padding: %f, %f\n", padh, padw);
    // float **padded_matrix = pad_matrix(matrix, height, width, padh, padw);
    //printif("Padding: %f, %f\n", padh, padw);
    // print_matrix(padded_matrix, oh, ow,3);

    // print_matrix(padded_matrix, (int)(height+2*padh), (int)(width+2*padw));
    


     
  
    /**
     * Convolution code goes here
     * 
     */
    omp_set_nested(1);
    

    
   printif("Starting convolution with stride %d,%d\n",sheight,swidth);
    int out_height, out_width;
    float **out_matrix;
    conv2d_stride_2d_MPI_OMP(padded_matrix, oh, ow, kmatrix, kheight, kwidth,sheight,swidth, &out_height, &out_width, &out_matrix, MPI_COMM_WORLD);
   printif("Convolution done\n");
    // print_matrix(out_matrix, out_height, out_width,3);

    
   printif("Output Matrix size: %dx%d\n", out_height, out_width);
    
    /**
     * Uncomment to compare to baseline single-threaded code
     */

    // int out_height2, out_width2;
    // float **out_matrix2;
    // if(rank==0){
    //     out_matrix2 =conv2d_stride_2d(padded_matrix, oh, ow, kmatrix, kheight, kwidth,sheight,swidth, &out_height2, &out_width2);
    //     // free_matrix(padded_matrix, oh);
    // }
    
    
    //printif("%d out matrix1 pointer\n %d out matrix2 pointer\n",out_matrix,out_matrix2)
    MPI_Finalize();
    if(rank==0){
        //printif("%d out matrix1 pointer\n %d out matrix2 pointer\n",out_matrix,out_matrix2);

        /**
         * Uncomment to compare to baseline single-threaded code
         */
        // if(arrays_equal(out_matrix, out_matrix2, out_height, out_width)) {
        //     //printif("OMP and NON-OMP results match.\n");
        //    printif("OMP and NON-OMP results match.\n");
        // } else {
        //     //printif("OMP and NON-OMP results do NOT match.\n");
        //     //printif("%d out matrix1 pointer\n %d out matrix2 pointer\n",out_matrix,out_matrix2);
        //     // print_matrix(out_matrix, out_height, out_width);
        //     // print_matrix(out_matrix2, out_height2, out_width2);
        //     //printif("%d out matrix1 pointer\n %d out matrix2 pointer\n",out_matrix,out_matrix2);
        //     //printif("Saving out_matrix[0][0]=%f to out1.txt\n", out_matrix[0][0]);
        //     //printif("Saving out_matrix2[0][0]=%f to out2.txt\n", out_matrix2[0][0]);
        //     save_matrix("out2.txt", out_matrix2, out_height2, out_width2,3);
        //     save_matrix("out1.txt", out_matrix, out_height, out_width,3);
        //     // print_matrix(out_matrix, out_height, out_width,3);
        //     // print_matrix(out_matrix2, out_height2, out_width2,3);
        // }


        // print_matrix(out_matrix, out_height, out_width);
        // convolve_2d(matrix, height, width, kmatrix, kheight, kwidth, out_matrix);
        // print_matrix(out_matrix, out_height, out_width);
        //printif("saving Matrixes\n");
        if (strlen(save) > 0) {
            save_matrix(save, padded_matrix, height+2*padh, width+2*padw,3);
        }
        if (strlen(ksave) > 0) {
            save_matrix(ksave, kmatrix, kheight, kwidth,3);
        }
        if (strlen(osave) > 0) {
            save_matrix(osave, out_matrix, out_height, out_width,3);
        }
    }else{
        return 0;
    }
    
    free_matrix(kmatrix, kheight);
    // free_matrix(out_matrix, out_height);
    free_matrix_contiguous(out_matrix);
    return 0;
}

