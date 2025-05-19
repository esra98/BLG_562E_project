#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cmath>
#include <algorithm>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Error checking macro for cuSOLVER calls
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Error checking macro for cuBLAS calls
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for shrinkage operation
__global__ void shrinkageKernel(double* M, double* result, int size, double tau) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double val = M[idx];
        double absVal = fabs(val);
        double sign = (val > 0) ? 1.0 : ((val < 0) ? -1.0 : 0.0);
        result[idx] = sign * max(absVal - tau, 0.0);
    }
}

// RPCA implementation using CUDA
class CUDA_R_pca {
public:
    int rows, cols;
    double* d_D;        // Data matrix (on device)
    double* d_S;        // Sparse component (on device)
    double* d_Y;        // Lagrange multiplier (on device)
    double* d_L;        // Low-rank component (on device)
    double* d_temp;     // Temporary storage (on device)
    
    // SVD workspace
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
    double* d_U;        // Left singular vectors
    double* d_VT;       // Transposed right singular vectors
    double* d_S_values; // Singular values
    double* d_work;     // Workspace for SVD
    int worksize;       // Size of workspace
    int* devInfo;       // Device info for SVD
    
    double mu;
    double mu_inv;
    double lmbda;

    // Constructor
    CUDA_R_pca(const cv::Mat& image, double mu = -1, double lmbda = -1) {
        rows = image.rows;
        cols = image.cols;
        int size = rows * cols;
        
        // Initialize CUDA handles
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_D, size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S, size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Y, size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_L, size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(double)));
        
        // SVD related allocations
        CUDA_CHECK(cudaMalloc(&d_U, rows * rows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_VT, cols * cols * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S_values, std::min(rows, cols) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
        
        // Initialize workspace for SVD
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(
            cusolverHandle, rows, cols, &worksize));
        CUDA_CHECK(cudaMalloc(&d_work, worksize * sizeof(double)));

        // Copy image data to device
        std::vector<double> h_data(size);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                h_data[i * cols + j] = static_cast<double>(image.at<uchar>(i, j));
            }
        }
        CUDA_CHECK(cudaMemcpy(d_D, h_data.data(), size * sizeof(double), cudaMemcpyHostToDevice));
        
        // Initialize S and Y with zeros
        CUDA_CHECK(cudaMemset(d_S, 0, size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_Y, 0, size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_L, 0, size * sizeof(double)));
        
        // Initialize mu
        if (mu > 0) {
            this->mu = mu;
        } else {
            // Calculate Frobenius norm for mu initialization
            double norm;
            CUBLAS_CHECK(cublasDnrm2(cublasHandle, size, d_D, 1, &norm));
            this->mu = size / (4.0 * norm);
        }
        
        this->mu_inv = 1.0 / this->mu;
        
        // Initialize lambda
        if (lmbda > 0) {
            this->lmbda = lmbda;
        } else {
            this->lmbda = 1.0 / std::sqrt(std::max(rows, cols));
        }
    }
    
    // Destructor
    ~CUDA_R_pca() {
        // Free device memory
        CUDA_CHECK(cudaFree(d_D));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(d_Y));
        CUDA_CHECK(cudaFree(d_L));
        CUDA_CHECK(cudaFree(d_temp));
        
        // Free SVD related memory
        CUDA_CHECK(cudaFree(d_U));
        CUDA_CHECK(cudaFree(d_VT));
        CUDA_CHECK(cudaFree(d_S_values));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(devInfo));
        
        // Destroy CUDA handles
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
        CUBLAS_CHECK(cublasDestroy(cublasHandle));
    }
    
    // Frobenius norm calculation using cuBLAS
    double frobenius_norm(double* d_M) {
        double norm;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, rows * cols, d_M, 1, &norm));
        return norm;
    }
    
    // Shrinkage operation using CUDA kernel
    void shrink(double* d_M, double* d_result, double tau) {
        int blockSize = 256;
        int numBlocks = (rows * cols + blockSize - 1) / blockSize;
        shrinkageKernel<<<numBlocks, blockSize>>>(d_M, d_result, rows * cols, tau);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // SVD thresholding using cuSOLVER
    void svd_threshold(double* d_M, double* d_result, double tau) {
        // Perform SVD: M = U * S * VT
        CUSOLVER_CHECK(cusolverDnDgesvd(
            cusolverHandle, 'A', 'A', rows, cols,
            d_M, rows, d_S_values, d_U, rows, d_VT, cols,
            d_work, worksize, nullptr, devInfo));
        
        // Check for convergence
        int devInfo_h;
        CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) {
            std::cerr << "SVD did not converge. Info: " << devInfo_h << std::endl;
            return;
        }
        
        // Apply thresholding to singular values on CPU for simplicity
        std::vector<double> h_S_values(std::min(rows, cols));
        CUDA_CHECK(cudaMemcpy(h_S_values.data(), d_S_values, h_S_values.size() * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Apply shrinkage to singular values
        for (int i = 0; i < h_S_values.size(); ++i) {
            h_S_values[i] = std::max(h_S_values[i] - tau, 0.0);
        }
        
        CUDA_CHECK(cudaMemcpy(d_S_values, h_S_values.data(), h_S_values.size() * sizeof(double), cudaMemcpyHostToDevice));
        
        // Reconstruct the matrix: U * S * VT
        // First, multiply U by S (scale columns of U by singular values)
        CUDA_CHECK(cudaMemset(d_result, 0, rows * cols * sizeof(double)));
        
        // We'll construct U*S first, then multiply by VT
        double* d_US;
        CUDA_CHECK(cudaMalloc(&d_US, rows * cols * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_US, 0, rows * cols * sizeof(double)));
        
        // Scale columns of U by singular values to get U*S
        for (int i = 0; i < std::min(rows, cols); ++i) {
            double alpha = h_S_values[i];
            if (alpha > 0) {
                CUBLAS_CHECK(cublasDcopy(cublasHandle, rows, 
                              d_U + i * rows, 1, 
                              d_US + i * rows, 1));
                CUBLAS_CHECK(cublasDscal(cublasHandle, rows, 
                              &alpha, 
                              d_US + i * rows, 1));
            }
        }
        
        // Then multiply U*S by VT to get the final result
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(cublasHandle, 
                               CUBLAS_OP_N, CUBLAS_OP_T, 
                               rows, cols, std::min(rows, cols),
                               &alpha,
                               d_US, rows,
                               d_VT, cols, 
                               &beta,
                               d_result, rows));
        
        CUDA_CHECK(cudaFree(d_US));
    }
    
    // Fit function for principal component pursuit
    void fit(double tol = 1E-7, int max_iter = 1000, int iter_print = 100) {
        int iter = 0;
        double err = std::numeric_limits<double>::infinity();
        int size = rows * cols;
        
        // Temporary storage for D - S + mu_inv * Y
        double* d_DSY;
        CUDA_CHECK(cudaMalloc(&d_DSY, size * sizeof(double)));
        
        // Temporary storage for error calculation
        double* d_err;
        CUDA_CHECK(cudaMalloc(&d_err, size * sizeof(double)));
        
        const double alpha = 1.0;
        const double beta = 0.0;
        const double neg_alpha = -1.0;
        
        while (err > tol && iter < max_iter) {
            // Compute D - S + mu_inv * Y
            CUDA_CHECK(cudaMemcpy(d_DSY, d_D, size * sizeof(double), cudaMemcpyDeviceToDevice));
            
            // Subtract S: d_DSY = d_DSY - d_S = d_D - d_S
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &neg_alpha, d_S, 1, d_DSY, 1));
            
            // Add mu_inv * Y: d_DSY = d_DSY + mu_inv * d_Y = d_D - d_S + mu_inv * d_Y
            double mu_inv_val = mu_inv;
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &mu_inv_val, d_Y, 1, d_DSY, 1));
            
            // Step 3: SVD Thresholding to get low-rank component L
            svd_threshold(d_DSY, d_L, mu_inv);
            
            // Step 4: Compute D - L + mu_inv * Y for shrinkage
            CUDA_CHECK(cudaMemcpy(d_DSY, d_D, size * sizeof(double), cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &neg_alpha, d_L, 1, d_DSY, 1));
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &mu_inv_val, d_Y, 1, d_DSY, 1));
            
            // Apply shrinkage to get sparse component S
            shrink(d_DSY, d_S, mu_inv * lmbda);
            
            // Step 5: Update Lagrange multiplier Y
            // First calculate D - L - S into d_err
            CUDA_CHECK(cudaMemcpy(d_err, d_D, size * sizeof(double), cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &neg_alpha, d_L, 1, d_err, 1));
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &neg_alpha, d_S, 1, d_err, 1));
            
            // Update Y: Y = Y + mu * (D - L - S)
            double mu_val = mu;
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, size, &mu_val, d_err, 1, d_Y, 1));
            
            // Compute error
            err = frobenius_norm(d_err);
            iter++;
            
            // Print progress
            if (iter % iter_print == 0 || iter == 1 || err <= tol) {
                std::cout << "Iteration: " << iter << ", Error: " << err << std::endl;
            }
        }
        
        // Clean up temporary buffers
        CUDA_CHECK(cudaFree(d_DSY));
        CUDA_CHECK(cudaFree(d_err));
    }
    
    // Get the sparse matrix (foreground) as a cv::Mat
    cv::Mat get_sparse_matrix() {
        int size = rows * cols;
        std::vector<double> h_S(size);
        CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, size * sizeof(double), cudaMemcpyDeviceToHost));
        
        cv::Mat result(rows, cols, CV_8UC1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double val = h_S[i * cols + j];
                result.at<uchar>(i, j) = static_cast<uchar>(std::min(std::max(val, 0.0), 255.0));
            }
        }
        return result;
    }
    
    // Get the low-rank matrix (background) as a cv::Mat
    cv::Mat get_low_rank_matrix() {
        int size = rows * cols;
        std::vector<double> h_L(size);
        CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, size * sizeof(double), cudaMemcpyDeviceToHost));
        
        cv::Mat result(rows, cols, CV_8UC1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double val = h_L[i * cols + j];
                result.at<uchar>(i, j) = static_cast<uchar>(std::min(std::max(val, 0.0), 255.0));
            }
        }
        return result;
    }
};

int main() {
    // Load an image using OpenCV
    cv::Mat img = cv::imread("lenna(1).png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Create CUDA RPCA object with the image
    CUDA_R_pca rpca(img);

    // Apply RPCA
    rpca.fit();

    // Get the sparse (foreground) and low-rank (background) matrices
    cv::Mat foreground = rpca.get_sparse_matrix();
    cv::Mat background = rpca.get_low_rank_matrix();

    // Show the results
    cv::imshow("Original", img);
    cv::imshow("Foreground (Sparse)", foreground);
    cv::imshow("Background (Low-rank)", background);
    cv::waitKey(0);

    // Save the results
    cv::imwrite("foreground_cuda.png", foreground);
    cv::imwrite("background_cuda.png", background);

    return 0;
}