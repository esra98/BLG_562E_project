#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

// CUDA kernel for shrinkage (soft-thresholding)
__global__ void shrink_kernel(float* mat, float tau, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = mat[idx];
        float mag = fabsf(x) - tau;
        mat[idx] = (mag > 0 ? copysignf(mag, x) : 0.0f);
    }
}

void shrink_on_gpu(float* d_mat, float tau, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    shrink_kernel<<<grid, block>>>(d_mat, tau, n);
    cudaDeviceSynchronize();
}

int main() {
    using namespace std::chrono;
    std::cout << "Starting cuBLAS/cuSOLVER RPCA demo\n";

    // 1. Load grayscale image
    auto t_load1 = high_resolution_clock::now();
    cv::Mat img = cv::imread("lenna(1).png", cv::IMREAD_GRAYSCALE);
    auto t_load2 = high_resolution_clock::now();
    if (img.empty()) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }
    int rows = img.rows, cols = img.cols, size = rows * cols;

    // 2. Copy image to host float array (column major)
    auto t_eigen1 = high_resolution_clock::now();
    std::vector<float> h_D(size);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h_D[j * rows + i] = static_cast<float>(img.at<uchar>(i, j));
    auto t_eigen2 = high_resolution_clock::now();

    // 3. Allocate device memory
    float *d_D, *d_S, *d_Y, *d_L, *d_tmp;
    cudaMalloc(&d_D, size * sizeof(float));
    cudaMalloc(&d_S, size * sizeof(float));
    cudaMalloc(&d_Y, size * sizeof(float));
    cudaMalloc(&d_L, size * sizeof(float));
    cudaMalloc(&d_tmp, size * sizeof(float));
    cudaMemcpy(d_D, h_D.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_S, 0, size * sizeof(float));
    cudaMemset(d_Y, 0, size * sizeof(float));
    cudaMemset(d_L, 0, size * sizeof(float));

    // 4. Set parameters
    float norm = 0.0f;
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cublasSnrm2(cublasH, size, d_D, 1, &norm);
    float mu = static_cast<float>(size) / (4 * norm);
    float mu_inv = 1.0f / mu;
    float lambda = 1.0f / std::sqrt(static_cast<float>(std::max(rows, cols)));
    float tol = 1E-7f;
    int max_iter = 1000, iter = 0;
    float err = INFINITY;

    // cuSOLVER handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // 5. RPCA iteration (timed)
    auto t_rpca1 = high_resolution_clock::now();
    std::vector<float> h_err(size);

    while (err > tol && iter < max_iter) {
        // D - S + mu_inv*Y -> d_tmp
        const float alpha1 = 1.0f, beta1 = 0.0f;
        cublasScopy(cublasH, size, d_D, 1, d_tmp, 1);
        const float alpha2 = -1.0f;
        cublasSaxpy(cublasH, size, &alpha2, d_S, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &mu_inv, d_Y, 1, d_tmp, 1);

        // SVD and thresholding (on CPU for simplicity in this skeleton)
        cudaMemcpy(h_D.data(), d_tmp, size * sizeof(float), cudaMemcpyDeviceToHost);
        cv::Mat tmp_mat(rows, cols, CV_32FC1, h_D.data());
        cv::Mat U, S, Vt;
        cv::SVD::compute(tmp_mat, S, U, Vt, cv::SVD::MODIFY_A);
        cv::Mat S_thr = cv::max(S - mu_inv, 0);
        cv::Mat Lk = U * cv::Mat::diag(S_thr) * Vt;
        cudaMemcpy(d_L, Lk.ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice);

        // D - L + mu_inv*Y -> d_tmp
        cublasScopy(cublasH, size, d_D, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &alpha2, d_L, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &mu_inv, d_Y, 1, d_tmp, 1);

        // Shrinkage (GPU)
        shrink_on_gpu(d_tmp, mu_inv * lambda, size);
        cudaMemcpy(d_S, d_tmp, size * sizeof(float), cudaMemcpyDeviceToDevice);

        // Y = Y + mu*(D-L-S)
        cublasScopy(cublasH, size, d_D, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &alpha2, d_L, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &alpha2, d_S, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &mu, d_tmp, 1, d_Y, 1);

        // Compute error on CPU
        cudaMemcpy(h_err.data(), d_tmp, size * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_err = 0.0f;
        for (float v : h_err) sum_err += v * v;
        err = std::sqrt(sum_err);
        iter++;
        if (iter == 1 || iter % 100 == 0 || err < tol)
            std::cout << "Iter: " << iter << ", Error: " << err << std::endl;
    }
    auto t_rpca2 = high_resolution_clock::now();

    // 6. Copy results back and save
    auto t_res1 = high_resolution_clock::now();
    std::vector<float> h_S(size), h_L(size);
    cudaMemcpy(h_S.data(), d_S, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L.data(), d_L, size * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat foreground(rows, cols, CV_32FC1, h_S.data());
    cv::Mat background(rows, cols, CV_32FC1, h_L.data());
    cv::Mat fg8, bg8;
    foreground.convertTo(fg8, CV_8UC1, 1, 0);
    background.convertTo(bg8, CV_8UC1, 1, 0);

    cv::imwrite("gpu_cusolver_foreground.png", fg8);
    cv::imwrite("gpu_cusolver_background.png", bg8);
    auto t_res2 = high_resolution_clock::now();

    // 7. Profiling
    std::cout << "Profiling (milliseconds):\n";
    std::cout << "Load image: " << duration_cast<milliseconds>(t_load2 - t_load1).count() << " ms\n";
    std::cout << "To device: " << duration_cast<milliseconds>(t_eigen2 - t_eigen1).count() << " ms\n";
    std::cout << "RPCA fit: " << duration_cast<milliseconds>(t_rpca2 - t_rpca1).count() << " ms\n";
    std::cout << "Save images: " << duration_cast<milliseconds>(t_res2 - t_res1).count() << " ms\n";
    std::cout << "DONE.\n";

    // Free
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
    cudaFree(d_D); cudaFree(d_S); cudaFree(d_Y); cudaFree(d_L); cudaFree(d_tmp);

    return 0;
}