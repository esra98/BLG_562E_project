#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#define N_max_iteration 100

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

struct ProfileResult {
    cv::Size img_size;
    long t_load, t_to_device, t_rpca, t_save, t_total;
    int iterations;
    float final_err;
};

ProfileResult run_rpca_for_size(const std::string& img_path, cv::Size target_size, float tol = 1e-7f) {
    using namespace std::chrono;
    ProfileResult result;
    result.img_size = target_size;

    auto t_load1 = high_resolution_clock::now();
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Failed to load image: " + img_path);
    if (img.size() != target_size)
        cv::resize(img, img, target_size, 0, 0, cv::INTER_LINEAR);
    auto t_load2 = high_resolution_clock::now();

    int rows = img.rows, cols = img.cols, size = rows * cols;

    auto t_eigen1 = high_resolution_clock::now();
    std::vector<float> h_D(size);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h_D[j * rows + i] = static_cast<float>(img.at<uchar>(i, j));
    auto t_eigen2 = high_resolution_clock::now();

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

    float norm = 0.0f;
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cublasSnrm2(cublasH, size, d_D, 1, &norm);
    float mu = static_cast<float>(size) / (4 * norm);
    float mu_inv = 1.0f / mu;
    float lambda = 1.0f / std::sqrt(static_cast<float>(std::max(rows, cols)));

    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    auto t_rpca1 = high_resolution_clock::now();
    std::vector<float> h_err(size);
    float err = INFINITY;
    int iter = 0;
    while (err > tol && iter < N_max_iteration) {
        const float alpha1 = 1.0f, beta1 = 0.0f;
        cublasScopy(cublasH, size, d_D, 1, d_tmp, 1);
        const float alpha2 = -1.0f;
        cublasSaxpy(cublasH, size, &alpha2, d_S, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &mu_inv, d_Y, 1, d_tmp, 1);

        cudaMemcpy(h_D.data(), d_tmp, size * sizeof(float), cudaMemcpyDeviceToHost);
        cv::Mat tmp_mat(rows, cols, CV_32FC1, h_D.data());
        cv::Mat U, S, Vt;
        cv::SVD::compute(tmp_mat, S, U, Vt, cv::SVD::MODIFY_A);
        cv::Mat S_thr = cv::max(S - mu_inv, 0);
        cv::Mat Lk = U * cv::Mat::diag(S_thr) * Vt;
        cudaMemcpy(d_L, Lk.ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice);

        cublasScopy(cublasH, size, d_D, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &alpha2, d_L, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &mu_inv, d_Y, 1, d_tmp, 1);

        shrink_on_gpu(d_tmp, mu_inv * lambda, size);
        cudaMemcpy(d_S, d_tmp, size * sizeof(float), cudaMemcpyDeviceToDevice);

        cublasScopy(cublasH, size, d_D, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &alpha2, d_L, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &alpha2, d_S, 1, d_tmp, 1);
        cublasSaxpy(cublasH, size, &mu, d_tmp, 1, d_Y, 1);

        cudaMemcpy(h_err.data(), d_tmp, size * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_err = 0.0f;
        for (float v : h_err) sum_err += v * v;
        err = std::sqrt(sum_err);
        iter++;
    }
    auto t_rpca2 = high_resolution_clock::now();

    auto t_res1 = high_resolution_clock::now();
    std::vector<float> h_S(size), h_L(size);
    cudaMemcpy(h_S.data(), d_S, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_L.data(), d_L, size * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat foreground(rows, cols, CV_32FC1, h_S.data());
    cv::Mat background(rows, cols, CV_32FC1, h_L.data());
    cv::Mat fg8, bg8;
    foreground.convertTo(fg8, CV_8UC1, 1, 0);
    background.convertTo(bg8, CV_8UC1, 1, 0);

    std::string size_str = std::to_string(rows) + "x" + std::to_string(cols);
    cv::imwrite("gpu_cusolver_foreground_" + std::to_string(iter) + "_" + size_str + ".png", fg8);
    cv::imwrite("gpu_cusolver_background_" + std::to_string(iter) + "_" + size_str + ".png", bg8);
    auto t_res2 = high_resolution_clock::now();

    result.t_load      = duration_cast<milliseconds>(t_load2 - t_load1).count();
    result.t_to_device = duration_cast<milliseconds>(t_eigen2 - t_eigen1).count();
    result.t_rpca      = duration_cast<milliseconds>(t_rpca2 - t_rpca1).count();
    result.t_save      = duration_cast<milliseconds>(t_res2 - t_res1).count();
    result.t_total     = duration_cast<milliseconds>(t_res2 - t_load1).count();

    result.iterations  = iter;
    result.final_err   = err;

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
    cudaFree(d_D); cudaFree(d_S); cudaFree(d_Y); cudaFree(d_L); cudaFree(d_tmp);
    return result;
}

int main() {
    std::string image_path = "lenna_512x512.png"; 

    std::vector<cv::Size> test_sizes = {
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
    };

    std::vector<ProfileResult> results;
    for (const auto& size : test_sizes) {
        try {
            std::cout << "\nProcessing size: " << size.width << "x" << size.height << std::endl;
            ProfileResult r = run_rpca_for_size(image_path, size);
            std::cout << "  Load: "      << r.t_load << " ms\n";
            std::cout << "  To device: " << r.t_to_device << " ms\n";
            std::cout << "  RPCA: "      << r.t_rpca << " ms\n";
            std::cout << "  Save: "      << r.t_save << " ms\n";
            std::cout << "  Total: "      << r.t_total << " ms\n";
            std::cout << "  Iter: "      << r.iterations << "\n";
            std::cout << "  Final err: " << r.final_err << "\n";
            results.push_back(r);
        } catch (const std::exception& ex) {
            std::cerr << "Error at size " << size.width << "x" << size.height << ": " << ex.what() << std::endl;
        }
    }

    std::cout << "\n=== Timings Summary (ms) ===\n";
    std::cout << "Size\t\tLoad\tToDev\tRPCA\tSave\tIter\tFinalErr\n";
    for (const auto& r : results) {
        std::cout << r.img_size.width << "x" << r.img_size.height << "\t"
                  << r.t_load << "\t"
                  << r.t_to_device << "\t"
                  << r.t_rpca << "\t"
                  << r.t_save << "\t"
                  << r.t_total << "\t"
                  << r.iterations << "\t"
                  << r.final_err << "\n";
    }
    std::cout << "DONE.\n";
    return 0;
}