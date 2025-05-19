#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <chrono> // For profiling

// Function for RPCA (from the previous implementation)
class R_pca {
public:
    Eigen::MatrixXd D;  // Data matrix (image)
    Eigen::MatrixXd S;  // Sparse component (foreground)
    Eigen::MatrixXd Y;  // Lagrange multiplier
    double mu;
    double mu_inv;
    double lmbda;

    // Constructor
    R_pca(const Eigen::MatrixXd& D, double mu = -1, double lmbda = -1) 
        : D(D), S(D.rows(), D.cols()), Y(D.rows(), D.cols()) {
        if (mu > 0) {
            this->mu = mu;
        } else {
            this->mu = D.size() / (4 * D.norm());
        }
        this->mu_inv = 1.0 / this->mu;
        if (lmbda > 0) {
            this->lmbda = lmbda;
        } else {
            this->lmbda = 1.0 / std::sqrt(std::max(D.rows(), D.cols()));
        }
    }

    static double frobenius_norm(const Eigen::MatrixXd& M) {
        return M.norm();
    }

    static Eigen::MatrixXd shrink(const Eigen::MatrixXd& M, double tau) {
        return M.unaryExpr([tau](double x) { return std::copysign(std::max(std::abs(x) - tau, 0.0), x); });
    }

    Eigen::MatrixXd svd_threshold(const Eigen::MatrixXd& M, double tau) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd singular_values = svd.singularValues();
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();

        for (int i = 0; i < singular_values.size(); ++i) {
            singular_values[i] = std::max(singular_values[i] - tau, 0.0);
        }
        return U * singular_values.asDiagonal() * V.transpose();
    }

    // Fit function for principal component pursuit
    void fit(double tol = 1E-7, int max_iter = 1000, int iter_print = 100) {
        int iter = 0;
        double err = std::numeric_limits<double>::infinity();
        Eigen::MatrixXd Sk = S;
        Eigen::MatrixXd Yk = Y;
        Eigen::MatrixXd Lk = Eigen::MatrixXd::Zero(D.rows(), D.cols());

        while (err > tol && iter < max_iter) {
            Lk = svd_threshold(D - Sk + mu_inv * Yk, mu_inv);
            Sk = shrink(D - Lk + mu_inv * Yk, mu_inv * lmbda);
            Yk = Yk + mu * (D - Lk - Sk);
            err = frobenius_norm(D - Lk - Sk);
            iter++;
            if (iter % iter_print == 0 || iter == 1 || err <= tol) {
                std::cout << "Iteration: " << iter << ", Error: " << err << std::endl;
            }
        }
        S = Sk;
        Y = Yk;
    }

    Eigen::MatrixXd get_sparse_matrix() {
        return S;
    }

    Eigen::MatrixXd get_low_rank_matrix() {
        return D - S;
    }
};

Eigen::MatrixXd mat_from_image(const cv::Mat& img) {
    Eigen::MatrixXd mat(img.rows, img.cols);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            mat(i, j) = img.at<uchar>(i, j);  // Assuming grayscale image
        }
    }
    return mat;
}

cv::Mat image_from_mat(const Eigen::MatrixXd& mat) {
    cv::Mat img(mat.rows(), mat.cols(), CV_8UC1);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            img.at<uchar>(i, j) = static_cast<uchar>(std::min(std::max(mat(i, j), 0.0), 255.0));
        }
    }
    return img;
}

int main() {
    using namespace std::chrono;
    std::cout << "Profiling steps...\n";

    // 1. Load the image
    auto t_load1 = high_resolution_clock::now();
    cv::Mat img = cv::imread("lenna(1).png", cv::IMREAD_GRAYSCALE);
    auto t_load2 = high_resolution_clock::now();
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // 2. Convert to Eigen matrix
    auto t_eigen1 = high_resolution_clock::now();
    Eigen::MatrixXd D = mat_from_image(img);
    auto t_eigen2 = high_resolution_clock::now();

    // 3. Initialize and run RPCA
    auto t_rpca1 = high_resolution_clock::now();
    R_pca rpca(D);
    rpca.fit();
    auto t_rpca2 = high_resolution_clock::now();

    // 4. Get results and convert back to cv::Mat
    auto t_res1 = high_resolution_clock::now();
    Eigen::MatrixXd S = rpca.get_sparse_matrix();
    Eigen::MatrixXd L = rpca.get_low_rank_matrix();
    cv::Mat foreground = image_from_mat(S);
    cv::Mat background = image_from_mat(L);
    auto t_res2 = high_resolution_clock::now();

    // 5. Save results
    auto t_save1 = high_resolution_clock::now();
    cv::imwrite("cpu_results_foreground.png", foreground);
    cv::imwrite("cpu_results_background.png", background);
    auto t_save2 = high_resolution_clock::now();

    // Profiling output
    std::cout << "Profiling (milliseconds):\n";
    std::cout << "Load image: " << duration_cast<milliseconds>(t_load2 - t_load1).count() << " ms\n";
    std::cout << "Convert to Eigen: " << duration_cast<milliseconds>(t_eigen2 - t_eigen1).count() << " ms\n";
    std::cout << "RPCA fit: " << duration_cast<milliseconds>(t_rpca2 - t_rpca1).count() << " ms\n";
    std::cout << "Convert/Prepare results: " << duration_cast<milliseconds>(t_res2 - t_res1).count() << " ms\n";
    std::cout << "Save images: " << duration_cast<milliseconds>(t_save2 - t_save1).count() << " ms\n";
    std::cout << "DONE.\n";

    return 0;
}