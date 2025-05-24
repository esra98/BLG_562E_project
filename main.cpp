#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <chrono>

class R_pca {
public:
    Eigen::MatrixXd D;
    Eigen::MatrixXd S;
    Eigen::MatrixXd Y;
    double mu;
    double mu_inv;
    double lmbda;

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

    void fit(double tol = 1E-7, int max_iter = 100, int iter_print = 100) {
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

    std::string filename = "lenna_512x512.png";

    std::vector<int> sizes = {64, 128, 256, 512};

    auto timer_1 = high_resolution_clock::now();
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    auto timer_2 = high_resolution_clock::now();
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << filename << std::endl;
        return -1;
    }

    std::string base_filename = filename.substr(0, filename.find_last_of('.'));

    for (int sz : sizes) {
        std::cout << "\nProcessing size: " << sz << "x" << sz << std::endl;

        auto timer_3 = high_resolution_clock::now();
        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(sz, sz));
        auto timer_4 = high_resolution_clock::now();

        auto timer_5 = high_resolution_clock::now();
        Eigen::MatrixXd D = mat_from_image(img_resized);
        auto timer_6 = high_resolution_clock::now();

        auto timer_7 = high_resolution_clock::now();
        R_pca rpca(D);
        rpca.fit();
        auto timer_8 = high_resolution_clock::now();

        auto timer_9 = high_resolution_clock::now();
        Eigen::MatrixXd S = rpca.get_sparse_matrix();
        Eigen::MatrixXd L = rpca.get_low_rank_matrix();
        cv::Mat foreground = image_from_mat(S);
        cv::Mat background = image_from_mat(L);
        auto timer_10 = high_resolution_clock::now();

        auto timer_11 = high_resolution_clock::now();
        std::string out_fg = base_filename + "_foreground_100_iteration" + std::to_string(sz) + ".png";
        std::string out_bg = base_filename + "_background_100_iteration" + std::to_string(sz) + ".png";
        cv::imwrite(out_fg, foreground);
        cv::imwrite(out_bg, background);
        auto timer_12 = high_resolution_clock::now();

        std::cout << "Load image: " << duration_cast<milliseconds>(timer_2 - timer_1).count() << " ms\n";
        std::cout << "Resize image: " << duration_cast<milliseconds>(timer_4 - timer_3).count() << " ms\n";
        std::cout << "Convert to Eigen matrix: " << duration_cast<milliseconds>(timer_6 - timer_5).count() << " ms\n";
        std::cout << "RPCA fit time: " << duration_cast<milliseconds>(timer_8 - timer_7).count() << " ms\n";
        std::cout << "Create result images: " << duration_cast<milliseconds>(timer_10 - timer_9).count() << " ms\n";
        std::cout << "Save images: " << duration_cast<milliseconds>(timer_12 - timer_11).count() << " ms\n";
        std::cout << "Total time for size " << sz << ": " << duration_cast<milliseconds>(timer_12 - timer_3).count() << " ms\n";
    }

    std::cout << "\nLoad original image: " << duration_cast<milliseconds>(timer_2 - timer_1).count() << " ms\n";
    std::cout << "COMPLETED\n";
    return 0;
}