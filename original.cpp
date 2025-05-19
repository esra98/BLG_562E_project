#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <algorithm>

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

        // Initialize mu
        if (mu > 0) {
            this->mu = mu;
        } else {
            this->mu = D.size() / (4 * D.norm());
        }

        this->mu_inv = 1.0 / this->mu;

        // Initialize lambda
        if (lmbda > 0) {
            this->lmbda = lmbda;
        } else {
            this->lmbda = 1.0 / std::sqrt(std::max(D.rows(), D.cols()));
        }
    }

    // Frobenius norm
    static double frobenius_norm(const Eigen::MatrixXd& M) {
        return M.norm();
    }

    // Shrinkage operator
    static Eigen::MatrixXd shrink(const Eigen::MatrixXd& M, double tau) {
        return M.unaryExpr([tau](double x) { return std::copysign(std::max(std::abs(x) - tau, 0.0), x); });
    }

    // Singular value thresholding
    Eigen::MatrixXd svd_threshold(const Eigen::MatrixXd& M, double tau) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd singular_values = svd.singularValues();
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();

        // Apply shrinkage to singular values
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
            // Step 3: SVD Thresholding
            Lk = svd_threshold(D - Sk + mu_inv * Yk, mu_inv);

            // Step 4: Shrinkage
            Sk = shrink(D - Lk + mu_inv * Yk, mu_inv * lmbda);

            // Step 5: Update Lagrange multiplier
            Yk = Yk + mu * (D - Lk - Sk);

            // Compute error
            err = frobenius_norm(D - Lk - Sk);
            iter++;

            // Print progress
            if (iter % iter_print == 0 || iter == 1 || err <= tol) {
                std::cout << "Iteration: " << iter << ", Error: " << err << std::endl;
            }
        }

        // Store results
        S = Sk;
        Y = Yk;
    }

    // Get the sparse matrix (foreground)
    Eigen::MatrixXd get_sparse_matrix() {
        return S;
    }

    // Get the low-rank matrix (background)
    Eigen::MatrixXd get_low_rank_matrix() {
        return D - S;
    }
};

// Function to convert an OpenCV image to Eigen::MatrixXd
Eigen::MatrixXd mat_from_image(const cv::Mat& img) {
    Eigen::MatrixXd mat(img.rows, img.cols);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            mat(i, j) = img.at<uchar>(i, j);  // Assuming grayscale image
        }
    }
    return mat;
}

// Function to convert an Eigen::MatrixXd back to an OpenCV image
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
    // Load an image using OpenCV
    cv::Mat img = cv::imread("lenna(1).png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Convert OpenCV image to Eigen matrix
    Eigen::MatrixXd D = mat_from_image(img);

    // Initialize RPCA model
    R_pca rpca(D);

    // Apply RPCA
    rpca.fit();

    // Get the sparse (foreground) matrix
    Eigen::MatrixXd S = rpca.get_sparse_matrix();

    // Convert the sparse matrix back to an image
    cv::Mat foreground = image_from_mat(S);

    // Show the result
    cv::imshow("Foreground (Sparse)", foreground);
    cv::waitKey(0);

    // Save the result
    cv::imwrite("foreground.png", foreground);

    return 0;
}
