#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>

// Use Eigen types for matrices and vectors
using Eigen::MatrixXd;
using Eigen::VectorXd;

// A simple class to read a CSV file into Eigen Matrices/Vectors
class CSVReader {
public:
    static bool read(const std::string& path, MatrixXd& X, VectorXd& y) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            return false;
        }

        std::vector<double> x_data, y_data;
        std::string line;
        std::getline(file, line); // Skip header

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value_str;
            
            // Skip 'Position' column
            std::getline(ss, value_str, ',');
            
            // Read 'Level'
            std::getline(ss, value_str, ',');
            x_data.push_back(std::stod(value_str));

            // Read 'Salary'
            std::getline(ss, value_str, ',');
            y_data.push_back(std::stod(value_str));
        }

        X = Eigen::Map<VectorXd>(x_data.data(), x_data.size());
        y = Eigen::Map<VectorXd>(y_data.data(), y_data.size());

        return true;
    }
};

// Function to create polynomial features
MatrixXd create_polynomial_features(const MatrixXd& X, int degree) {
    MatrixXd X_poly(X.rows(), degree);
    for (int i = 0; i < degree; ++i) {
        X_poly.col(i) = X.array().pow(i + 1);
    }
    return X_poly;
}

class PolynomialRegression {
public:
    PolynomialRegression() {}

    void fit(const MatrixXd& X, const VectorXd& y) {
        MatrixXd X_b(X.rows(), X.cols() + 1);
        X_b << MatrixXd::Ones(X.rows(), 1), X;
        theta = (X_b.transpose() * X_b).colPivHouseholderQr().solve(X_b.transpose() * y);
    }

    double predict(const VectorXd& x) {
        VectorXd x_b(x.size() + 1);
        x_b << 1, x;
        return x_b.transpose() * theta;
    }

    const VectorXd& get_parameters() const {
        return theta;
    }

private:
    VectorXd theta;
};

int main() {
    // 1. Load Data
    MatrixXd X_raw;
    VectorXd y;
    if (!CSVReader::read("../position_salaries.csv", X_raw, y)) {
        return 1;
    }

    // 2. Create Polynomial Features
    int degree = 4;
    MatrixXd X_poly = create_polynomial_features(X_raw, degree);

    // 3. Create and train the model
    PolynomialRegression model;
    model.fit(X_poly, y);

    // 4. Print the learned parameters
    VectorXd params = model.get_parameters();
    std::cout << "--- C++ Polynomial Regression Model (Degree " << degree << ") ---" << std::endl;
    std::cout << "Model Parameters (theta):" << std::endl;
    std::cout << "  - Bias (theta_0): " << params(0) << std::endl;
    for (int i = 1; i < params.size(); ++i) {
        std::cout << "  - Weight (theta_" << i << "): " << params(i) << std::endl;
    }

    // 5. Make a prediction
    double level_to_predict = 6.5;
    VectorXd new_level_raw(1);
    new_level_raw << level_to_predict;
    
    // Create polynomial features for the new data point
    VectorXd new_level_poly = create_polynomial_features(new_level_raw, degree).row(0);
    
    double predicted_salary = model.predict(new_level_poly);

    std::cout << "\n--- Prediction ---" << std::endl;
    std::cout << "Predicted salary for level " << level_to_predict << ": " << predicted_salary << std::endl;

    return 0;
}
