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
// This is a basic implementation for demonstration purposes.
class CSVReader {
public:
    // Reads a CSV file and separates it into features (X) and target (y)
    static bool read(const std::string& path, MatrixXd& X, VectorXd& y) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            return false;
        }

        std::vector<double> x_data, y_data;
        std::string line;
        // Skip header
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value_str;
            
            // Read YearsExperience
            std::getline(ss, value_str, ',');
            x_data.push_back(std::stod(value_str));

            // Read Salary
            std::getline(ss, value_str, ',');
            y_data.push_back(std::stod(value_str));
        }

        // Convert std::vector to Eigen types
        X = Eigen::Map<VectorXd>(x_data.data(), x_data.size());
        y = Eigen::Map<VectorXd>(y_data.data(), y_data.size());

        return true;
    }
};

class LinearRegression {
public:
    LinearRegression() {}

    // Fit the model using the Normal Equation: theta = (X^T * X)^-1 * X^T * y
    void fit(const MatrixXd& X, const VectorXd& y) {
        // Add a bias column of ones to the feature matrix X
        MatrixXd X_b(X.rows(), X.cols() + 1);
        X_b << MatrixXd::Ones(X.rows(), 1), X;

        // The Normal Equation calculation
        // Using Eigen's robust and stable solvers is better than direct inversion
        // colPivHouseholderQr is a good choice for this type of problem.
        theta = (X_b.transpose() * X_b).colPivHouseholderQr().solve(X_b.transpose() * y);
    }

    // Predict a single value
    double predict(const VectorXd& x) {
        // Add bias term to the input vector
        VectorXd x_b(x.size() + 1);
        x_b << 1, x;
        
        return x_b.transpose() * theta;
    }

    // Get the learned parameters (theta)
    const VectorXd& get_parameters() const {
        return theta;
    }

private:
    VectorXd theta; // This vector holds the bias (theta_0) and weights (theta_1, ...)
};

int main() {
    // 1. Load Data
    MatrixXd X;
    VectorXd y;
    if (!CSVReader::read("../salary_data.csv", X, y)) {
        return 1; // Exit if file loading fails
    }

    // 2. Create and train the model
    LinearRegression model;
    model.fit(X, y);

    // 3. Print the learned parameters
    VectorXd params = model.get_parameters();
    double bias = params(0);
    double weight = params(1);
    std::cout << "--- C++ Linear Regression Model ---" << std::endl;
    std::cout << "Model Parameters (theta):" << std::endl;
    std::cout << "  - Bias (Intercept): " << bias << std::endl;
    std::cout << "  - Weight (Slope):   " << weight << std::endl;
    std::cout << "\nEquation: Salary = " << weight << " * YearsExperience + " << bias << std::endl;

    // 4. Make a prediction
    VectorXd new_experience(1);
    new_experience << 5.0; // Predict salary for 5 years of experience
    double predicted_salary = model.predict(new_experience);

    std::cout << "\n--- Prediction ---" << std::endl;
    std::cout << "Predicted salary for " << new_experience(0) << " years of experience: " << predicted_salary << std::endl;

    return 0;
}
