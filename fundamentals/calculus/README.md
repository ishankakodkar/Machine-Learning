# Calculus for Machine Learning

Calculus is a cornerstone of machine learning, providing the tools to understand and optimize the functions that power machine learning models.

## Why is Calculus Important for Machine Learning?

*   **Optimization:** The process of training a machine learning model is essentially an optimization problem. We want to find the model parameters (weights and biases) that minimize a loss function. Calculus, specifically differential calculus, provides the methods to do this.
*   **Gradient Descent:** This is the most common optimization algorithm used in machine learning. It works by iteratively moving in the direction of the negative gradient of the loss function to find its minimum. The gradient is a vector of partial derivatives, a core concept from multivariable calculus.
*   **Backpropagation:** In neural networks, backpropagation is the algorithm used to calculate the gradients of the loss function with respect to the network's weights. It is an application of the chain rule from calculus.

## Key Concepts

*   **Derivatives:** Measure the rate of change of a function. In machine learning, they tell us how a small change in a parameter will affect the loss.
*   **Partial Derivatives:** Used for functions with multiple variables. They measure the rate of change with respect to one variable while holding others constant.
*   **Gradients:** A vector of all the partial derivatives of a function. It points in the direction of the steepest ascent of the function.
*   **The Chain Rule:** Used to find the derivative of composite functions. It is the foundation of the backpropagation algorithm.

This directory contains code examples to help you understand these concepts.

## Further Reading

*   [Khan Academy - Calculus](https://www.khanacademy.org/math/calculus-1)
*   [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
*   [MIT OpenCourseWare - Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)
