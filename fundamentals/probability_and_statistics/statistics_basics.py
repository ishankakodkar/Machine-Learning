import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Descriptive Statistics ---
print("--- Descriptive Statistics ---")

# Sample data
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(f"Data: {data}")

# Mean
mean = np.mean(data)
print(f"Mean: {mean}")

# Median
median = np.median(data)
print(f"Median: {median}")

# Mode
mode = stats.mode(data)
# The stats.mode function returns a ModeResult object with the mode and its count.
print(f"Mode: {mode.mode[0]} (appears {mode.count[0]} times)")

# Standard Deviation
std_dev = np.std(data)
print(f"Standard Deviation: {std_dev:.2f}")

# Variance
variance = np.var(data)
print(f"Variance: {variance:.2f}")


# --- Probability Distribution (Normal Distribution) ---
print("\n--- Normal Distribution ---")

# Generate 1000 random numbers from a normal distribution
# with a mean of 0 and a standard deviation of 1
mu, sigma = 0, 1
normal_data = np.random.normal(mu, sigma, 1000)

# Create a histogram to visualize the distribution
count, bins, ignored = plt.hist(normal_data, 30, density=True)

# Plot the probability density function (PDF)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Save the plot to a file
plt.savefig('normal_distribution.png')
print("Generated a plot of a normal distribution: 'normal_distribution.png'")

# Show the plot
plt.show()
