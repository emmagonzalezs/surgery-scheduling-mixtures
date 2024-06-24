import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from WaitingList import mixture_distribution

# Assuming the functions and data have already been loaded as per your provided code

# Select a specific case (prefix)
prefix = '5-454'
distr = 'normal'

# Generate parameters for the selected case
parameters_surgeries, mixture_params = mixture_distribution(distr, prefix)

# Extract means and standard deviations of individual distributions
means = [params[0] for params in parameters_surgeries.values()]
sds = [params[1] for params in parameters_surgeries.values()]

# Extract mean and standard deviation of the mixture distribution
mixture_mean = mixture_params[0]
mixture_sd = mixture_params[1]

# Generate x values for plotting
x = np.linspace(min(means) - 3 * max(sds), max(means) + 3 * max(sds), 1000)

# Plot individual normal distributions
plt.figure(figsize=(10, 6))
for mean, sd in zip(means, sds):
    y = norm.pdf(x, mean, sd)
    plt.plot(x, y, label=f'Individual Normal (μ={mean:.2f}, σ={sd:.2f})')

# Plot mixture distribution
y_mixture = norm.pdf(x, mixture_mean, mixture_sd)
plt.plot(x, y_mixture, label=f'Mixture Normal (μ={mixture_mean:.2f}, σ={mixture_sd:.2f})', linewidth=2, linestyle='--')

# Add labels and legend
plt.title(f'Normal Distributions for {prefix} Surgeries')
plt.xlabel('Duration')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
