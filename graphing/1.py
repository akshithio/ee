import matplotlib.pyplot as plt
import numpy as np

# Data
dataset_sizes = ["50m", "100m", "150m"]
dimensionalities = ["50v", "100v", "150v"]
categories = ["User", "System", "Total"]

fasttext_times = np.array([[
    3462.62, 4824.95, 9976.06,
    7244.03, 10660.1, 14492.79,
    9976.06, 14441.55, 21215.46
], [
    26.41, 37.38, 52.64,
    58.14, 78.49, 75.23,
    71.22, 111.64, 112.65
], [
    374.47, 493.39, 731.35,
    758.38, 1099.31, 1345.18,
    1035.41, 1525.53, 2512.98
]])

glove_times = np.array([[
    1903.81, 2800.18, 3608.31,
    3467.3, 4995.51, 6607.51,
    4896.75, 7048.54, 8834.71
],
    [
    44.39, 53.41, 61.2,
    79.5, 94.45, 90.61,
    112.58, 136.64, 150.82
],
    [
    313.75, 408.52, 497.73,
    572.37, 745.66, 1061.24,
    497.73, 836.89, 1269.77
]])

# check difference between each and plot or some shi

# Plotting
for i, category in enumerate(categories):
    plt.figure(figsize=(10, 6))
    plt.plot([f'{dim} * {size}' for dim in dimensionalities for size in dataset_sizes],
             fasttext_times[i, :], marker='o', label=f'fastText - {category}')
    plt.plot([f'{dim} * {size}' for dim in dimensionalities for size in dataset_sizes],
             glove_times[i, :], marker='o', label=f'GloVe - {category}')
    plt.xlabel('Dimensionality * Dataset sizes')
    plt.ylabel('Time (seconds)')
    plt.title(f'Training Time Comparison for {category}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
