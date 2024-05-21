import matplotlib.pyplot as plt
import numpy as np

# Define the dimensionalities and datasets
dimensionalities = ['50v', '100v', '150v']
datasets = ['50m', '100m', '150m']

# FastText data
fasttext_times = np.array([
    [374.47, 493.39, 731.35],
    [758.38, 1099.31, 1345.18],
    [1035.41, 1525.53, 2512.98]
])

fasttext_accuracy = np.array([
    [0.3687566776, 0.3901588751, 0.392602404],
    [0.3790136679, 0.400918275, 0.4084761384],
    [0.3788906007, 0.3982426514, 0.4073845768]
])

# GloVe data
glove_times = np.array([
    [313.75, 408.52, 497.73],
    [572.37, 745.66, 1061.24],
    [497.73, 836.89, 1269.77]
])

glove_accuracy = np.array([
    [0.2381339257, 0.2521391664, 0.2517264948],
    [0.2705170819, 0.2866226759, 0.2903481532],
    [0.2752158259, 0.3013723803, 0.3058608385]
])

# Function to perform min-max normalization


# def min_max_normalize(data):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     normalized_data = (data - min_val) / (max_val - min_val)
#     return normalized_data


# # Normalize the data
# fasttext_times_norm = min_max_normalize(fasttext_times)
# fasttext_accuracy_norm = min_max_normalize(fasttext_accuracy)
# glove_times_norm = min_max_normalize(glove_times)
# glove_accuracy_norm = min_max_normalize(glove_accuracy)

# print(fasttext_times_norm)
# print(fasttext_accuracy_norm)

# print(glove_times_norm)
# print(glove_accuracy_norm)


# print(fasttext_times_norm / fasttext_accuracy_norm)

# Plot heatmaps for fastText
plt.figure(figsize=(6, 6))
plt.imshow(fasttext_accuracy / fasttext_times, cmap='plasma',
           aspect='auto', interpolation='nearest')
plt.colorbar(label='FastText (average ρ / time)')
plt.title('FastText (average ρ / time)')
plt.xticks(np.arange(len(dimensionalities)), dimensionalities)
plt.yticks(np.arange(len(datasets)), datasets)
plt.xlabel('Dimensionality')
plt.ylabel('Dataset')
plt.tight_layout()
plt.show()

# print(glove_times / glove_accuracy_norm)

# Plot heatmaps for GloVe
plt.figure(figsize=(6, 6))
plt.imshow(glove_accuracy / glove_times, cmap='plasma',
           aspect='auto', interpolation='nearest')
plt.colorbar(label='GloVe (average ρ / time)')
plt.title('GloVe (average ρ / time)')
plt.xticks(np.arange(len(dimensionalities)), dimensionalities)
plt.yticks(np.arange(len(datasets)), datasets)
plt.xlabel('Dimensionality')
plt.ylabel('Dataset')
plt.tight_layout()
plt.show()


# brightest colour is most sensible model to train
