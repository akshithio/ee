import matplotlib.pyplot as plt
import numpy as np

# Define the arrays
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

# Define the axes labels
dimensionalities = ['50v', '100v', '150v']
datasets = ['50m', '100m', '150m']

# Plot heatmaps
plt.figure(figsize=(6, 4))

# FastText Times Heatmap
plt.imshow(fasttext_times, cmap='plasma', aspect='auto',
           interpolation='nearest')
plt.colorbar(label='Time (seconds)')
plt.title('FastText Times')
plt.xticks(np.arange(len(dimensionalities)), dimensionalities)
plt.yticks(np.arange(len(datasets)), datasets)
plt.xlabel('Dimensionality')
plt.ylabel('Dataset')
plt.show()

plt.figure(figsize=(6, 4))
# FastText Accuracy Heatmap
plt.imshow(fasttext_accuracy, cmap='plasma', aspect='auto',
           interpolation='nearest')
plt.colorbar(label='Accuracy')
plt.title('FastText Accuracy')
plt.xticks(np.arange(len(dimensionalities)), dimensionalities)
plt.yticks(np.arange(len(datasets)), datasets)
plt.xlabel('Dimensionality')
plt.ylabel('Dataset')
plt.show()

plt.figure(figsize=(6, 4))
# GloVe Times Heatmap
plt.imshow(glove_times, cmap='plasma', aspect='auto',
           interpolation='nearest')
plt.colorbar(label='Time (seconds)')
plt.title('GloVe Times')
plt.xticks(np.arange(len(dimensionalities)), dimensionalities)
plt.yticks(np.arange(len(datasets)), datasets)
plt.xlabel('Dimensionality')
plt.ylabel('Dataset')
plt.show()

plt.figure(figsize=(6, 4))
# GloVe Accuracy Heatmap
plt.imshow(glove_accuracy, cmap='plasma', aspect='auto',
           interpolation='nearest')
plt.colorbar(label='Accuracy')
plt.title('GloVe Accuracy')
plt.xticks(np.arange(len(dimensionalities)), dimensionalities)
plt.yticks(np.arange(len(datasets)), datasets)
plt.xlabel('Dimensionality')
plt.ylabel('Dataset')
plt.show()


# vmin, vmax can be different cuz we're just tryna see what affects the models more: vec or m.
