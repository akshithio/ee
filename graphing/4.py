import matplotlib.pyplot as plt
import numpy as np

# FastText data
fasttext_times = [
    [374.47, 758.38, 1035.41],
    [493.39, 1099.31, 1525.53],
    [731.35, 1345.18, 2512.98]
]
fasttext_rho = [
    [0.3687566776, 0.3790136679, 0.3788906007],
    [0.3901588751, 0.400918275, 0.3982426514],
    [0.392602404, 0.4084761384, 0.4073845768]
]

# GloVe data
glove_times = [
    [313.75, 572.37, 497.73],
    [408.52, 745.66, 836.89],
    [497.73, 1061.24, 1269.77]
]
glove_rho = [
    [0.2381339257, 0.2705170819, 0.2752158259],
    [0.2521391664, 0.2866226759, 0.3013723803],
    [0.2517264948, 0.2903481532, 0.3058608385]
]





# Plot scatter plot
plt.figure(figsize=(10, 6))

# Flatten the lists for scatter plot
flat_fasttext_times = [item for sublist in fasttext_times for item in sublist]
flat_fasttext_rho = [item for sublist in fasttext_rho for item in sublist]
flat_glove_times = [item for sublist in glove_times for item in sublist]
flat_glove_rho = [item for sublist in glove_rho for item in sublist]

# Plot fastText data
plt.scatter(flat_fasttext_times, flat_fasttext_rho,
            color='#00688B', label='fastText')

# Plot GloVe data
plt.scatter(flat_glove_times, flat_glove_rho, color='#DAA520', label='GloVe')

# Calculate lines of best fit
fasttext_fit = np.polyfit(flat_fasttext_times, flat_fasttext_rho, 1)
glove_fit = np.polyfit(flat_glove_times, flat_glove_rho, 1)

# Plot lines of best fit with solid lines
plt.plot(np.array(flat_fasttext_times), np.polyval(fasttext_fit,
         flat_fasttext_times), color='blue', linestyle='-', label='fastText Fit')
plt.plot(np.array(flat_glove_times), np.polyval(
    glove_fit, flat_glove_times), color='orange', linestyle='-', label='GloVe Fit')

plt.xlabel('Processing Time (seconds)')
plt.ylabel('Accuracy (Spearman\'s ρ)')
plt.title('Relationship between Processing Time and Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# for x% increase in time spent how much % increase in performance. GloVe benefits more from more data since slope is steeper
# there are however moments when the same amount of time spent yields better for fastText
# for above might be worth to check the model configurability? maybe being done by model-6?
