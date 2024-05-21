import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ["50v*50m", "50v*100m", "50v*150m",
              "100v*50m", "100v*100m", "100v*150m",
              "150v*50m", "150v*100m", "150v*150m"]

fastText_values = [0.3687566776, 0.3790136679, 0.3788906007,
                   0.3901588751, 0.400918275, 0.3982426514,
                   0.392602404, 0.4084761384, 0.4073845768]

GloVe_values = [0.2381339257, 0.2705170819, 0.2752158259,
                0.2521391664, 0.2866226759, 0.3013723803,
                0.2517264948, 0.2903481532, 0.3058608385]

# Width of each bar
bar_width = 0.35

# Position of bars on x-axis
r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]

# Create grouped bar chart
plt.bar(r1, fastText_values, color='#d0e0e3',
        edgecolor='grey', width=bar_width, label='fastText')
plt.bar(r2, GloVe_values, color='#fff2cc',
        edgecolor='grey', width=bar_width, label='GloVe')

# Add labels, title, legend, etc.
plt.xlabel('Trained Models (Dimensionality * Dataset Size)', fontweight='bold')
plt.ylabel("Average Spearman's ρ", fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(categories))],
           categories, rotation=45, ha='right')
plt.title("Average Spearman\'s ρ per model")
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
