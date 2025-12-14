import matplotlib.pyplot as plt

# 1. DATA (Extracted from our results)
# Format: [Power (uW), Gain (dB)]
data_06V = [[5.17, 36.75], [5.41, 36.77], [5.61, 36.79], [6.07, 36.80], [6.23, 36.82], [6.60, 36.84]]
data_08V = [[1.4, 37.57], [1.9, 37.7], [2.6, 37.85], [3.5, 38.0], [5.2, 38.25], [7.3, 38.45], [10.5, 38.65], [14.0, 38.8]]
data_10V = [[1.8, 37.2], [2.5, 38.0], [3.5, 38.5], [5.2, 39.1], [8.7, 40.2], [13.0, 40.8], [21.5, 41.2], [34.5, 41.5]]
data_12V = [[2.8, 38.4], [4.5, 39.3], [6.2, 39.8], [10.0, 40.7], [15.0, 42.1], [25.0, 42.8], [35.0, 43.1], [48.0, 43.2]]
data_14V = [[2.1, 36.6], [3.5, 38.6], [5.0, 39.3], [9.5, 40.4], [15.0, 42.5], [25.0, 43.6], [40.0, 44.1], [60.0, 44.2]]
data_16V = [[2.5, 35.9], [4.0, 38.8], [7.0, 40.0], [12.0, 41.4], [20.0, 43.7], [35.0, 44.7], [50.0, 45.1], [70.0, 45.3]]
data_18V = [[5.3, 38.5], [8.0, 39.5], [12.0, 40.8], [18.0, 42.1], [26.0, 44.0], [35.0, 45.1], [53.0, 45.6], [90.0, 45.8], [98.0, 46.0]]

# 2. PLOTTING
plt.figure(figsize=(10, 6))
# Plot each voltage line (1.8V first to appear at top of legend)
# Using standard markers 'o' and solid lines
plt.plot([x[0] for x in data_18V], [x[1] for x in data_18V], marker='o', label='Vdd = 1.8V')
plt.plot([x[0] for x in data_16V], [x[1] for x in data_16V], marker='o', label='Vdd = 1.6V')
plt.plot([x[0] for x in data_14V], [x[1] for x in data_14V], marker='o', label='Vdd = 1.4V')
plt.plot([x[0] for x in data_12V], [x[1] for x in data_12V], marker='o', label='Vdd = 1.2V')
plt.plot([x[0] for x in data_10V], [x[1] for x in data_10V], marker='o', label='Vdd = 1.0V')
plt.plot([x[0] for x in data_08V], [x[1] for x in data_08V], marker='o', label='Vdd = 0.8V')
plt.plot([x[0] for x in data_06V], [x[1] for x in data_06V], marker='o', label='Vdd = 0.6V', color='red', linewidth=2)

# 3. LABELS
plt.title('Combined Pareto Fronts: Gain vs Power')
plt.xlabel('Power Consumption (uW)')
plt.ylabel('DC Gain (dB)')
plt.grid(True)
plt.legend()
# Simple arrow for the limit
plt.annotate('Technology Limit (0.6V)', xy=(6.6, 36.8), xytext=(20, 36.5),arrowprops=dict(facecolor='black', shrink=0.05))
# Simple arrow for the nominal
plt.annotate('Nominal (1.8V)', xy=(60, 45.7), xytext=(60, 43), arrowprops=dict(facecolor='black', shrink=0.05))
plt.tight_layout()
plt.show()