import matplotlib.pyplot as plt

# 1. DATA (From our Summary Table)
labels = ['1.8V\n(Nominal)', '0.6V\n(Stress Limit)']
areas = [6.89, 111.41] # in um^2
colors = ['#1f77b4', '#d62728'] # Standard Blue vs Red
# 2. PLOTTING
plt.figure(figsize=(7, 6))
# Create Bars
bars = plt.bar(labels, areas, color=colors, width=0.5)
# 3. FORMATTING
plt.title('Impact of Voltage Scaling on Silicon Area', fontsize=14, fontweight='bold')
plt.ylabel('Total Active Area ($\mu m^2$)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, 
             f'{height:.1f} $\mu m^2$', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
mid_x = (bars[0].get_x() + bars[1].get_x()) / 2 + 0.25
plt.annotate('~16x Area Increase\nto maintain current', 
             xy=(1, 111), xytext=(0, 80),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),
             fontsize=11, fontweight='bold', ha='center')
plt.tight_layout()
plt.show()