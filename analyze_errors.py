import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Read Data
# ==========================================
file_path = 'all_contact_errors.csv'
print(f"Reading data: {file_path} ...")
df = pd.read_csv(file_path)

# ==========================================
# 2. Terminal Data Analysis Report
# ==========================================
print("\n" + "="*30)
print("🎯 Data Analysis Summary")
print("="*30)

# Calculate overall average error
total_mean = df['Error_ms'].mean()
print(f"Overall average error: {total_mean:.2f} ms")

# Calculate average and max errors by foot
foot_stats = df.groupby('Foot')['Error_ms'].agg(['mean', 'max', 'count']).round(2)
foot_stats.columns = ['Avg Error(ms)', 'Max Error(ms)', 'Total Steps']
print("\n[Left/Right Foot Comparison]:")
print(foot_stats)

# ==========================================
# 3. Plot high-quality Boxplot
# ==========================================
print("\nGenerating distribution chart...")

# Set Seaborn plot style (with grid for academic look)
sns.set_theme(style="whitegrid")

# Create a relatively wide figure
plt.figure(figsize=(14, 7))

# Plot grouped boxplot
# x-axis: Subject
# y-axis: Error in ms (Error_ms)
# hue: Differentiate left/right foot by color (Foot)
sns.boxplot(
    data=df, 
    x='Subject', 
    y='Error_ms', 
    hue='Foot', 
    palette='Set2', # Use soft academic color palette
    linewidth=1.2,
    fliersize=4     # Size of outlier circles
)

# Add title and axis labels
plt.title('Distribution of Contact Time Absolute Errors by Subject and Foot', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Subject', fontsize=14)
plt.ylabel('Absolute Error (ms)', fontsize=14)

# Slightly rotate X-axis labels to prevent overlapping
plt.xticks(rotation=30)

# Adjust legend position
plt.legend(title='Foot', title_fontsize='12', fontsize='11', loc='upper right')

# Automatically adjust layout to prevent text clipping
plt.tight_layout()

# ==========================================
# 4. Save Chart
# ==========================================
# Save as high-resolution PNG (dpi=300 meets publication standards)
output_img = 'contact_errors_distribution.png'
plt.savefig(output_img, dpi=300)
print(f"✅ Chart successfully saved as high-res image: {output_img}")

# Note: Do not use plt.show() if running on a headless Linux server
# If running locally on Mac/Windows, uncomment the line below to view the plot:
# plt.show()