import pandas as pd

# Load the CSV file
df = pd.read_csv('data.csv')

# Define the mapping from text to numerical values
likert_scale = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Apply the mapping to the DataFrame, converting relevant columns
for col in df.columns:
    if col.startswith('Structure:'):
        df[col] = df[col].map(likert_scale)

# Initialize a list to hold the results
results = []

# Calculate the average values for each structure
for i in range(1, 38, 4):
    cohesiveness = df.iloc[:, i].mean()
    complexity = df.iloc[:, i+1].mean()
    aesthetics = df.iloc[:, i+2].mean()
    results.append([cohesiveness, complexity, aesthetics])

structure_names = [
    "E-DCGAN",
    "Dataset",
    "ERA-DCGANx2",
    "RA-DCGAN",
    "ERA-DCGANx4",
    "RA-DCGAN",
    "E-DCGAN",
    "Dataset",
    "ERA-DCGANx2",
    "ERA-DCGANx4"
]

# Convert the results to a DataFrame for easier visualization
results_df = pd.DataFrame(results, columns=['Cohesiveness', 'Complexity', 'Aesthetics'], index=structure_names)

# Combine the averages of identical structure names
combined_results_df = results_df.groupby(results_df.index).mean()

# Add a column for the average of the three ratings to sort by
combined_results_df['Average'] = combined_results_df.mean(axis=1)

# Sort the DataFrame by the average rating
combined_results_df = combined_results_df.sort_values(by='Average', ascending=False)

# Drop the 'Average' column as it was only for sorting purposes
combined_results_df = combined_results_df.drop(columns=['Average'])

# Print the results
print(combined_results_df)

# Save the results to a CSV file
combined_results_df.to_csv('combined_structure_averages.csv', index=True)
