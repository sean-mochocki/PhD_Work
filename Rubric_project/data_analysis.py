import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy import stats

knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Data/Knowledge_Nodes.txt"
learner_profile = "/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Data/Learner_Profile_8_Jan_2025.xlsx"

profile_database = pd.read_excel(learner_profile)

# Convert string representation of lists to actual lists using a lambda function
profile_database['cognitive_levels'] = profile_database['cognitive_levels'].apply(lambda x: ast.literal_eval(x))
profile_database['goals'] = profile_database['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])
#print(profile_database['goals'])

#print(profile_database['cognitive_levels'].iloc[0])

# Loop through each row and print the number of elements in each 'cognitive_levels' list
#for index, row in profile_database.iterrows():
    #print(f"Row {index}: {len(row['cognitive_levels'])} elements")
#print(profile_database.cognitive_levels)

# Identify the names of the knowledge nodes
with open(knowledge_nodes, 'r') as file:
    # Read the lines into a list
    labels = file.read().splitlines()

# Initialize a list to store the sum of cognitive levels for each label
cognitive_sums = [0] * len(labels)
cognitive_counts = [0] * len(labels)  # To keep track of the number of entries for each label
cognitive_max = [-float('inf')] * len(labels)
cognitive_min = [float('inf')] * len(labels)
cognitive_modes = [[] for _ in range(len(labels))]

# Iterate through each row in the DataFrame
for index, row in profile_database.iterrows():
    cognitive_levels = row['cognitive_levels']
    for i, level in enumerate(cognitive_levels):
        cognitive_sums[i] += level
        cognitive_counts[i] += 1
        cognitive_modes[i].append(level)
        if level > cognitive_max[i]:
            cognitive_max[i] = level
        if level < cognitive_min[i]:
            cognitive_min[i] = level

average_cognitive_levels = [sums / counts if counts > 0 else 0 for sums, counts in zip(cognitive_sums, cognitive_counts)]

# Calculate mode for each label
mode_cognitive_levels = []
for modes in cognitive_modes:
    if modes:
        mode_value = stats.mode(modes)[0]
        if isinstance(mode_value, (list, tuple)): # Check if it's a list or tuple
            mode_value = mode_value[0]
        mode_cognitive_levels.append(mode_value)
    else:
        mode_cognitive_levels.append(0)

# Plot the graph
plt.figure(figsize=(12, 8))
bar_width = 0.35
x = range(len(labels))

# Plot average cognitive levels
plt.bar(x, average_cognitive_levels, width=bar_width, color='blue', label='Average')

# Plot mode cognitive levels
plt.bar([p + bar_width for p in x], mode_cognitive_levels, width=bar_width, color='red', label='Mode')

plt.xlabel('Labels')
plt.ylabel('Cognitive Levels')
plt.title('Average and Mode Cognitive Levels for Each Label')
plt.xticks([p + bar_width/2 for p in x], labels, rotation=90, fontsize=8, fontweight='bold')  # Rotate labels and adjust their position
plt.legend()
plt.grid(axis='y')  # Add horizontal grid lines
plt.tight_layout()  # Adjust layout to fit labels

# Save the graph to the specified folder
plt.savefig('/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Figures/average_and_mode_cognitive_levels.png')

# Show the plot
plt.show()

plt.close()

# Plot the graph as a horizontal bar chart
plt.figure(figsize=(15, 10))
bar_width = 0.35
y = range(len(labels))

# Plot average cognitive levels
plt.barh(y, average_cognitive_levels, height=bar_width, color='red', label='Average')

# Plot mode cognitive levels
plt.barh([p + bar_width for p in y], mode_cognitive_levels, height=bar_width, color='green', label='Mode')

plt.ylabel('Labels')
plt.xlabel('Cognitive Levels')
plt.title('Average and Mode Cognitive Levels for Each Label')

# Apply bold formatting to labels
plt.yticks([p + bar_width/2 for p in y], labels, fontsize=10, fontweight='bold')
plt.legend()
plt.grid(axis='x')  # Add vertical grid lines
plt.tight_layout()  # Adjust layout to fit labels

# Save the graph to the specified folder
plt.savefig('/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Figures/average_and_mode_cognitive_levels_horizontal.png')

# Show the plot
plt.show()







