import pandas as pd
import numpy as np
import re
import math
import ast

student_data = "Data/Learner_Profile_8_Jan_2025.xlsx"
knowledge_nodes = "Data/Knowledge_Nodes.txt"

student_dataset = pd.read_excel(student_data)

with open(knowledge_nodes, 'r') as file:
    # Read the lines into a list
    KNs = file.read().splitlines()

student_dataset['cognitive_levels'] = student_dataset['cognitive_levels'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])
student_dataset['goals'] = student_dataset['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

student_dataset['average_cognitive_level'] = student_dataset['cognitive_levels'].apply(lambda x: np.mean(x))
student_dataset['variance_cognitive_level'] = student_dataset['cognitive_levels'].apply(lambda x: np.var(x))

student_dataset['goal_names'] = student_dataset['goals'].apply(lambda goal_indices: [KNs[i-1] for i in goal_indices])

output_csv_file = "Experiment_Results/Updated_Profile_data.csv"
student_dataset.to_csv(output_csv_file, index=False)

print("stuff")