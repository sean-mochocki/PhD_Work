import pandas as pd
import numpy as np
import re

plp_solutions = "Experiment_Results/LM_Selection_Parameter_30_iteration.csv"
plp_set = pd.read_csv(plp_solutions)

categories = [
    'LM_Difficulty_Matching',
    'CTML_Principle',
    'media_matching',
    'time_interval_score',
    'coherence_principle',
    'segmenting_principle',
    'balance',
    'cohesiveness',
    'MDIP',
    'Rubric Average'
]

# Function to get the rows with the maximum 'Rubric Average' for each Student_id and Iteration
def get_best_solutions(df, categories):
    best_solutions = df.loc[df.groupby(['Student_id', 'Iteration'])['Rubric Average'].idxmax()]
    return best_solutions

# Get best solutions
best_solutions_df = get_best_solutions(plp_set, categories)

# 1. Group by Student_id and Iteration, then calculate the mean for each category
student_iteration_means = best_solutions_df.groupby(['Student_id', 'Iteration'])[categories].mean().reset_index()

# 2. Group by Student_id, then calculate the variance of the means
student_variances = student_iteration_means.groupby('Student_id')[categories].var()

variance_csv_file = "Experiment_Results/student_variances.csv"
student_variances.to_csv(variance_csv_file, index = False)

# print("Variances of category averages across iterations for each student:")
# print(student_variances)

# Calculate the mean of the category averages for each student
student_category_means = student_iteration_means.groupby('Student_id')[categories].mean().reset_index()

means_csv_file = "Experiment_Results/30_iterations_student_means.csv"
student_category_means.to_csv(means_csv_file, index = False)

# If you want this in a long format for further analysis or CSV export:
student_variances_long = student_variances.reset_index().melt(id_vars='Student_id', value_name='Variance', var_name='Category')
print("\nVariances in long format:")
print(student_variances_long)



# # Example of how to access a specific variance:
# student_11_rubric_variance = student_variances.loc[11, 'Rubric Average']  # Assuming Student_id is 11
# print(f"\nVariance of Rubric Average for Student 11: {student_11_rubric_variance:.4f}")
