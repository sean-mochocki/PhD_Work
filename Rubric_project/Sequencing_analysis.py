import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#SA_Solutions = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Sequencing_results_SA.csv"
Random_Solutions = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Sequencing_results_random.csv"

df = pd.read_csv(Random_Solutions)
filtered_df = df[df['Number LMs'] > 10]
# min_scores = filtered_df.groupby('Student_id')['Combined'].min()
# filtered_df = SA_df[SA_df[['Student_id', 'Combined']].apply(tuple, axis=1).isin(min_scores.to_dict().items())]
#parameter_columns = ['Number Iterations', 'Cooling Rate', 'Initial Temperature']
parameter_columns = ['Number Iterations']

# Group by parameter columns and Student_id, then get the min Combined score for each group
min_scores_per_student_and_params = filtered_df.groupby(parameter_columns + ['Student_id'])['Combined'].min().reset_index()

# Group by parameter columns and get the mean of the min Combined scores.
mean_min_scores_per_params = min_scores_per_student_and_params.groupby(parameter_columns)['Combined'].mean().reset_index()

# Find the minimum mean score
best_mean_score = mean_min_scores_per_params['Combined'].min()

# Filter for the parameter combination with the best mean score
best_parameters = mean_min_scores_per_params[mean_min_scores_per_params['Combined'] == best_mean_score]

# Extract the best parameter values
best_params_values = best_parameters[parameter_columns].iloc[0].to_dict()

# Filter the original DataFrame to get rows with the best parameter combination
best_parameter_rows = df.copy()
for col, val in best_params_values.items():
    best_parameter_rows = best_parameter_rows[best_parameter_rows[col] == val]

Experiment = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Experiment_Results/best_random_Solution.csv"
best_parameter_rows.to_csv(Experiment)
