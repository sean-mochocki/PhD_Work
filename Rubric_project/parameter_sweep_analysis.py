import pandas as pd
import numpy as np
import re

plp_solutions = "Experiment_Results/LM_Selection_Parameter_Sweep_Final.csv"
report = pd.DataFrame()

df = pd.read_csv(plp_solutions)

# 1. Group by 'Student_id', 'Crossover Probability', and 'Num_parents_mating'
grouped_student_params = df.groupby(['Student_id', 'Crossover Probability', 'Num_parents_mating'])

# 2. Find the maximum 'Rubric Average' for each student and parameter combination
max_rubric_per_student_param = grouped_student_params['Rubric Average'].max()

# Reset the index to make 'Student_id', 'Crossover Probability', and 'Num_parents_mating' regular columns again
max_rubric_df = max_rubric_per_student_param.reset_index()

# 3. Group by 'Crossover Probability' and 'Num_parents_mating'
grouped_params = max_rubric_df.groupby(['Crossover Probability', 'Num_parents_mating'])

# 4. Calculate the average of the maximum 'Rubric Average' across students for each parameter combination
average_of_max_rubric = grouped_params['Rubric Average'].mean()

# 5. Find the maximum of these averages
max_average_of_max = average_of_max_rubric.max()

# 6. Identify the parameter combinations that achieved this maximum average
best_combinations = average_of_max_rubric[average_of_max_rubric == max_average_of_max].index.tolist()

print("The best performing parameter combination(s) (based on the average of the maximum 'Rubric Average' per student):")
for combo in best_combinations:
    print(f"Crossover Probability: {combo[0]}, Num_parents_mating: {combo[1]}")