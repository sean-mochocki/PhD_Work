import pandas as pd
import numpy as np
import re
import math
import plotly.express as px

plp_solutions = "Experiment_Results/LM_Selection_RSD_first_look.csv"
#plp_solutions = "Experiment_Results/LM_Selection_30_iterations_with_MDIP.csv"
report = pd.DataFrame()


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

id_max = plp_set["Student_id"].max()

for id in range(0, id_max+1):

    # Filter for data belonging to Student ID id
    student_data = plp_set[plp_set['Student_id'] == id].copy()

    # Ensure 'Iteration' is treated as an integer
    if 'Iteration' in student_data.columns:
        student_data['Iteration'] = pd.to_numeric(student_data['Iteration'], errors='coerce').astype('Int64')
    else:
        print("Warning: 'Iteration' column not found.")
        exit()

    # Find the index of the row with the maximum 'Rubric Average' for each 'Iteration'
    idx_top_scoring = student_data.groupby('Iteration')['Rubric Average'].idxmax()

    # Select all top-scoring rows
    top_scoring_student = student_data.loc[idx_top_scoring].copy()

    # Identify unique top-scoring solutions based on the 'categories'
    unique_top_scoring = top_scoring_student[categories].drop_duplicates().reset_index(drop=True)
    unique_top_scoring['Unique_Solution_ID'] = range(len(unique_top_scoring))
    # print("--- Unique Top-Scoring Solutions ---")
    # print(unique_top_scoring)
    # print(f"\nNumber of unique top-scoring solutions: {unique_top_scoring.shape[0]}")

    # Convert 'Personalized Learning Path' to NumPy arrays in the entire dataframe
    student_data['Personalized Learning Path'] = student_data['Personalized Learning Path'].apply(lambda x: np.array([int(i) for i in re.sub(r'[\[\]]', '', x).split()]))

    # Dictionary to store the counts of unique PLPs for each top-scoring solution
    plp_counts_per_solution = {}

    # Iterate through each unique top-scoring solution and count unique PLPs
    for index, row in unique_top_scoring.iterrows():
        match_condition = True
        for category in categories:
            match_condition &= (student_data[category] == row[category])

        matching_rows = student_data[match_condition]

        if not matching_rows.empty:
            unique_plps = matching_rows['Personalized Learning Path'].apply(tuple).nunique()
            plp_counts_per_solution[index] = unique_plps
        else:
            plp_counts_per_solution[index] = 0

    # print("\n--- Count of Unique Personalized Learning Paths per Top-Scoring Solution ---")
    # for solution_id, count in plp_counts_per_solution.items():
    #     print(f"Unique Top-Scoring Solution ID {solution_id}: {count} unique Personalized Learning Paths")

    # Merge the unique PLP counts back into the unique_top_scoring DataFrame
    unique_top_scoring['Unique_PLP_Count'] = unique_top_scoring.index.map(plp_counts_per_solution)

    # --- Merge with other information (Iterations Appeared In and Times Top Scoring) ---
    merged_data_for_tracking = pd.merge(student_data, unique_top_scoring[categories + ['Unique_Solution_ID']], on=categories, how='left')

    solution_occurrence = merged_data_for_tracking.groupby('Unique_Solution_ID').agg(
        Total_Iterations_Appears=('Iteration', 'nunique'),
        Times_Top_Scoring=('Unique_Solution_ID', 'count'),
        Iterations_Appeared_In=('Iteration', lambda x: sorted([int(i) for i in x.unique()]))
    ).reset_index().fillna(0)

    final_report = pd.merge(unique_top_scoring, solution_occurrence, on='Unique_Solution_ID', how='left').fillna(0)
    final_report['Student_id'] = id # Add Student_id column

    # --- Select the desired columns for the final CSV ---
    final_csv_data = final_report[['Student_id', 'Unique_Solution_ID'] + categories + ['Unique_PLP_Count', 'Total_Iterations_Appears', 'Times_Top_Scoring', 'Iterations_Appeared_In']]
    final_csv_data = final_csv_data.sort_values(by='Rubric Average', ascending = True)
    final_csv_data['Rubric Average'] = final_csv_data['Rubric Average'].apply(lambda x: f"{x:.2f}")

    report = pd.concat([report, final_csv_data])

# Save the final data to a CSV file
output_csv_file = "Experiment_Results/unique_top_scoring_solutions_report_1_iteration.csv"
report.to_csv(output_csv_file, index=False)

#print(f"\n--- Final Report Saved to: {output_csv_file} ---")
#print(final_csv_data)

# plp_set = pd.read_csv(plp_solutions)
#
#
# categories = [
#     'LM_Difficulty_Matching',
#     'CTML_Principle',
#     'media_matching',
#     'time_interval_score',
#     'coherence_principle',
#     'segmenting_principle',
#     'balance',
#     'cohesiveness',
#     'MDIP',
#     'Rubric Average'
# ]
#
# # Find the highest Student_id in the DataFrame
# highest_student_id = plp_set['Student_id'].max()
#
# student_11_data =plp_set[plp_set['Student_id'] == 11].copy()
#
# # Find the index of the row with the maximum 'Rubric Average' for each 'Iteration'
# idx_top_scoring = student_11_data.groupby('Iteration')['Rubric Average'].idxmax()
#
# # Select all top-scoring rows
# top_scoring_student_11 = student_11_data.loc[idx_top_scoring].copy()
#
# # Identify unique top-scoring solutions based on the 'categories'
# unique_top_scoring = top_scoring_student_11[categories].drop_duplicates().reset_index(drop=True)
# unique_top_scoring['Unique_Solution_ID'] = range(len(unique_top_scoring))
# print("--- Unique Top-Scoring Solutions ---")
# print(unique_top_scoring)
# print(f"\nNumber of unique top-scoring solutions: {unique_top_scoring.shape[0]}")
#
# # Convert 'Personalized Learning Path' to NumPy arrays in the entire dataframe
# plp_set['Personalized Learning Path'] = plp_set['Personalized Learning Path'].apply(lambda x: np.array([int(i) for i in re.sub(r'[\[\]]', '', x).split()]))
#
# # Dictionary to store the counts of unique PLPs for each top-scoring solution
# plp_counts_per_solution = {}
#
# # Iterate through each unique top-scoring solution and count unique PLPs
# for index, row in unique_top_scoring.iterrows():
#     match_condition = True
#     for category in categories:
#         match_condition &= (plp_set[category] == row[category])
#
#     matching_rows = plp_set[match_condition]
#
#     if not matching_rows.empty:
#         unique_plps = matching_rows['Personalized Learning Path'].apply(tuple).nunique()
#         plp_counts_per_solution[index] = unique_plps
#     else:
#         plp_counts_per_solution[index] = 0
#
# print("\n--- Count of Unique Personalized Learning Paths per Top-Scoring Solution ---")
# for solution_id, count in plp_counts_per_solution.items():
#     print(f"Unique Top-Scoring Solution ID {solution_id}: {count} unique Personalized Learning Paths")
#
# # Merge the unique PLP counts back into the unique_top_scoring DataFrame
# unique_top_scoring['Unique_PLP_Count'] = unique_top_scoring.index.map(plp_counts_per_solution)
#
# # --- Merge with other information (Total_Iterations_Appears and Times_Top_Scoring) ---
# merged_data_for_counts = pd.merge(student_11_data, unique_top_scoring[categories + ['Unique_Solution_ID']], on=categories, how='left')
#
# solution_occurrence = merged_data_for_counts.groupby('Unique_Solution_ID').agg(
#     Total_Iterations_Appears=('Iteration', 'nunique'),
#     Times_Top_Scoring=('Unique_Solution_ID', 'count')
# ).reset_index().fillna(0)
#
# final_report = pd.merge(unique_top_scoring, solution_occurrence, on='Unique_Solution_ID', how='left').fillna(0)
#
# # --- Select the desired columns for the final CSV ---
# final_csv_data = final_report[['Unique_Solution_ID'] + categories + ['Unique_PLP_Count', 'Total_Iterations_Appears', 'Times_Top_Scoring']]
#
# # Save the final data to a CSV file
# output_csv_file = "unique_top_scoring_solutions_report.csv"
# final_csv_data.to_csv(output_csv_file, index=False)
#
# print(f"\n--- Final Report Saved to: {output_csv_file} ---")
# print(final_csv_data)

# # --- Step 1: Identify Unique Top-Scoring Solutions ---
# # Find the index of the row with the maximum 'Rubric Average' for each 'Iteration'
# idx_top_scoring = student_11_data.groupby('Iteration')['Rubric Average'].idxmax()
#
# # Select all top-scoring rows
# top_scoring_student_11 = student_11_data.loc[idx_top_scoring].copy()
#
# # Identify unique top-scoring solutions based on the 'categories'
# unique_top_scoring = top_scoring_student_11[categories].drop_duplicates().reset_index(drop=True)
# unique_top_scoring['Unique_Solution_ID'] = range(len(unique_top_scoring))
# print("--- Unique Top-Scoring Solutions ---")
# print(unique_top_scoring)
# print(f"\nNumber of unique top-scoring solutions: {unique_top_scoring.shape[0]}")
#
#
# # Merge unique top-scoring solutions back to the original student 11 data to track occurrences
# merged_data = pd.merge(student_11_data, unique_top_scoring, on=categories, how='left')
#
# # --- Step 2: Count Iterations Each Unique Solution Appears In and Times It's Top Scoring ---
# solution_occurrence = merged_data.groupby('Unique_Solution_ID').agg(
#     Total_Iterations_Appears=('Iteration', 'nunique'),
#     Times_Top_Scoring=('Unique_Solution_ID', 'count') # Count of rows where this Unique_Solution_ID is not NaN
# ).reset_index()
#
# # Convert 'Personalized Learning Path' to NumPy arrays in the entire dataframe
# plp_set['Personalized Learning Path'] = plp_set['Personalized Learning Path'].apply(lambda x: np.array([int(i) for i in re.sub(r'[\[\]]', '', x).split()]))
#
# # Dictionary to store the counts of unique PLPs for each top-scoring solution
# plp_counts_per_solution = {}
#
# # Iterate through each unique top-scoring solution
# for index, row in unique_top_scoring.iterrows():
#     # Filter the entire solution_database for rows matching the current top-scoring solution
#     match_condition = True
#     for category in categories:
#         match_condition &= (plp_set[category] == row[category])
#
#     matching_rows = plp_set[match_condition]
#
#     if not matching_rows.empty:
#         # Get the unique Personalized Learning Paths for these matching rows
#         unique_plps = matching_rows['Personalized Learning Path'].apply(tuple).nunique() # Convert to tuple for hashability
#
#         plp_counts_per_solution[index] = unique_plps
#     else:
#         plp_counts_per_solution[index] = 0
#
# # Merge back to include the actual unique solutions
# final_report = pd.merge(unique_top_scoring, solution_occurrence, unique_plps, on='Unique_Solution_ID', how='left').fillna(0)
#
# # Display the results
# print("\n--- Count of Unique Personalized Learning Paths per Top-Scoring Solution ---")
# for solution_id, count in plp_counts_per_solution.items():
#     print(f"Unique Top-Scoring Solution ID {solution_id}: {count} unique Personalized Learning Paths")
#
# print("\n--- Unique Top-Scoring Solution Report ---")
# print(final_report)

#
#
# idx = student_11_data.groupby('Iteration')['Rubric Average'].idxmax()
# # Select the top scoring rows
# top_scoring_student_11 = student_11_data.loc[idx]
#
# unique_top_scoring = top_scoring_student_11[categories].drop_duplicates()
# print(f"Number of unique top-scoring solutions: {unique_top_scoring.shape[0]}")
# unique_top_scoring['Solution_ID'] = range(len(unique_top_scoring))
#
# # Create the parallel coordinates chart
# fig = px.parallel_coordinates(top_scoring_student_11,
#                              dimensions=categories,
#                              color='Solution_ID',  # Color lines by Iteration
#                              color_continuous_scale=px.colors.sequential.Plasma,
#                              labels={col: col for col in categories},  # Use original category names
#                              title='Parallel Coordinates of Top Scoring Iterations for Student 11'
#                              )
#
# # Manually set the tick values and range for all discrete categories
# for dimension in fig.data[0]['dimensions']:
#     if dimension['label'] != 'Rubric Average':
#         dimension['tickvals'] = [1, 2, 3, 4]
#         dimension['ticktext'] = ['1', '2', '3', '4']
#         dimension['range'] = [0.5, 4.5]  # Set a small buffer
#
# # Set the range and tick values for 'Rubric Average' to align with 1-4
# min_rubric = 0.5  # Set a small buffer below 1
# max_rubric = 4.5  # Set a small buffer above 4
# rubric_tickvals = [1, 2, 3, 4]
# rubric_ticktext = ['1', '2', '3', '4']
#
# for dimension in fig.data[0]['dimensions']:
#     if dimension['label'] == 'Rubric Average':
#         dimension['range'] = [min_rubric, max_rubric]
#         dimension['tickvals'] = rubric_tickvals
#         dimension['ticktext'] = rubric_ticktext
#         break # Assuming 'Rubric Average' appears only once
#
# fig.show()



# def get_max_rows(group):
#     max_rubric = group['Rubric Average'].max()
#     return group[group['Rubric Average'] == max_rubric]
#
#
# top_scoring_student_11 = student_11_data.groupby('Iteration', group_keys=False).apply(get_max_rows)
# print(
#     f"Shape of top scoring data for Student 11 (all max Rubric Average per Iteration): {top_scoring_student_11.shape}")
# print("\nAll rows with the maximum Rubric Average for Student 11 per Iteration:")
# print(top_scoring_student_11)



# # # Create an empty dictionary to store the results for each student
# student_stats = {}
# #
# # Iterate through each student ID
# for student_id in range(highest_student_id + 1):
#     # Filter the DataFrame for the current student
#     student_data = plp_set[plp_set['Student_id'] == student_id]
#
#     if not student_data.empty:
#         # Calculate the average score for each category
#         average_scores = student_data[categories].mean()
#
#         # Calculate the variance for each category
#         variance_scores = student_data[categories].var()
#
#         # Store the results for the current student
#         student_stats[student_id] = {
#             'Average_' + col: average_scores[col] for col in categories
#         }
#         student_stats[student_id].update({
#             'Variance_' + col: variance_scores[col] for col in categories
#         })
#     else:
#         # Handle the case where a student might not have any data
#         student_stats[student_id] = {
#             'Average_' + col: np.nan for col in categories
#         }
#         student_stats[student_id].update({
#             'Variance_' + col: np.nan for col in categories
#         })
#
# print("done")
# # Convert the dictionary of student statistics into a pandas DataFrame
# results_df = pd.DataFrame.from_dict(student_stats, orient='index')
# results_df.index.name = 'Student_id'  # Set the index name
#
# # Save the resulting DataFrame to a CSV file
# results_df.to_csv('Experiment_Results/student_category_stats.csv')