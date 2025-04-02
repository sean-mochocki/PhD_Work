import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


initial_population_ga = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/GA_Results_Exhaustive_search_Initial_Population.csv"
initial_population_ga_2 = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/GA_Results_Exhaustive_search_Initial_Population_2.csv"
random_population_ga = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/GA_Results_Exhaustive_Random_Population.csv"
random_population_ga_2 = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/GA_Results_Exhaustive_Random_Population_2.csv"

def find_consistent_best_parameters(df):
    """
    Finds the single parameter set that consistently results in the best scores across all students.

    Args:
        df (pandas.DataFrame): The DataFrame containing GA search results.

    Returns:
        tuple or None: The most frequent parameter set that results in the best scores, or None if no consistent set is found.
    """

    # Find the maximum Rubric Average for each student
    max_scores = df.groupby('Student_id')['Rubric Average'].max()

    # Filter the DataFrame to include only rows with the maximum Rubric Average for each student
    filtered_df = df[df[['Student_id', 'Rubric Average']].apply(tuple, axis=1).isin(max_scores.to_dict().items())]

    # Find the most frequent parameter set among the best scores
    parameter_columns = ['Num_Generation', 'Sol_per_pop', 'Num_parents_mating',
                         'Parent Selection Type', 'Mutation Type', 'Crossover Type',
                         'Mutation Probability', 'Crossover Probability']
    if not filtered_df.empty:
        parameter_counts = filtered_df.groupby(parameter_columns).size()
        max_count = parameter_counts.max()
        tying_params = parameter_counts[parameter_counts == max_count].index.tolist()

        # Get Rubric Average for each student using the best parameters
        rubric_averages_per_student = {}
        for student_id in df['Student_id'].unique():
            student_data = df[df['Student_id'] == student_id]
            rubric_averages_per_student[student_id] = {}
            for param_set in tying_params:
                best_params_data = student_data[student_data[parameter_columns].apply(tuple, axis=1) == param_set]
                if not best_params_data.empty:
                    rubric_averages_per_student[student_id][param_set] = best_params_data['Rubric Average'].values[0]
                else:
                    rubric_averages_per_student[student_id][param_set] = None

        print("Global Max Rubric Average for Each Student:")
        print(max_scores)
        print("\nRubric Average Achieved with Consistent Best Parameters for Each Student:")
        print(rubric_averages_per_student)

        return tying_params
    else:
        print("No single parameter set consistently produced the best scores across all students.")
        return None

    # if not filtered_df.empty:
    #     best_params = filtered_df.groupby(parameter_columns).size().idxmax()
    #
    #     # Get Rubric Average for each student using the best parameters
    #     rubric_averages_per_student = {}
    #     for student_id in df['Student_id'].unique():
    #         student_data = df[df['Student_id'] == student_id]
    #         best_params_data = student_data[student_data[parameter_columns].apply(tuple, axis=1) == best_params]
    #
    #         if not best_params_data.empty:
    #             rubric_averages_per_student[student_id] = best_params_data['Rubric Average'].values[
    #                 0]  # Grab the first value in the array.
    #         else:
    #             rubric_averages_per_student[student_id] = None  # Or some other indicator if not found
    #
    #     print("Global Max Rubric Average for Each Student:")
    #     print(max_scores)
    #     print("\nRubric Average Achieved with Consistent Best Parameters for Each Student:")
    #     print(rubric_averages_per_student)
    #
    #     return best_params
    # else:
    #     print("No single parameter set consistently produced the best scores across all students.")
    #     return None

    # if not filtered_df.empty:
    #     best_params = filtered_df.groupby(parameter_columns).size().idxmax()
    #     return best_params
    # else:
    #     return None


def plot_best_parameters(df):
    """
    Plots the best parameter configurations for each student, showing max and average rubric scores.

    Args:
        df (pandas.DataFrame): The DataFrame containing GA search results.
    """

    # Find the maximum Rubric Average for each student
    max_scores = df.groupby('Student_id')['Rubric Average'].max()

    # Filter the DataFrame to include only rows with the maximum Rubric Average for each student
    filtered_df = df[df[['Student_id', 'Rubric Average']].apply(tuple, axis=1).isin(max_scores.to_dict().items())]

    # Find the frequency of each parameter set
    parameter_columns = ['Num_Generation', 'Sol_per_pop', 'Num_parents_mating',
                         'Parent Selection Type', 'Mutation Type', 'Crossover Type',
                         'Mutation Probability', 'Crossover Probability']

    parameter_counts = filtered_df.groupby(parameter_columns).size()
    max_count = parameter_counts.max()
    tying_params = parameter_counts[parameter_counts == max_count].index.tolist()

    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    student_ids = sorted(df['Student_id'].unique())
    bar_width = 0.8 / (len(tying_params) + 1)  # Adjust bar width to fit all configurations and max scores

    # Plot max scores
    x_positions_max = np.arange(len(student_ids))
    plt.bar(x_positions_max, max_scores, width=bar_width, label="Rubric Max Score", color='black') #Add Max scores as black bars.

    for i, param_set in enumerate(tying_params):
        avg_scores = []
        for student_id in student_ids:
            student_data = df[df['Student_id'] == student_id]
            best_params_data = student_data[student_data[parameter_columns].apply(tuple, axis=1) == param_set]
            if not best_params_data.empty:
                avg_scores.append(best_params_data['Rubric Average'].mean())
            else:
                avg_scores.append(0)  # Or some other indicator if not found

        x_positions = np.arange(len(student_ids)) + (i + 1) * bar_width
        plt.bar(x_positions, avg_scores, width=bar_width, label=f"Config {i+1}")

    plt.xlabel('Student ID')
    plt.ylabel('Rubric Score')
    plt.title('Best Parameter Configurations and Rubric Scores')
    plt.xticks(np.arange(len(student_ids)) + bar_width * (len(tying_params) / 2), student_ids)
    plt.ylim(1, 4)  # Set y-axis limits
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_non_dominated_parameters(df):
    """
    Finds the non-dominated parameter sets, those that achieved the highest score for at least one student.

    Args:
        df (pandas.DataFrame): The DataFrame containing GA search results.

    Returns:
        list: A list of non-dominated parameter sets.
    """

    # Find the maximum Rubric Average for each student
    max_scores = df.groupby('Student_id')['Rubric Average'].max()

    # Filter the DataFrame to include only rows with the maximum Rubric Average for each student
    filtered_df = df[df[['Student_id', 'Rubric Average']].apply(tuple, axis=1).isin(max_scores.to_dict().items())]

    # Identify unique parameter sets
    parameter_columns = ['Num_Generation', 'Sol_per_pop', 'Num_parents_mating',
                         'Parent Selection Type', 'Mutation Type', 'Crossover Type',
                         'Mutation Probability', 'Crossover Probability']

    non_dominated_params = filtered_df[parameter_columns].drop_duplicates().values.tolist()

    return non_dominated_params

# initial_population_df = pd.read_csv(initial_population_ga)
# initial_population_2_df = pd.read_csv(initial_population_ga_2)
# initial_population_df = pd.concat([initial_population_df, initial_population_2_df])
initial_population_df = pd.read_csv(random_population_ga)
initial_population_2_df = pd.read_csv(random_population_ga_2)
initial_population_df = pd.concat([initial_population_df, initial_population_2_df])

# non_dominated_params = find_non_dominated_parameters(initial_population_df)
# print (len(non_dominated_params))
# print("Non-Dominated Parameter Sets:")
# for params in non_dominated_params:
#     print(params)

best_consistent_params = find_consistent_best_parameters(initial_population_df)

#plot_best_parameters(initial_population_df)

parameter_columns = ['Num_Generation', 'Sol_per_pop', 'Num_parents_mating',
                         'Parent Selection Type', 'Mutation Type', 'Crossover Type',
                         'Mutation Probability', 'Crossover Probability']

mask =  initial_population_df[parameter_columns].apply(tuple, axis=1) == best_consistent_params[0]
matching_rows = initial_population_df[mask]
matching_rows = matching_rows.drop_duplicates(subset=["Student_id"], keep='first')

Experiment = "Experiment_Results/Experiment.csv"
matching_rows.to_csv(Experiment)

if best_consistent_params:
    print("Consistently Best Performing Parameter Set:")
    print(best_consistent_params)
else:
    print("No single parameter set consistently produced the best scores across all students.")