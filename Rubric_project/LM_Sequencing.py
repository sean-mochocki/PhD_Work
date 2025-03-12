import igraph as ig
import pandas as pd
import numpy as np
import re
import random
import copy
import math
from deap import base, creator, tools
import ast
import time
from scipy.optimize import dual_annealing

knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Knowledge_Nodes.txt"
knowledge_graph_edges = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Knowledge_Graph_Edges.txt"
learning_materials = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Learning_Materials_Base_set.xlsx"
LM_selection_solutions = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/best_initial_population_solution.csv"

# Identify the names of the knowledge nodes
with open(knowledge_nodes, 'r') as file:
    # Read the lines into a list
    KNs = file.read().splitlines()

# Read the edges from the file
with open(knowledge_graph_edges, "r") as file:
    edges = [line.strip().split(" -> ") for line in file]

# Create knowledge graph from data files
KG = ig.Graph(directed=True)
KG.add_vertices(KNs)
KG.add_edges(edges)

# Read in LM_database
LM_database = pd.read_excel(learning_materials)
LM_database['KNs Covered'] = LM_database['KNs Covered'].str.split(', ')

# Read in solutions
solution_database = pd.read_csv(LM_selection_solutions)
solution_database['Personalized Learning Path'] = solution_database['Personalized Learning Path'].apply(lambda x: np.array([int(i) for i in re.sub(r'[\[\]]', '', x).split()]))
#print(solution_database['Personalized Learning Path'])

#Solve problem for each student
for student_id in solution_database["Student_id"]:
    print("Student number:", student_id)
    student_solution = solution_database[solution_database["Student_id"] == student_id]
    personalized_learning_path = student_solution["Personalized Learning Path"].iloc[0]
    mask = personalized_learning_path == 1
    filtered_lm_database = LM_database[mask]
    #print(filtered_lm_database)

    # Define the LM difficulty and difficulty matching values for the learner
    def difficulty_to_int(difficulty):
        """
        :param difficulty: Takes the string description of difficulty
        :return: an integer corresponding to the cognitive level, 1,2,3, or 4
        """
        if difficulty == "Low":
            return 1
        elif difficulty == "Medium":
            return 2
        elif difficulty == "High":
            return 3
        elif difficulty == "Very High":
            return 4
        else:
            return 0  # default value for unknown strings


    LM_difficulty = filtered_lm_database['Knowledge Density (Subjective)']
    # Define LM_difficulty_int as integers between 1 and 4 where 1 is low and 4 is very high
    LM_difficulty = LM_difficulty.map(difficulty_to_int)
    original_lm_indices = LM_difficulty.index.tolist()

    LM_difficulty_list = LM_difficulty.tolist()

    LM_KNs_Covered = filtered_lm_database['KNs Covered']

    # Create look up table for LM difficulty
    n = len(LM_difficulty)
    difficulty_table = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:  # Exclude diagonal (penalty is 0)
                if LM_difficulty.iloc[i] > LM_difficulty.iloc[j]:
                    difficulty_table[i, j] = 1  # Penalty if row LM is more difficult than column LM.
                else:
                    difficulty_table[i, j] = 0

    many_to_many = False

    # Create a look up table for Prerequisites. We check the entire chain of prerequisites back to AI in General,
    # and if there are any violations we increment 1. Given that KNs are many-to-many there may be multiple penalties
    # between two LMs.

    prerequisite_table = np.zeros((n, n))
    root_node = "AI in General"

    for i in range(n):
        for j in range(n):
            if i != j:  # Exclude diagonal
                lm_i_kn_list = LM_KNs_Covered.iloc[i]
                lm_j_kn_list = LM_KNs_Covered.iloc[j]
                for kn_i in lm_i_kn_list:
                    for kn_j in lm_j_kn_list:
                        path_vertex_ids = KG.get_shortest_path(kn_i, root_node, output="vpath")
                        path_names = [KG.vs[v]["name"] for v in path_vertex_ids]
                        if kn_j in path_names and kn_j != kn_i:
                        #print(path_names)
                        # if [kn_i, kn_j] in edges:
                            prerequisite_table[i, j] += 1  # Increment penalty

    # create interleaving look up table
    interleaving_table = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lm_i_kn_list = LM_KNs_Covered.iloc[i]
            lm_j_kn_list = LM_KNs_Covered.iloc[j]
            for kn_i in lm_i_kn_list:
                if kn_i in lm_j_kn_list and i != j: interleaving_table[i, j] += 1 #Increment penalty, they cover the same KN

    print("Completed tables")

    for kn_list in LM_KNs_Covered:
        if len(kn_list) > 1:
            many_to_many = True
            break
    if many_to_many: print("many-to-many")
    else: print("many-to-one")

    many_to_many = True


    def generate_random_permutation(n):
        """
        Generates a random permutation of integers from 0 to n-1.

        Args:
            n: The number of LMs (the length of the permutation).

        Returns:
            A list representing a random permutation.
        """
        if n < 0:
            raise ValueError("n must be a non-negative integer.")

        permutation = list(range(n))  # Create a list of 0, 1, ..., n-1
        random.shuffle(permutation)  # Shuffle the list in place
        return permutation


    def generate_difficulty_sorted_permutation(n, LM_difficulty_list):
      # Create a list of (difficulty, index) tuples
        difficulty_tuples = [(difficulty, index) for index, difficulty in enumerate(LM_difficulty_list)]

        # Sort the tuples by difficulty (ascending)
        sorted_tuples = sorted(difficulty_tuples)

      # Create a mapping of original index to new position

        sorted_indices = [0] * n  # Initialize with zeros
        for new_position, (_, original_index) in enumerate(sorted_tuples):
            sorted_indices[original_index] = new_position

        return sorted_indices


    def generate_prerequisite_sorted_permutation(n, prerequisite_table):
        remaining_lms = list(range(n))
        sorted_positions = [0] * n
        position = 0

        while remaining_lms:
            no_prerequisites = []
            for lm in remaining_lms:
                is_no_prerequisite = True
                for other_lm in remaining_lms:
                    if lm != other_lm:
                        if prerequisite_table[lm][other_lm] != 0:
                            is_no_prerequisite = False
                            break
                if is_no_prerequisite:
                    no_prerequisites.append(lm)

            if not no_prerequisites:
                print(position)
                raise ValueError("Cycle detected in prerequisite table. Cannot create valid sequence.")

            for lm in no_prerequisites:
                sorted_positions[lm] = position
                position += 1
                remaining_lms.remove(lm)


        return sorted_positions

    def fitness_function (solution):
        difficulty_score = 0
        prerequisite_score = 0
        interleaving_score = 0

        for i in range(len(solution)):
            for j in range(len(solution)):
                if i != j:
                    if solution[i] < solution[j]:
                        difficulty_score += difficulty_table[i][j]
                        prerequisite_score += prerequisite_table[i][j]

        for i in range(len(solution) - 1):
            for j in range(i + 1, len(solution)):
                if abs(solution[i] - solution[j]) == 1:
                    interleaving_score += interleaving_table[i][j]

        combined_score = difficulty_score * weights[0] + prerequisite_score * weights[1] + interleaving_score * weights[2]

        return combined_score


    def hill_climber(solution, fitness_function):
        """
        Basic hill-climbing algorithm.

        Args:
            solution: The initial solution (a list).
            fitness_function: A function that evaluates the fitness of a solution.

        Returns:
            The best solution found.
        """

        best_fitness = fitness_function(solution)
        best_solution = copy.deepcopy(solution)  # Create a deep copy
        current_solution = copy.deepcopy(solution)

        improved = True  # add a variable to track if an improvement has been found.

        while improved:
            improved = False  # reset the improved variable.
            for i in range(len(current_solution)):
                for j in range(len(current_solution)):
                    if i != j:
                        neighbor_solution = copy.deepcopy(current_solution)  # create a copy of the current solution
                        # swap the values
                        neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
                        neighbor_fitness = fitness_function(neighbor_solution)

                        if neighbor_fitness < best_fitness:
                            best_fitness = neighbor_fitness
                            best_solution = neighbor_solution
                            current_solution = neighbor_solution  # update the current solution
                            improved = True  # set improved to true.
            if not improved:  # if no improvement was found, exit the while loop.
                break

        return best_solution

    # Create sorted difficulty score
    #lm_sequence_indices = generate_difficulty_sorted_permutation(n, LM_difficulty_list)
    weights = [0.4, 0.4, 0.3]

    lm_sequence_indices = generate_random_permutation(n)

    difficulty_score = 0
    prerequisite_score = 0
    interleaving_score = 0

    for i in range(len(lm_sequence_indices)):
        for j in range(len(lm_sequence_indices)):
            if i != j:
                if lm_sequence_indices[i] < lm_sequence_indices[j]:
                    difficulty_score += difficulty_table[i][j]
                    prerequisite_score += prerequisite_table[i][j]

    for i in range(len(lm_sequence_indices) - 1):
        for j in range(i + 1, len(lm_sequence_indices)):
            if abs(lm_sequence_indices[i] - lm_sequence_indices[j]) == 1:
                interleaving_score += interleaving_table[i][j]

    combined_score = difficulty_score * weights[0] + prerequisite_score * weights[1] + interleaving_score * weights[2]
    print("difficulty", difficulty_score)
    print("prerequisite", prerequisite_score)
    print("interleaving", interleaving_score)
    print("combined", combined_score)

    lm_sequence_indices = hill_climber(lm_sequence_indices, fitness_function)

    #lm_sequence_indices = generate_prerequisite_sorted_permutation(n, prerequisite_table)

    difficulty_score = 0
    prerequisite_score = 0
    interleaving_score = 0

    for i in range(len(lm_sequence_indices)):
        for j in range(len(lm_sequence_indices)):
            if i != j:
                if lm_sequence_indices[i] < lm_sequence_indices[j]:
                    difficulty_score += difficulty_table[i][j]
                    prerequisite_score += prerequisite_table[i][j]

    for i in range(len(lm_sequence_indices) - 1):
        for j in range(i + 1, len(lm_sequence_indices)):
            if abs(lm_sequence_indices[i] - lm_sequence_indices[j]) == 1:
                interleaving_score += interleaving_table[i][j]

    combined_score = difficulty_score * weights[0] + prerequisite_score * weights[1] + interleaving_score * weights[2]
    print("difficulty", difficulty_score)
    print("prerequisite", prerequisite_score)
    print("interleaving", interleaving_score)
    print("combined", combined_score)












