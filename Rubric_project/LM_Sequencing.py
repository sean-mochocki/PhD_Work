import igraph as ig
import pandas as pd
import numpy as np
import re
import random
import copy
import itertools
import math
import time

knowledge_nodes = "Data/Knowledge_Nodes.txt"
knowledge_graph_edges = "Data/Knowledge_Graph_Edges.txt"
learning_materials = "Data/Learning_Materials_Base_set.xlsx"
LM_selection_solutions = "Data/best_initial_population_solution.csv"

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

experiment_df = pd.DataFrame()

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

    # Calculate max for 3 objectives
    difficulty_max = np.sum(difficulty_table)
    prerequisite_max = np.sum(prerequisite_table)
    interleaving_max = np.sum(np.tril(interleaving_table))

    for kn_list in LM_KNs_Covered:
        if len(kn_list) > 1:
            many_to_many = True
            break

    if many_to_many: print("many-to-many")
    else: print("many-to-one")

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


    def fitness_function(solution):
        """Optimized fitness function."""

        solution_np = np.array(solution)
        n = len(solution)

        # Precompute difficulty and prerequisite scores
        difficulty_score = 0
        prerequisite_score = 0
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] < solution[j]:
                    difficulty_score += difficulty_table[i][j]
                    prerequisite_score += prerequisite_table[i][j]
                else:
                    difficulty_score += difficulty_table[j][i]
                    prerequisite_score += prerequisite_table[j][i]

        # Precompute interleaving scores
        interleaving_score = 0
        for i, j in itertools.combinations(range(n), 2):
            if abs(solution_np[i] - solution_np[j]) == 1:
                interleaving_score += interleaving_table[i][j]

        if difficulty_max > 0:
            difficulty_raw_score = (difficulty_score / difficulty_max)
        else:
            difficulty_raw_score = 0.0

        if prerequisite_max > 0:
            prerequisite_raw_score = prerequisite_score / prerequisite_max
        else:
            prerequisite_raw_score = 0.0

        if interleaving_max > 0:
            interleaving_raw_score = (interleaving_score / interleaving_max)
        else:
            interleaving_raw_score = 0.0

        # Define fitness function using raw scores for prerequisite, interleaving and difficulty sequencing.

        combined_score = difficulty_raw_score * weights[0] + prerequisite_raw_score * weights[1] + interleaving_raw_score * weights[2]
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

    def calculate_combined_cost(lm_index, remaining_lms_excluding_current):
        """Calculates the combined cost of placing lm_index before other LMs."""
        total_cost = 0
        for other_lm in remaining_lms_excluding_current:
            total_cost += difficulty_table[lm_index][other_lm] + prerequisite_table[lm_index][other_lm]
        return total_cost


    def sort_lms_by_combined_cost(remaining_lms, difficulty_table, prerequisite_table):
        """
        Sorts Learning Materials (LMs) based on combined prerequisite and difficulty costs,
        with tie-breaking based on the next best LM's cost.

        Args:
            remaining_lms: A list of LM indices to be sorted.
            difficulty_table: A 2D NumPy array representing difficulty costs.
            prerequisite_table: A 2D NumPy array representing prerequisite costs.

        Returns:
            A list of LM indices sorted according to combined costs.
        """

        sorted_lms = []
        remaining_lms_copy = remaining_lms[:]  # Create a copy to avoid modifying the original list.

        while remaining_lms_copy:
            costs = []
            for lm in remaining_lms_copy:
                other_lms = [other for other in remaining_lms_copy if other != lm]
                costs.append((calculate_combined_cost(lm, other_lms), lm))

            costs.sort()  # Sort by combined cost (ascending)

            best_cost = costs[0][0]
            tie_lms = [lm for cost, lm in costs if cost == best_cost]

            if len(tie_lms) > 1:  # Tie detected
                tie_breaker_scores = {}
                for tied_lm in tie_lms:
                    temp_remaining = remaining_lms_copy[:]
                    temp_remaining.remove(tied_lm)
                    if temp_remaining:  # check if remaining is not empty.
                        next_costs = []
                        for next_lm in temp_remaining:
                            next_other_lms = [other for other in temp_remaining if other != next_lm]
                            next_costs.append((calculate_combined_cost(next_lm, next_other_lms), next_lm))
                        next_costs.sort()
                        next_best_cost = next_costs[0][0]
                        tie_breaker_scores[tied_lm] = next_best_cost
                    else:
                        tie_breaker_scores[tied_lm] = float(
                            'inf')  # if no remaining LMs, set tie breaker score to infinity.

                best_lm = min(tie_breaker_scores, key=tie_breaker_scores.get)
            else:
                best_lm = tie_lms[0]

            sorted_lms.append(best_lm)
            remaining_lms_copy.remove(best_lm)

        return sorted_lms

    def greedy_prerequisite_sequencing(n, difficulty_table, prerequisite_table):
        """
        Greedy algorithm to sequence LMs based on combined prerequisite and difficulty costs.

        Args:
            n: The number of LMs.
            difficulty_table: A 2D NumPy array representing difficulty costs.
            prerequisite_table: A 2D NumPy array representing prerequisite costs.

        Returns:
            A list representing the greedy sequence of LM indices.
        """

        solution = []
        remaining_lms = list(range(n))

        while remaining_lms:
            sorted_lms = sort_lms_by_combined_cost(remaining_lms, difficulty_table, prerequisite_table)
            best_lm = sorted_lms[0]  # Get the LM with the lowest cost (first element)
            solution.append(best_lm)
            remaining_lms.remove(best_lm)

        return solution


    def simulated_annealing(solution, fitness_function, initial_temperature=100, cooling_rate=0.99,
                            max_iterations=1000):
        """
        Simulated annealing algorithm.

        Args:
            solution: The initial solution (a list).
            fitness_function: A function that evaluates the fitness of a solution.
            initial_temperature: The initial temperature.
            cooling_rate: The cooling rate (0 < cooling_rate < 1).
            max_iterations: The maximum number of iterations.

        Returns:
            The best solution found.
        """

        best_fitness = fitness_function(solution)
        best_solution = copy.deepcopy(solution)
        current_solution = copy.deepcopy(solution)
        temperature = initial_temperature

        for _ in range(max_iterations):
            i, j = random.sample(range(len(current_solution)), 2)  # Randomly select two distinct indices

            neighbor_solution = copy.deepcopy(current_solution)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
            neighbor_fitness = fitness_function(neighbor_solution)

            delta_fitness = neighbor_fitness - best_fitness  # calculate the change in fitness

            if delta_fitness < 0:  # better solution
                best_fitness = neighbor_fitness
                best_solution = neighbor_solution
                current_solution = neighbor_solution
            else:  # worse solution
                acceptance_probability = math.exp(-delta_fitness / temperature)
                if random.random() < acceptance_probability:
                    current_solution = neighbor_solution

            temperature *= cooling_rate  # cool down

        return best_solution


    def exhaustive_search(n, fitness_function):
        """
        Performs an exhaustive search on the solution space and returns the best solution.

        Args:
            solution_space: An iterable (e.g., list, tuple) representing the possible solutions.
                            If the solution space is a permutation problem, this should be a list of the items to be permuted.
            fitness_function: A function that takes a solution as input and returns its fitness score.

        Returns:
            The best solution found and its fitness score.
        """
        solution_space = list(range(n))
        best_solution = None
        best_fitness = float('inf')  # Initialize with positive infinity (assuming minimization)

        for solution in itertools.permutations(solution_space):

            current_fitness = fitness_function(list(solution))  # Convert tuple to list for consistency

            if current_fitness < best_fitness:  # Assuming minimization
                best_fitness = current_fitness
                best_solution = list(solution)  # Store the solution as a list

        return best_solution

    def random_search(solution, fitness_function, max_iterations=1000):
        """
        Random search algorithm.

        Args:
            solution: The initial solution (a list).
            fitness_function: A function that evaluates the fitness of a solution.
            max_iterations: The maximum number of iterations.

        Returns:
            The best solution found.
        """

        best_fitness = fitness_function(solution)
        best_solution = copy.deepcopy(solution)
        current_solution = copy.deepcopy(solution)

        # iterations_without_improvement = 0

        for _ in range(max_iterations):
            i, j = random.sample(range(len(current_solution)), 2)  # Randomly select two distinct indices

            neighbor_solution = copy.deepcopy(current_solution)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
            neighbor_fitness = fitness_function(neighbor_solution)

            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_solution = neighbor_solution
                current_solution = neighbor_solution

        return best_solution

    def report_fitness_metrics(solution):
        difficulty_score = 0
        prerequisite_score = 0
        interleaving_score = 0

        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                if solution[i] < solution[j]:
                    difficulty_score += difficulty_table[i][j]
                    prerequisite_score += prerequisite_table[i][j]
                else:
                    difficulty_score += difficulty_table[j][i]
                    prerequisite_score += prerequisite_table[j][i]

        for i in range(len(solution) - 1):
            for j in range(i + 1, len(solution)):
                if abs(solution[i] - solution[j]) == 1:
                    interleaving_score += interleaving_table[i][j]

        combined_score = difficulty_score * weights[0] + prerequisite_score * weights[1] + interleaving_score * weights[2]
        return difficulty_score, prerequisite_score, interleaving_score, combined_score


    # Create sorted difficulty score
    #lm_sequence_indices = generate_difficulty_sorted_permutation(n, LM_difficulty_list)
    weights = [0.33333, 0.33333, 0.33333]

    num_iterations = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    initial_temperature = [200, 150, 100]
    cooling_rate = [0.999, 0.99, 0.95]

    # num_iterations = [1000, 2000]
    # initial_temperature = [200, 100]
    # cooling_rate = [0.99, 0.95]

    parameter_combinations = list(itertools.product(
        num_iterations,
        initial_temperature,
        cooling_rate
    ))

    # Run experiment

    #if n < 10: lm_sequence_indices = exhaustive_search(n, fitness_function)
    #else:
    #lm_sequence_indices = hill_climber(lm_sequence_indices, fitness_function)
    #lm_sequence_indices = greedy_prerequisite_sequencing(n, difficulty_table, prerequisite_table)

    #for iterations in num_iterations:
    for combination in parameter_combinations:
        num_iterations, initial_temperature, cooling_rate = combination

        lm_sequence_indices = generate_random_permutation(n)
        start_time = time.time()
        #lm_sequence_indices = random_search(lm_sequence_indices, fitness_function, iterations)
        lm_sequence_indices = simulated_annealing(lm_sequence_indices, fitness_function, initial_temperature, cooling_rate, num_iterations)
        #lm_sequence_indices = generate_difficulty_sorted_permutation(n, LM_difficulty_list)
        end_time = time.time()

        elapsed_time = end_time - start_time

        difficulty_score, prerequisite_score, interleaving_score, combined_score = report_fitness_metrics(lm_sequence_indices)

        # Check for worst scores of 0 and award perfect score in these cases

        if difficulty_max > 0: difficulty_raw_score = (difficulty_score / difficulty_max)
        else: difficulty_raw_score = 0.0

        if prerequisite_max > 0: prerequisite_raw_score = prerequisite_score / prerequisite_max
        else: prerequisite_raw_score = 0.0

        if interleaving_max > 0:
            interleaving_raw_score = (interleaving_score / interleaving_max)
        else: interleaving_raw_score = 0.0

        difficulty_rubric_score = 1
        if difficulty_raw_score <= 0.25: difficulty_rubric_score = 4
        elif difficulty_raw_score <= 0.5: difficulty_rubric_score = 3
        elif difficulty_raw_score <= 0.75: difficulty_rubric_score = 2

        prerequisite_rubric_score = 1
        if prerequisite_raw_score <= 0.25: prerequisite_rubric_score = 4
        elif prerequisite_raw_score <= 0.5: prerequisite_rubric_score = 3
        elif prerequisite_raw_score <= 0.75: prerequisite_rubric_score = 2

        interleaving_rubric_score = 1
        if interleaving_raw_score <= 0.25: interleaving_rubric_score = 4
        elif interleaving_raw_score <= 0.5: interleaving_rubric_score = 3
        elif interleaving_raw_score <= 0.75: interleaving_rubric_score = 2

        print("Score of solved solution:")
        print("Number of iterations is: ", num_iterations)
        print("Cooling Rate is:", cooling_rate)
        print("Initial Temperature is:", initial_temperature)
        print("Number of LMs is:", len(lm_sequence_indices))
        print("difficulty", difficulty_score)
        print("difficulty max", difficulty_max)
        print("difficulty raw score", difficulty_raw_score)
        print("difficulty rubric score", difficulty_rubric_score)
        print("prerequisite", prerequisite_score)
        print("prerequisite max score", prerequisite_max)
        print("prerequisite raw score", prerequisite_raw_score)
        print("prerequisite rubric score", prerequisite_rubric_score)
        print("interleaving", interleaving_score)
        print("interleaving max", interleaving_max)
        print("interleaving raw score", interleaving_raw_score)
        print("interleaving rubric score", interleaving_rubric_score)
        print("combined", combined_score)
        print(f"Solved Solution elapsed time: {elapsed_time} seconds")
        print("****************************************************************")

        data = {
            "Student_id": int(student_id),
            "Number Iterations": num_iterations,
            "Cooling Rate": cooling_rate,
            "Initial Temperature": initial_temperature,
            "Personalized Learning Path": str(personalized_learning_path),
            "Number LMs": len(lm_sequence_indices),
            "Many-to-Many": many_to_many,
            "Sequence": str(lm_sequence_indices),
            "Difficulty": difficulty_score,
            "Difficulty Max": difficulty_max,
            "Difficulty Raw Score": difficulty_raw_score,
            "Prerequisite": prerequisite_score,
            "Prerequisite Max": prerequisite_max,
            "Interleaving": interleaving_score,
            "Interleaving Max": interleaving_max,
            "Prerequisite_raw_score": prerequisite_raw_score,
            "Interleaving_raw_score": interleaving_raw_score,
            "Difficulty Rubric Score": difficulty_rubric_score,
            "Prerequisite Rubric Score": prerequisite_rubric_score,
            "Interleaving Rubric Score": interleaving_rubric_score,
            "Combined": combined_score,
            "Algorithm time": elapsed_time
        }

        data = pd.DataFrame(data, index=[0])
        experiment_df = pd.concat([experiment_df, data], ignore_index=True)


Experiment = "Experiment_Results/Experiment.csv"
experiment_df.to_csv(Experiment)
    #lm_sequence_indices = generate_prerequisite_sorted_permutation(n, prerequisite_table)














