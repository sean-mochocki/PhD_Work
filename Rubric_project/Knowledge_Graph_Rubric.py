import igraph as ig
import pandas as pd
import ast
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import pygad
import torch
from pulp import *
import time
import itertools
import random

knowledge_nodes = "Data/Knowledge_Nodes.txt"
knowledge_graph_edges = "Data/Knowledge_Graph_Edges.txt"
learner_profile = "Data/Learner_Profile_8_Jan_2025.xlsx"
learning_materials = "Data/Learning_Materials_Base_set.xlsx"

# This is the section where we create the Knowledge Graph

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

# Create LM database and define parameters
LM_database = pd.read_excel(learning_materials)
LM_database['KNs Covered'] = LM_database['KNs Covered'].str.split(', ')
LM_database['Time to Complete'] = LM_database['Time to Complete'].str.split(":").apply(lambda x: int(x[0]) + int(x[1]) / 60)
# Define the duration of the LMs - this does not depend on individual learners
lm_time_taken = LM_database['Time to Complete'].to_numpy()
lm_time_taken = [int(x * 100) for x in lm_time_taken]

# Find the global max time for all LMs
global_max_time = np.sum(lm_time_taken)

# Get Sentence Transformer model to be used to calculate cohesiveness
model = SentenceTransformer('all-mpnet-base-v2')
LM_database['embeddings'] = LM_database['Description'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def calculate_cohesiveness(personalized_learning_path, LM_database):
    """Calculates cohesiveness efficiently using vectorized operations."""
    num_LMs = len(personalized_learning_path)  # Get total number of LMs
    included_indices = [i for i in range(num_LMs) if personalized_learning_path[i] == 1]
    included_embeddings = [LM_database['embeddings'][i] for i in included_indices]

    num_included_LMs = len(included_embeddings)

    if num_included_LMs < 2:  # Handle the case of 0 or 1 included LMs
        return 0.0

    # Convert list of tensors to a single tensor
    included_embeddings_tensor = torch.stack(included_embeddings)

    # Efficiently calculate all pairwise cosine similarities
    similarity_matrix = util.pytorch_cos_sim(included_embeddings_tensor, included_embeddings_tensor)

    # Normalize to 0-1 (vectorized):
    normalized_similarity_matrix = (similarity_matrix + 1) / 2

    # Zero out the diagonal and upper triangle (or lower triangle, but be consistent):
    mask = torch.ones_like(normalized_similarity_matrix, dtype=torch.bool).triu(diagonal=0)  # Upper triangle including diagonal
    normalized_similarity_matrix = normalized_similarity_matrix.masked_fill(mask, 0.0)

    # Calculate the sum of the remaining (lower triangle) similarities
    total_similarity = normalized_similarity_matrix.sum().item()

    count = num_included_LMs * (num_included_LMs - 1) / 2 # Number of pairs

    average_cohesiveness = total_similarity / count if count > 0 else 0.0 # Normalized score
    average_cohesiveness = max(0.0, min(1.0, average_cohesiveness))

    return average_cohesiveness

# Calculate the CTML score of each LM
lm_Multimedia_score = LM_database['Multimedia Principle']

CTML_List = ['Coherence Principle', 'Segmenting Principle', 'Worked Example Principle', 'Signaling Principle', 'Spatial Contiguity Principle', 'Temporal Contiguity Principle', 'Modality Principle',
             'Redundancy Principle', 'Personalization Principle', 'Voice Principle', 'Sourcing Principle']

lm_CTML_score = []

#Find the average CTML score for each LM
for index, score in enumerate(lm_Multimedia_score):
    if score== 1: lm_CTML_score.append(1)
    else:
        lm_running_average = 0
        lm_CTML_count = 0
        for element in CTML_List:
            if LM_database[element][index]!=0:
                lm_running_average += LM_database[element][index]
                lm_CTML_count += 1
        # Determine the lm_CTML_score based on the running average and the multimedia score.
        lm_CTML_score.append((lm_running_average * score/4)/lm_CTML_count)

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

LM_difficulty = LM_database['Knowledge Density (Subjective)']
# Define LM_difficulty_int as integers between 1 and 4 where 1 is low and 4 is very high
LM_difficulty_int = LM_difficulty.map(difficulty_to_int)

# ************************ Metadata so far ***************************
# LM_difficulty_int - a list of integers describing the LM difficulty from 1 to 4
# lm_CTML_score - a list of floats describing the CTML value of the LMs from 1 to 4
# lm_time_taken - a list of integers describing the length of LMs multiplied (Minutes + seconds) *100
# KNs - The list of all KNs as strings
# LM_database['embeddings'] - used to calculate cohesiveness score
# ********************************************************************

# Derive the set of learner profiles from the dataframe
profile_database = pd.read_excel(learner_profile)
profile_database['goals'] = profile_database['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

experiment_df = pd.DataFrame()
num_iterations = 30
for _ in range(num_iterations):
    print("Iteration number: ", _, "of", num_iterations)
    for student_profile_id in range(len(profile_database)):
        print("Student profile is: ", student_profile_id )

        goal_nodes = profile_database['goals'][student_profile_id]

        # The Knowledge graph is derived from a taxonomy, so it has the property that there is only one path from each KN to
        # the root KN. We use that function here to derive KNs that should be included in the goal KNs.
        KS = []
        for goals in goal_nodes:
            if goals != 1:
                paths = KG.get_shortest_paths(goals-1, 0)
                for path in paths:
                    KS.extend(path)  # Append individual nodes from each path

        # Delete duplicates from the knowledge set
        KS = list(set(KS))
        # Calculate the names of the KNs in KS
        KS_names = [KNs[i] for i in KS]

        # Multiply by 100 for convenience when comparing student learning goal times to LM times
        max_time = int(profile_database['maximum_time'][student_profile_id]) * 100
        min_time = int(profile_database['minimum_time'][student_profile_id]) * 100

        # Get the first element of the column as a string
        cog_levels_str = profile_database['cognitive_levels'][student_profile_id]

        # Evaluate the string as a Python expression and convert it into a list of integers
        # These are ranked from CL 1 to 4, where 4 is most advanced
        cog_levels_list = list(ast.literal_eval(cog_levels_str))

        # Capture the users rankings for preferred content type from 1-9, where 1 is most preferred
        research = int(profile_database['research'][student_profile_id])
        website = int(profile_database['website'][student_profile_id])
        discussion = int(profile_database['discussion'][student_profile_id])
        educational = int(profile_database['educational'][student_profile_id])
        news_article = int(profile_database['news_article'][student_profile_id])
        diy = int(profile_database['diy'][student_profile_id])
        lecture = int(profile_database['lecture'][student_profile_id])
        powerpoint = int(profile_database['powerpoint'][student_profile_id])
        textbook_excerpt = int(profile_database['textbook_excerpt'][student_profile_id])

        # Create a dictionary that maps each content type to its ranking
        content_ranking = {
            'Research': research,
            'Website': website,
            'Discussion': discussion,
            'Educational': educational,
            'Article': news_article,
            'DIY': diy,
            'Lecture': lecture,
            'Powerpoint': powerpoint,
            'Textbook': textbook_excerpt
        }

        # Create a list that scores each LM based on the dictionary
        LM_preferredcontent_score = [content_ranking[x] for x in LM_database['Content Type']]
        # Now we need to flip the score to rank preferred content more highly
        flipped_preference_score = [round(((10 - x) * 0.1) + 0.1, 1) for x in LM_preferredcontent_score]

        # Capture the users preferred media type and score LMs accordingly. In the case that the learner's preference is No-preference we won't use preferred_media as a factor in the fitness function
        preferred_media = profile_database['preferred_media'][student_profile_id]

        # Decide where the student's preferred_media matches the LM media type
        LM_media_match = [1 if x == preferred_media else 0 for x in LM_database['Engagement Type']]

        LM_overall_preference_score = [0]*len(LM_media_match)
        if preferred_media == 'no_preference':
            LM_overall_preference_score = flipped_preference_score
        else:
            for index, value in enumerate(LM_media_match):
                LM_overall_preference_score[index] = (LM_media_match[index] + flipped_preference_score[index]) / 2

        LM_titles = LM_database['Title']

        # Next, calculate the difficulty matching score of each LM. We do this by looking at the KNs covered by each LM, the difficulty of these LMs,
        # and the learner cognitive level of each KN. We then average according to how many of these KNs are matched by the LM.

        # Create a dictionary that lets us look up KNs and determine the learner cognitive level
        KN_cog_level_dict = dict(zip(KNs, cog_levels_list))

        # Define the many-to-many relationship between LMs and KNs.
        LM_KNs_Covered = LM_database['KNs Covered']

        def max_non_goal_kns_covered(KS_names, LM_KNs_Covered):
            """
            Finds the maximum number of non-goal KNs covered by a single LM.

            Args:
                KS_names: A list of goal KN names (strings).
                LM_KNs_Covered: A list of lists or Series, where each inner list contains
                               the KN names (strings) covered by a specific LM.

            Returns:
                The maximum number of non-goal KNs covered by a single LM (integer).
                Returns 0 if LM_KNs_Covered is empty or if no non-goal KNs are found.
            """

            max_non_goal_count = 0

            if LM_KNs_Covered.empty: # Check for empty list
                return 0

            for kns_covered_by_lm in LM_KNs_Covered:
                if isinstance(kns_covered_by_lm, pd.Series): # Check if it is a pandas series
                    kns_covered_by_lm = kns_covered_by_lm.to_numpy() # convert to numpy array

                non_goal_count = 0
                for kn in kns_covered_by_lm:
                    if kn not in KS_names:
                        non_goal_count += 1
                max_non_goal_count = max(max_non_goal_count, non_goal_count)

            return max_non_goal_count


        # Example Usage (assuming KS_names and LM_KNs_Covered are defined as you described):
        max_non_goals = max_non_goal_kns_covered(KS_names, LM_KNs_Covered)

        #Identify the most KNs covered by any individual LM
        average_segmenting_max = max_non_goal_kns_covered([], LM_KNs_Covered)
        # This function calculates the difficulty matching score of LMs to learner CLs. In the case that a LM only covers 1
        #
        def calculate_matching_scores(LM_difficulty_int, LM_KNs_Covered, KN_cog_level_dict):
            """
            This function calculates the difficulty matching score of LMs to learner CLs. for the topics covered by the LM.
            :param LM_difficulty_int: List of LM difficulties
            :param LM_KNs_Covered: List of Lists of KNs covered by LMs
            :param KN_cog_level_dict: Composed of integers and strings where strings are names of KNs
            :return: list of scores corresponding to the number of LMs.
            """
            scores = []
            for i, lms in enumerate(LM_KNs_Covered):
                lm_difficulty = LM_difficulty_int[i]
                matches = 0
                for kn in lms:
                    if KN_cog_level_dict.get(kn) == lm_difficulty:
                        matches += 1
                score = matches / len(lms) if lms else 0
                scores.append(score)
            return scores

        def calculate_kn_coverage(personalized_learning_path, LM_KNs_Covered, KS_names):
            """Calculates total covering goals and non-goals using NumPy."""

            personalized_learning_path = np.array(personalized_learning_path, dtype=bool)

            covered_kns_list = [LM_KNs_Covered[i] for i, selected in enumerate(personalized_learning_path) if selected]

            if not covered_kns_list:  # Check if the list is empty
                return 0, 0

            covered_kns = np.concatenate(covered_kns_list)

            is_goal = np.isin(covered_kns, KS_names)

            total_covering_goals = np.sum(is_goal)
            total_covering_non_goals = len(covered_kns) - total_covering_goals

            return total_covering_goals, total_covering_non_goals

        def normalize_segmenting(average_segmenting_principle, average_segmenting_max, num_LMs):
            """Normalizes segmenting score, handling the all-1-KNs case."""

            if average_segmenting_max == 0:  # Handle max = 0 case (no KNs covered at all)
                if average_segmenting_principle == 1:
                    return 1.0  # Perfect score if max is 0 and average is 1
                else:
                    return 0.0  # Error if max is 0 and average is not 1

            if average_segmenting_max == 1 and num_LMs > 0:  # All LMs cover only 1 KN
                return 1.0  # Perfect score in this many-to-one scenario

            normalized_segmenting = 1.0 - ((average_segmenting_principle - 1.0) / (average_segmenting_max - 1.0))

            normalized_segmenting = max(0.0, min(1.0, normalized_segmenting))  # Clip to [0, 1] (Good practice)

            return normalized_segmenting

        #Calculate the max average Multiple Document Integration Principle Score
        personalized_learning_path = np.ones(len(LM_database), dtype=int)  # Creates an array of 1s (integers)
        total_covering_goals, total_covering_non_goals = calculate_kn_coverage(personalized_learning_path, LM_KNs_Covered, KS_names)
        max_average_MDIP = total_covering_goals/ (len(KS_names))

        # Create a boolean matrix where rows are learning path items and columns are KS_names to speed up topic balance calculations
        # True indicates that the KS is covered by the learning path item
        max_len = max(len(lst) for lst in LM_KNs_Covered)  # find the largest list in LM_KNs_Covered
        covered_matrix = np.zeros((len(LM_KNs_Covered), len(KS_names)), dtype=bool)
        for i, kn_list in enumerate(LM_KNs_Covered):
            for j, kn in enumerate(KS_names):
                if kn in kn_list:
                    covered_matrix[i, j] = True

        def check_kn_coverage(solution, LM_KNs_Covered, KS_names):
            """Checks if all KNs are covered by at least one LM in the solution."""

            num_ks = len(KS_names)
            covered_kns = set()  # Use a set for efficient checking of covered KNs

            for i, lm_included in enumerate(solution):
                if lm_included == 1:  # If the LM is included in the solution
                    for kn in LM_KNs_Covered[i]:  # Iterate through KNs covered by this LM
                        covered_kns.add(kn)

            return len(covered_kns) == num_ks  # Return True if all KNs are covered

        def calculate_balanced_cover(personalized_learning_path, LM_KNs_Covered, KS_names, total_covering_goals):
            """Calculates balanced cover using NumPy for efficiency."""
            personalized_learning_path = np.array(personalized_learning_path)  # Convert to NumPy array
            KS_names = np.array(KS_names)  # Convert to NumPy array
            num_kn = len(KS_names)
            covering_counts = np.zeros(num_kn)
            # Efficiently count coverings using boolean indexing and summing
            active_path_indices = np.where(personalized_learning_path == 1)[0]
            #covering_counts is an array composed of integers that represent how many LMs cover each KN in KS
            covering_counts = np.sum(covered_matrix[active_path_indices], axis=0)

            # Calculate balanced cover using NumPy's vectorized operations
            ideal_cover = total_covering_goals / num_kn
            balanced_cover_total = np.sum(np.abs(covering_counts - ideal_cover))

            average_balanced_cover = balanced_cover_total / num_kn
            return average_balanced_cover

        # Calculate the matching scores for all LMs
        matching_scores = calculate_matching_scores(LM_difficulty_int, LM_KNs_Covered, KN_cog_level_dict)


        correlation_graph = False

        if correlation_graph:
            data = {
                'LM Difficulty': LM_difficulty_int,
                'LM_CTML Score': lm_CTML_score,
                'LM_Duration': lm_time_taken,
                'content_type': LM_database['Content Type'],
                'media_type': LM_database['Engagement Type']
            }

            df = pd.DataFrame(data)

            one_hot_encoder = OneHotEncoder()

            # Encode preferred content type
            content_type_encoded = one_hot_encoder.fit_transform(df[['content_type']]).toarray()
            content_type_labels = one_hot_encoder.categories_[0]
            content_type_df = pd.DataFrame(content_type_encoded, columns=content_type_labels)

            # Encode preferred media type
            media_type_encoded = one_hot_encoder.fit_transform(df[['media_type']]).toarray()
            media_type_labels = one_hot_encoder.categories_[0]
            media_type_df = pd.DataFrame(media_type_encoded, columns=media_type_labels)


            # Combine encoded variables with the original data
            df_encoded = pd.concat([df.drop(['content_type', 'media_type'], axis=1), content_type_df, media_type_df], axis=1)

            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            corr_matrix = df_encoded.corr(method='spearman')
            print(corr_matrix)

            # Function to format the annotations
            def format_annotation(val):
                if val < 0:
                    return f'({-val:.2f})'
                return f'{val:.2f}'

            annot = corr_matrix.applymap(format_annotation)

            plt.figure(figsize=(12, 10))
            heatmap = sns.heatmap(corr_matrix, annot=annot, fmt="", cmap="coolwarm_r", cbar=True)

            # Rotate the x-axis labels for better readability
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')

            # Format color bar tick labels
            colorbar = heatmap.collections[0].colorbar
            tick_labels = colorbar.get_ticks()
            formatted_tick_labels = [format_annotation(tick) if tick < 0 else f'{tick:.2f}' for tick in tick_labels]
            colorbar.set_ticks(tick_labels)
            colorbar.set_ticklabels(formatted_tick_labels)

            # Save the plot as an image

            # Save the plot as an image
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to ensure nothing is clipped
            # Save the plot as an image
            plt.savefig('correlation_matrix.png', dpi=300)

        # Show the plot

        run_GA = True
        if run_GA:
            def generate_initial_population(population_size, num_genes, inclusion_probability):
                population = np.zeros((population_size, num_genes), dtype=int)  # Pre-allocate

                i = 0
                while i < population_size:
                    solution = np.random.choice([0, 1], num_genes, p=[1 - inclusion_probability, inclusion_probability])
                    num_LMs = np.sum(solution)  # More efficient way to count 1s
                    if num_LMs >= 2:
                        population[i] = solution  # Assign directly to the pre-allocated array
                        i += 1

                print("Initial population is complete")
                return population

            def solve_set_cover_ilp(KS_names, LM_KNs_Covered, lm_time_taken, Rubric_max_time):
                """Solves set cover using Integer Linear Programming."""

                num_lms = len(LM_KNs_Covered)
                num_ks = len(KS_names)

                # Create the problem
                prob = LpProblem("SetCover", LpMinimize)

                # Decision variables (1 if LM is selected, 0 otherwise)
                x = [LpVariable(f"x_{i}", 0, 1, LpInteger) for i in range(num_lms)]

                # Objective function (minimize the number of selected LMs)
                prob += lpSum(x)

                # Constraint: Each KN must be covered
                for j in range(num_ks):
                    covered_by = []
                    for i in range(num_lms):
                        if KS_names[j] in LM_KNs_Covered[i]:
                            covered_by.append(x[i])
                    prob += lpSum(covered_by) >= 1

                # Constraint: total time must be less than or equal to the Rubric Max Time.
                prob += lpSum([x[i] * lm_time_taken[i] for i in range(num_lms)]) <= Rubric_max_time

                # Solve the problem
                prob.solve()

                # Extract the solution
                if prob.status == LpStatusOptimal:
                    selected_lms = [i for i in range(num_lms) if value(x[i]) == 1]
                    return "There is a solution to the set cover problem"
                else:
                    return "No solution to set cover problem"  # No solution found


            def generate_valid_initial_population(population_size, num_genes, KS_names, LM_KNs_Covered, lm_time_taken,
                                                  Rubric_min_time, Rubric_max_time, max_failed_attempts=10000):

                population = np.zeros((population_size, num_genes), dtype=int)
                valid_count = 0
                failed_attempts = 0

                while valid_count < population_size:
                    if failed_attempts >= max_failed_attempts:
                        print(
                            f"Maximum failed attempts reached ({max_failed_attempts}). Time constraints dropped. Generating remaining population with set cover only.")

                        while valid_count < population_size:  # fill the rest of the population.
                            chromosome = np.zeros(num_genes, dtype=int)
                            uncovered_kn = set(KS_names)

                            while uncovered_kn:
                                potential_lms = []
                                for lm_index, covered_kn in enumerate(LM_KNs_Covered):
                                    if any(kn in covered_kn for kn in uncovered_kn):
                                        potential_lms.append(lm_index)

                                if not potential_lms:
                                    break

                                lm_to_add = random.choice(potential_lms)
                                chromosome[lm_to_add] = 1
                                uncovered_kn -= set(LM_KNs_Covered[lm_to_add])

                            if uncovered_kn:  # if set cover failed.
                                continue  # try again.

                            # Fill remaining LMs randomly
                            for i in range(num_genes):
                                if chromosome[i] == 0:
                                    chromosome[i] = random.choice([0, 1])

                            population[valid_count] = chromosome
                            valid_count += 1
                        break  # stop the outer loop.

                    chromosome = np.zeros(num_genes, dtype=int)
                    uncovered_kn = set(KS_names)

                    # Randomly choose LMs that cover KNs until set cover is achieved
                    while uncovered_kn:
                        potential_lms = []
                        for lm_index, covered_kn in enumerate(LM_KNs_Covered):
                            if any(kn in covered_kn for kn in uncovered_kn) and lm_time_taken[lm_index] <= Rubric_max_time:
                                potential_lms.append(lm_index)

                        if not potential_lms:
                            break

                        lm_to_add = random.choice(potential_lms)
                        chromosome[lm_to_add] = 1
                        uncovered_kn -= set(LM_KNs_Covered[lm_to_add])

                    if uncovered_kn:
                        failed_attempts += 1
                        continue

                    total_duration = np.sum(chromosome * lm_time_taken)

                    if total_duration > Rubric_max_time:
                        failed_attempts += 1
                        continue

                    elif total_duration < Rubric_min_time:
                        available_lms = [i for i, val in enumerate(chromosome) if val == 0]
                        random.shuffle(available_lms)
                        for lm_index in available_lms:
                            if total_duration + lm_time_taken[lm_index] <= Rubric_max_time:
                                chromosome[lm_index] = 1
                                total_duration += lm_time_taken[lm_index]
                                if Rubric_min_time <= total_duration <= Rubric_max_time:
                                    population[valid_count] = chromosome
                                    valid_count += 1
                                    failed_attempts = 0
                                    break
                            else:
                                continue
                        else:
                            if Rubric_min_time <= total_duration <= Rubric_max_time:
                                population[valid_count] = chromosome
                                valid_count += 1
                                failed_attempts = 0
                    else:
                        population[valid_count] = chromosome
                        valid_count += 1
                        failed_attempts = 0

                print("Population successfully created")
                return population


            def fitness_func(ga_instance, solution, solution_idx):
                solution = np.round(solution).astype(int)  # Round and cast to int
                included_indices = np.where(solution == 1)[0]
                num_LMs = len(included_indices)

                # Check for no LMs
                if num_LMs < 2:
                    return [1, 1, 1, 1, 1]

                difficulty_matching_average = np.sum(solution*matching_scores) / num_LMs
                difficulty_matching_average = max(0.0, min(1.0, difficulty_matching_average))

                CTML_average = np.sum(solution*lm_CTML_score) / num_LMs
                CTML_average_normalized = (CTML_average - 1) / (4.0 - 1.0)
                CTML_average_normalized = max(0.0, min(1.0, CTML_average_normalized))

                media_matching_average = np.sum(solution*LM_overall_preference_score) / num_LMs
                media_matching_average = max(0.0, min(1.0, media_matching_average))

                total_duration = np.sum(solution*lm_time_taken)

                #slightly adjust the rubric min and max times
                adjusted_rubric_min_time = Rubric_min_time - 1
                adjusted_rubric_max_time = Rubric_max_time + 1

                if total_duration >= min_time:
                    min_time_compliance = 1.0
                elif adjusted_rubric_min_time <= total_duration < min_time:  # Now within rubric range but below min_time
                    min_time_compliance = 1.0 - (abs(min_time - total_duration) / abs(adjusted_rubric_min_time - min_time)) if adjusted_rubric_min_time != min_time else 0.0
                else: min_time_compliance = 0

                if total_duration <= max_time:
                    max_time_compliance = 1.0
                elif max_time < total_duration <= adjusted_rubric_max_time:  # Now within rubric range but below above max_time
                    max_time_compliance = 1.0 - abs(max_time - total_duration) / abs(adjusted_rubric_max_time - max_time) if adjusted_rubric_max_time != max_time else 0.0
                else: max_time_compliance = 0

                if min_time_compliance < 1.0: max_time_compliance = min_time_compliance
                elif max_time_compliance < 1.0: min_time_compliance = max_time_compliance

                min_time_compliance = max(0.0, min(1.0, min_time_compliance))
                max_time_compliance = max(0.0, min(1.0, max_time_compliance))

                # Begin section which examines relationships between LMs and KNs
                total_covering_goals, total_covering_non_goals = calculate_kn_coverage(solution, LM_KNs_Covered, KS_names)

                average_balanced_cover = calculate_balanced_cover(solution, LM_KNs_Covered, KS_names, total_covering_goals)
                if average_balanced_cover >= max_average_balanced_cover: normalized_average_balanced_cover = 0
                else: normalized_average_balanced_cover = 1 - (average_balanced_cover / max_average_balanced_cover)

                normalized_average_balanced_cover = max(0.0, min(1.0, normalized_average_balanced_cover))

                average_coherence = (total_covering_non_goals / num_LMs)

                if max_non_goals == 0:  # Handle the case where max_non_goals is 0
                    normalized_average_coherence = 1.0 if average_coherence == 0 else 0.0  # Perfect coherence if no non-goals
                elif average_coherence > Rubric_max_coherence:
                    normalized_average_coherence = 0 # use if coherence exceeds the Rubrics standards.
                else:
                    normalized_average_coherence = 1.0 - (average_coherence / Rubric_max_coherence)

                normalized_average_coherence = max(0.0, min(1.0, normalized_average_coherence))

                MDIP_average = total_covering_goals / len(KS_names)
                normalized_MDIP = MDIP_average / Rubric_max_MDIP
                normalized_MDIP = max(0.0, min(1.0, normalized_MDIP))

                average_segmenting = (total_covering_non_goals + total_covering_goals) / num_LMs
                normalized_segmenting = normalize_segmenting(average_segmenting, Rubric_max_segmenting, num_LMs)

                average_cohesiveness = calculate_cohesiveness(solution, LM_database)

                rubric_scores = {
                    "LM_Difficulty_Matching": 1,
                    "CTML_Principle": 1,
                    "media_matching": 1,
                    "time_interval_score": 1,
                    "coherence_principle": 1,
                    "segmenting_principle": 1,
                    "balance": 1,
                    "cohesiveness": 1,
                    "MDIP": 1,
                    "Rubric Average": 1
                }

                if difficulty_matching_average >= 0.75:
                    rubric_scores["LM_Difficulty_Matching"] = 4
                elif difficulty_matching_average >= 0.5:
                    rubric_scores["LM_Difficulty_Matching"] = 3
                elif difficulty_matching_average >= 0.25:
                    rubric_scores["LM_Difficulty_Matching"] = 2

                if CTML_average >= 3.25:
                    rubric_scores["CTML_Principle"] = 4
                elif CTML_average >= 2.5:
                    rubric_scores["CTML_Principle"] = 3
                elif CTML_average >= 1.75:
                    rubric_scores["CTML_Principle"] = 2

                if media_matching_average >= 0.75:
                    rubric_scores["media_matching"] = 4
                elif media_matching_average >= 0.5:
                    rubric_scores["media_matching"] = 3
                elif media_matching_average >= 0.25:
                    rubric_scores["media_matching"] = 2

                if min_time <= total_duration <= max_time:
                    rubric_scores["time_interval_score"] = 4
                else:
                    if total_duration < min_time:
                        if 0 < abs(total_duration - min_time) / min_time <= 0.1:
                            rubric_scores["time_interval_score"] = 3
                        elif 0.1 < abs(total_duration - min_time) / min_time <= 0.2:
                            rubric_scores["time_interval_score"] = 2
                    if total_duration > max_time:
                        if 0 < abs(total_duration - max_time) / max_time <= 0.1:
                            rubric_scores["time_interval_score"] = 3
                        elif 0.1 < abs(total_duration - max_time) / max_time <= 0.2:
                            rubric_scores["time_interval_score"] = 2

                if average_coherence <= 0.25:
                    rubric_scores["coherence_principle"] = 4
                elif average_coherence <= 0.5:
                    rubric_scores["coherence_principle"] = 3
                elif average_coherence <= 1.0:
                    rubric_scores["coherence_principle"] = 2

                if average_segmenting <= 2:
                    rubric_scores["segmenting_principle"] = 4
                elif average_segmenting <= 3:
                    rubric_scores["segmenting_principle"] = 3
                elif average_segmenting <= 4:
                    rubric_scores["segmenting_principle"] = 2

                if average_cohesiveness >= 0.75:
                    rubric_scores["cohesiveness"] = 4
                elif average_cohesiveness >= 0.5:
                    rubric_scores["cohesiveness"] = 3
                elif average_cohesiveness >= 0.25:
                    rubric_scores["cohesiveness"] = 2

                if average_balanced_cover <= 1:
                    rubric_scores["balance"] = 4
                elif average_balanced_cover <= 2.9:
                    rubric_scores["balance"] = 3
                elif average_balanced_cover <= 4.9:
                    rubric_scores["balance"] = 2

                if MDIP_average >= 4:
                    rubric_scores["MDIP"] = 4
                elif MDIP_average >= 3:
                    rubric_scores["MDIP"] = 3
                elif MDIP_average >= 2:
                    rubric_scores["MDIP"] = 2

                rubric_scores["Rubric Average"] = sum(rubric_scores.values()) / (len(rubric_scores) - 1)

                # Check for complete set cover and satisfaction of time restrictions - otherwise put set_cover to 1
                set_cover = 4
                kn_coverage = {kn: False for kn in KS_names}

                for i, lm_active in enumerate(solution):
                    if lm_active:
                        for kn in LM_KNs_Covered[i]:
                            if kn in kn_coverage:
                                kn_coverage[kn] = True

                if not all(kn_coverage.values()):
                    set_cover = 1 # Not a valid set cover or time constraints violated


                LM_compliance_score = ((difficulty_matching_average + CTML_average_normalized + media_matching_average + average_cohesiveness) / 4) * 3 + 1

                KN_compliance_score = ((normalized_segmenting + normalized_MDIP + normalized_average_coherence + normalized_average_balanced_cover) / 4) * 3 + 1

                time_compliance = ((min_time_compliance + max_time_compliance) / 2 ) * 3 + 1

                return [rubric_scores["Rubric Average"], LM_compliance_score, KN_compliance_score, time_compliance, set_cover]


            gene_space = {'low': 0, 'high': 1}  # Each gene is either 0 or 1

            # Initial Exhaustive Search
            # sol_per_pop = [50, 100]
            # num_generations = [50, 100]
            # parent_selection_type = ["nsga2", "tournament_nsga2"]
            # mutation_type = ["swap", "random"]
            # crossover_type = ["single_point", "two_points"]
            # mutation_probability = [0.1, 0.3, 0.5]
            # crossover_probability = [0.1, 0.3, 0.5]
            # num_parents_mating = [10, 25]

            # Second Exhaustive Search for specialized Initial Population
            # sol_per_pop = [50, 100, 150, 200, 250]
            # num_generations = [50, 100, 150, 200, 250]
            # parent_selection_type = ["nsga2"]
            # mutation_type = ["swap"]
            # crossover_type = ["single_point"]
            # mutation_probability = [0.1, 0.3, 0.5]
            # crossover_probability = [0.5]
            # num_parents_mating = [10]

            # # Second Exhaustive Search for random initial population
            # sol_per_pop = [50, 100, 150, 200, 250]
            # num_generations = [50, 100, 150, 200, 250]
            # parent_selection_type = ["nsga2"]
            # mutation_type = ["swap"]
            # crossover_type = ["two_points"]
            # mutation_probability = [0.1, 0.3, 0.5]
            # crossover_probability = [0.3]
            # num_parents_mating = [25]

            # quick run
            sol_per_pop = [100]
            num_generations = [100]
            parent_selection_type = ["nsga2"]
            mutation_type = ["swap"]
            crossover_type = ["single_point"]
            mutation_probability = [0.3]
            crossover_probability = [0.5]
            num_parents_mating = [10]

            parameter_combinations = list(itertools.product(
                sol_per_pop,
                num_generations,
                parent_selection_type,
                mutation_type,
                crossover_type,
                mutation_probability,
                crossover_probability,
                num_parents_mating
            ))

            # num_generations = 50
            # num_parents_mating = 25
            # sol_per_pop = 100
            # parent_selection_type = "nsga2"
            # mutation_type = "random"
            # crossover_type = "two_points"
            # mutation_probability = 0.3
            # crossover_probability = 0.3
            keep_elitism = 2

            num_genes = len(LM_database)

            # Define Rubric Parameters that will be used in the GA Fitness Function
            Rubric_max_time = math.ceil(max_time*1.2)
            Rubric_min_time = math.floor(min_time * 0.8)
            Rubric_max_coherence = 1.1
            max_average_balanced_cover = 5
            Rubric_max_MDIP = 4
            Rubric_max_segmenting = 5


            for combination in parameter_combinations:
                pop_size, generations, parent_type, mut_type, cross_type, mut_prob, cross_prob, parents_mating = combination

                ga_instance = pygad.GA(num_generations=generations,
                                       num_parents_mating=parents_mating,
                                       sol_per_pop=pop_size,
                                       num_genes=num_genes,
                                       gene_space=gene_space,
                                       #initial_population = generate_initial_population(pop_size, num_genes, 0.5),
                                       initial_population=generate_valid_initial_population(pop_size, num_genes,
                                                                                            KS_names, LM_KNs_Covered,
                                                                                            lm_time_taken, Rubric_min_time,
                                                                                            Rubric_max_time, 1000),
                                       fitness_func=fitness_func,
                                       parent_selection_type=parent_type,
                                       mutation_type=mut_type,
                                       mutation_probability=mut_prob,
                                       crossover_type=cross_type,
                                       crossover_probability=cross_prob,
                                       keep_elitism=keep_elitism
                                       #random_seed=42
                                       )

                print(f"Running with: {combination}")
                start_time = time.time()
                ga_instance.run()
                end_time = time.time()
                elapsed_time = end_time - start_time

                #ga_instance.plot_fitness(label=['Rubric Average', 'LM compliance average', "KN compliance average", "Time Interval Compliance", "Set Cover"])

                print("Finished running GA")

                solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

                solution = np.round(solution).astype(int)  # Round and cast to int
                num_LMs = np.sum(solution)

                # Check pareto front for best solution according to rubric
                pareto_front = ga_instance.pareto_fronts
                # highest_fitness_value = -1  # Access the first element of the fitness array
                # best_chromosome_index = -1
                # LM_compliance_average = 0
                # KN_compliance_average = 0
                # Time_interval_compliance = 0
                # set_cover = 1
                size_pareto_front = len(pareto_front[0])

                for row_number, row_data in enumerate(pareto_front[0]):  # Iterate from the second row onwards
                    chromosome_index = row_data[0]
                    # fitness_value = row[1][0]
                    #
                    # if fitness_value > highest_fitness_value:
                    #     highest_fitness_value = fitness_value
                    #     best_chromosome_index = chromosome_index
                    #     LM_compliance_average = row[1][1]
                    #     KN_compliance_average = row[1][2]
                    #     Time_interval_compliance = row[1][3]
                    #     set_cover = row[1][4]
                    fitness_value = row_data[1][0]
                    LM_compliance_average = row_data[1][1]
                    KN_compliance_average = row_data[1][2]
                    Time_interval_compliance = row_data[1][3]
                    set_cover = row_data[1][4]

                    #print("Fitness value of best solution from pareto front is:", highest_fitness_value)

                    #solution = ga_instance.population[best_chromosome_index]
                    solution = ga_instance.population[chromosome_index]
                    solution = np.round(solution).astype(int)  # Round and cast to int

                    total_difficulty_score = 0
                    total_media_score = 0
                    total_CTML_score = 0
                    total_time = 0
                    count = 0
                    total_covering_non_goals = 0
                    total_covering_goals = 0

                    # Iterate through the candidate learning path and match scores
                    for i in range(len(solution)):
                        if solution[i] == 1:
                            # Add up the number of non-goal KNs covered by learning material
                            for kn in LM_KNs_Covered[i]:
                                if kn not in KS_names:
                                    total_covering_non_goals += 1
                                else:
                                    total_covering_goals += 1
                            total_media_score += LM_overall_preference_score[i]
                            total_difficulty_score += matching_scores[i]
                            total_CTML_score += lm_CTML_score[i]
                            total_time += lm_time_taken[i]
                            count += 1

                    balanced_cover_total = 0
                    for kn in KS_names:
                        num_coverings = 0
                        for i in range(len(solution)):
                            if solution[i] == 1:
                                if kn in LM_KNs_Covered[i]: num_coverings += 1
                        balanced_cover_total += abs(num_coverings - total_covering_goals / len(KS_names))
                    balanced_average = balanced_cover_total / len(KS_names)

                    average_difficulty_matching_score = total_difficulty_score / count if count > 0 else 0
                    average_media_preference_score = total_media_score / count if count > 0 else 0
                    average_CTML_score = total_CTML_score / count if count > 0 else 0
                    average_coherence = total_covering_non_goals / count if count > 0 else 0
                    multiple_document_principle_average = total_covering_goals / len(KS_names)
                    average_segmenting_principle = (total_covering_non_goals + total_covering_goals) / count if count > 0 else 0

                    included_embeddings = [LM_database['embeddings'][i] for i in range(len(LM_database)) if
                                           solution[i] == 1]

                    num_included_LMs = len(included_embeddings)
                    total_similarity = 0
                    count = 0

                    for i in range(num_included_LMs):
                        for j in range(i + 1, num_included_LMs):
                            similarity_score = util.pytorch_cos_sim(included_embeddings[i], included_embeddings[j]).item()
                            normalized_similarity = (similarity_score + 1) / 2  # Normalize similarity score
                            total_similarity += normalized_similarity
                            count += 1

                    average_cohesiveness = total_similarity / count if count > 0 else 0

                    raw_data = {
                        "Student_id": int(student_profile_id),
                        "Iteration": _+1,
                        "Size_Pareto_Front": size_pareto_front,
                        "GA Run Time": elapsed_time,
                        "Num_Generation": generations,
                        "Sol_per_pop": pop_size,
                        "Num_parents_mating": parents_mating,
                        "Parent Selection Type": parent_type,
                        "Mutation Type": mut_type,
                        "Crossover Type": cross_type,
                        "Mutation Probability": mut_prob,
                        "Crossover Probability": cross_prob,
                        "Keep Elitism": keep_elitism,
                        "Personalized Learning Path": str(solution),
                        "Total number of LMs": num_included_LMs,
                        "LM Compliance": LM_compliance_average,
                        "KN Compliance": KN_compliance_average,
                        "Time Interval Compliance": Time_interval_compliance,
                        "Set Cover Compliance": set_cover,
                        "Difficulty Average": average_difficulty_matching_score,
                        "Media Matching Average": average_media_preference_score,
                        "CTML Average": average_CTML_score,
                        "Cohesiveness Average": average_cohesiveness,
                        "Balance Average": balanced_average,
                        "PLP Duration": total_time,
                        "Coherence Average": average_coherence,
                        "Segmenting Average": average_segmenting_principle,
                        "MDIP Average": multiple_document_principle_average
                    }

                    rubric_scores = {
                        "LM_Difficulty_Matching": 1,
                        "CTML_Principle": 1,
                        "media_matching": 1,
                        "time_interval_score": 1,
                        "coherence_principle": 1,
                        "segmenting_principle": 1,
                        "balance": 1,
                        "cohesiveness": 1,
                        "MDIP": 1,
                        "Rubric Average": 1
                    }

                    if average_difficulty_matching_score >= 0.75:
                        rubric_scores["LM_Difficulty_Matching"] = 4
                    elif average_difficulty_matching_score >= 0.5:
                        rubric_scores["LM_Difficulty_Matching"] = 3
                    elif average_difficulty_matching_score >= 0.25:
                        rubric_scores["LM_Difficulty_Matching"] = 2

                    if average_CTML_score >= 3.25:
                        rubric_scores["CTML_Principle"] = 4
                    elif average_CTML_score >= 2.5:
                        rubric_scores["CTML_Principle"] = 3
                    elif average_CTML_score >= 1.75:
                        rubric_scores["CTML_Principle"] = 2

                    if average_media_preference_score >= 0.75:
                        rubric_scores["media_matching"] = 4
                    elif average_media_preference_score >= 0.5:
                        rubric_scores["media_matching"] = 3
                    elif average_media_preference_score >= 0.25:
                        rubric_scores["media_matching"] = 2

                    if min_time < total_time < max_time:
                        rubric_scores["time_interval_score"] = 4
                    else:
                        if total_time < min_time:
                            if 0 < abs(total_time - min_time) / min_time <= 0.1:
                                rubric_scores["time_interval_score"] = 3
                            elif 0.1 < abs(total_time - min_time) / min_time <= 0.2:
                                rubric_scores["time_interval_score"] = 2
                        if total_time > max_time:
                            if 0 < abs(total_time - max_time) / max_time <= 0.1:
                                rubric_scores["time_interval_score"] = 3
                            elif 0.1 < abs(total_time - max_time) / max_time <= 0.2:
                                rubric_scores["time_interval_score"] = 2

                    if average_coherence <= 0.25:
                        rubric_scores["coherence_principle"] = 4
                    elif average_coherence <= 0.5:
                        rubric_scores["coherence_principle"] = 3
                    elif average_coherence <= 1.0:
                        rubric_scores["coherence_principle"] = 2

                    if average_segmenting_principle <= 2:
                        rubric_scores["segmenting_principle"] = 4
                    elif average_segmenting_principle <= 3:
                        rubric_scores["segmenting_principle"] = 3
                    elif average_segmenting_principle <= 4:
                        rubric_scores["segmenting_principle"] = 2

                    if average_cohesiveness >= 0.75:
                        rubric_scores["cohesiveness"] = 4
                    elif average_cohesiveness >= 0.5:
                        rubric_scores["cohesiveness"] = 3
                    elif average_cohesiveness >= 0.25:
                        rubric_scores["cohesiveness"] = 2

                    if balanced_average <= 1:
                        rubric_scores["balance"] = 4
                    elif balanced_average <= 2.9:
                        rubric_scores["balance"] = 3
                    elif balanced_average <= 4.9:
                        rubric_scores["balance"] = 2

                    if multiple_document_principle_average >= 4:
                        rubric_scores["MDIP"] = 4
                    elif multiple_document_principle_average >= 3:
                        rubric_scores["MDIP"] = 3
                    elif multiple_document_principle_average >= 2:
                        rubric_scores["MDIP"] = 2

                    rubric_scores["Rubric Average"] = sum(rubric_scores.values()) / (len(rubric_scores) - 1)

                    #print("Best score for student: ", student_profile_id, " is ",  rubric_scores["Rubric Average"])
                    combined_data = {**raw_data, **rubric_scores}
                    combined_data_df = pd.DataFrame(combined_data, index=[0])
                    experiment_df = pd.concat([experiment_df, combined_data_df], ignore_index=True)


Experiment = "Experiment_Results/LM_Selection_30_iterations.csv"
experiment_df.to_csv(Experiment)
print("Finished Test")







