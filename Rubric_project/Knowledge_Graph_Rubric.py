import igraph as ig
import pandas as pd
import ast
import os
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import pygad
import torch
import math
import random

knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Knowledge_Nodes.txt"
knowledge_graph_edges = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Knowledge_Graph_Edges.txt"
learner_profile = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Learner_Profile_8_Jan_2025.xlsx"
learning_materials = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Data/Learning_Materials_Base_set.xlsx"

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
# ********************************************************************

# Derive the set of learner profiles from the dataframe
profile_database = pd.read_excel(learner_profile)
profile_database['goals'] = profile_database['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

# This is the point in the code where we start solving problems for individual learners. This will be a loop in the final version
#experiment_df = pd.DataFrame(columns=["Student_id", "Personalized Learning Path", "Total number of LMs", "Difficulty Average", "Media Matching Average", "CTML Average", "Cohesiveness Average",
#                                      "Balance Average", "PLP Duration", "Coherence Average", "Segmenting Average", "MDIP Average"])
experiment_df = pd.DataFrame()

for student_profile_id in range(len(profile_database)):

    #student_profile_id = 0
    print("Student profile is: ", student_profile_id )

    goal_nodes = profile_database['goals'][student_profile_id]

    KS = []
    for goals in goal_nodes:
        if goals != 1:
            paths = KG.get_shortest_paths(goals-1, 0)
            for path in paths:
                KS.extend(path)  # Append individual nodes from each path
        #if goals != 1: KS.extend(KG.get_shortest_paths(goals-1, 0))

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
    #print(preferred_media)
    #print(LM_media_match)

    LM_overall_preference_score = [0]*len(LM_media_match)
    if preferred_media == 'no_preference':
        LM_overall_preference_score = flipped_preference_score
    else:
        for index, value in enumerate(LM_media_match):
            LM_overall_preference_score[index] = (LM_media_match[index] + flipped_preference_score[index]) / 2
        #LM_overall_preference_score =average_arrays(flipped_preference_score, LM_media_match)
    #print(LM_overall_preference_score)

    LM_titles = LM_database['Title']

    # Next, calculate the difficulty matching score of each LM. We do this by looking at the KNs covered by each LM, the difficulty of these LMs,
    # and the learner cognitive level of each KN. We then average according to how many of these KNs are matched by the LM.

    # Create a dictionary that lets us look up KNs and determine the learner cognitive level
    KN_cog_level_dict = dict(zip(KNs, cog_levels_list))

    # Define the many-to-one relationship between LMs and KNs.
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

    #print(matching_scores)
    run_GA = True
    if run_GA:
        gene_space = {'low': 0, 'high': 1}  # Each gene is either 0 or 1

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


        def generate_initial_population_seeds(population_size, num_genes, lm_time_taken, min_time, max_time, num_seeds, matching_scores, LM_overall_preference_score, LM_KNs_Covered, KS_names):
            population = np.zeros((population_size, num_genes), dtype=int)  # Pre-allocate
            lm_time_taken = np.array(lm_time_taken)
            matching_scores = np.array(matching_scores)

            if population_size <= num_seeds *3: num_seeds = population_size //3

            time_interval_seeds_count = 0

            while time_interval_seeds_count < num_seeds:
                original_indices = np.argsort(lm_time_taken)[::-1]  # sort in descending order
                sorted_times = lm_time_taken[original_indices]
                included_LM_indices = []
                probability = 0.5
                current_time = 0

                #Creating this code will be composed of deterministic and stochastic steps
                while current_time < min_time and len(sorted_times) > 0:
                    if random.random() < probability:
                        included_LM_indices.append(original_indices[0])
                        current_time += sorted_times[0]
                        sorted_times = np.delete(sorted_times, 0)
                        original_indices = np.delete(original_indices, 0)
                    else:
                        if len(sorted_times) > 0:  # Check if sorted_times is not empty
                            random_index = random.randint(0, len(sorted_times) - 1)  # get a random index
                            random_num = sorted_times[random_index]
                            if current_time + random_num <= max_time:
                                included_LM_indices.append(original_indices[random_index])  # append the original index.
                                current_time += random_num
                                sorted_times = np.delete(sorted_times, random_index)
                                original_indices = np.delete(original_indices, random_index)

                if min_time <= current_time <= max_time and len(included_LM_indices) >= 2:
                    #print("Time of initialized PLP is:", current_time)
                    for lm_index in included_LM_indices:
                        population[time_interval_seeds_count, lm_index] = 1

                    time_interval_seeds_count = time_interval_seeds_count + 1
                else: pass

            # Add random seeds where only LMs with favorable scores are included
            matching_scores_seeds_count = 0
            while matching_scores_seeds_count < num_seeds:
                chromosome = np.zeros(num_genes, dtype=int)
                for i, score in enumerate(matching_scores):
                    if score > 0 and random.random() < 0.5:
                        chromosome[i] = 1
                population[time_interval_seeds_count + matching_scores_seeds_count] = chromosome
                matching_scores_seeds_count += 1

            # Add KS_Names seed section.
            KS_names_seeds_count = 0
            while KS_names_seeds_count < num_seeds:
                chromosome = np.zeros(num_genes, dtype=int)
                for lm_index, kn_list in enumerate(LM_KNs_Covered):
                    if any(kn in kn_list for kn in KS_names) and random.random() < 0.5:
                        chromosome[lm_index] = 1 if chromosome[lm_index] == 0 else 0  # flip the gene
                population[
                    time_interval_seeds_count + matching_scores_seeds_count + KS_names_seeds_count] = chromosome
                KS_names_seeds_count += 1

            remaining_population_size = population_size - (time_interval_seeds_count + matching_scores_seeds_count + KS_names_seeds_count)

            if remaining_population_size > 0:
                random_population = np.random.choice([0, 1], size=(remaining_population_size, num_genes), p=[0.5, 0.5])
                population[time_interval_seeds_count + matching_scores_seeds_count + KS_names_seeds_count:population_size, :] = random_population

            np.random.shuffle(population)

            # Check to confirm if all KNs are covered by LMs. If not - add LMs randomly until all KNs are covered
            for chromosome in population:
                # Check coverage for each KN in KS_names
                while True:
                    kn_coverage = {kn: False for kn in KS_names}  # track coverage

                    for i, lm_active in enumerate(chromosome):
                        if lm_active:
                            for kn in LM_KNs_Covered[i]:
                                if kn in kn_coverage:
                                    kn_coverage[kn] = True

                    if all(kn_coverage.values()):  # All KNs covered, break the loop
                        break

                    # If not all KNs are covered, randomly flip LMs to fix it
                    missing_kns = [kn for kn, covered in kn_coverage.items() if not covered]

                    # Find LMs that cover the missing KNs.
                    potential_lms_to_flip = []
                    for kn in missing_kns:
                        for i, covered_kns in enumerate(LM_KNs_Covered):
                            if kn in covered_kns:
                                potential_lms_to_flip.append(i)

                    if not potential_lms_to_flip:
                        # This should rarely happen, but it's a safety check.
                        # If no LMs cover the missing KNs, raise an error or handle accordingly.
                        print("Error: No LMs cover all missing KNs. Check LM_KNs_Covered.")
                        return population  # or raise an error

                    lm_to_flip = random.choice(potential_lms_to_flip)
                    chromosome[lm_to_flip] = 1 - chromosome[lm_to_flip]  # Flip the LM

            print("initial population is complete")
            return population

        def generate_initial_population_sliding_probability(population_size, num_genes, KS_names, LM_KNs_Covered,
                                                            ratio):
            """
            Generates an initial population with sliding inclusion probabilities.

            Args:
                population_size: The number of solutions in the population.
                num_genes: The number of learning materials (genes).
                KS_names: A list of Knowledge Statement names.
                LM_KNs_Covered: A list of sets, where each set contains the KNs covered by a learning material.
                ratio: an integer describing the probability of inclusion of LMs that cover goal KNs vs non-goal KNs
            Returns:
                A NumPy array representing the initial population.
            """

            population = np.zeros((population_size, num_genes), dtype=int)
            i = 1  # Start at i = 1

            # Calculate probability increment
            probability_increment = 1.0 / (population_size - 1) if population_size > 1 else 1.0

            while i < population_size + 1:  # Iterate up to population_size + 1
                solution = np.zeros(num_genes, dtype=int)  # Initialize solution with zeros

                # Calculate base probability for non-KS LMs
                base_probability = i * probability_increment
                base_probability = min(base_probability, 1.0)  # Ensure base probability never exceeds 1.0

                for gene_index in range(num_genes):
                    covered_kn_names = LM_KNs_Covered[gene_index]
                    covers_ks = any(kn_name in KS_names for kn_name in covered_kn_names)

                    if covers_ks:
                        inclusion_prob = min(base_probability * ratio,
                                             1.0)  # KS probability (ratio times, capped at 1.0)
                    else:
                        inclusion_prob = base_probability  # non-KS probability

                    solution[gene_index] = np.random.choice([0, 1], p=[1 - inclusion_prob, inclusion_prob])

                num_LMs = np.sum(solution)
                if num_LMs >= 2:
                    population[i - 1] = solution  # adjust for i starting at 1.
                    i += 1

            print("Initial population with sliding probability is complete")
            return population

        def generate_initial_population_weighted(population_size, num_genes, inclusion_probability_non_KS,
                                                 inclusion_probability_KS, KS_names, LM_KNs_Covered):
            """
            Generates an initial population with weighted inclusion probabilities.

            Args:
                population_size: The number of solutions in the population.
                num_genes: The number of learning materials (genes).
                inclusion_probability_non_KS: Inclusion probability for LMs not covering any KS.
                inclusion_probability_KS: Inclusion probability for LMs covering at least one KS.
                KS_names: A list of Knowledge Statement names.
                LM_KNs_Covered: A list of sets, where each set contains the KNs covered by a learning material.

            Returns:
                A NumPy array representing the initial population.
            """

            population = np.zeros((population_size, num_genes), dtype=int)
            i = 0

            while i < population_size:
                solution = np.zeros(num_genes, dtype=int)  # Initialize solution with zeros

                for gene_index in range(num_genes):
                    covered_kn_names = LM_KNs_Covered[gene_index]
                    covers_ks = any(kn_name in KS_names for kn_name in covered_kn_names)

                    if covers_ks:
                        inclusion_prob = inclusion_probability_KS
                    else:
                        inclusion_prob = inclusion_probability_non_KS

                    solution[gene_index] = np.random.choice([0, 1], p=[1 - inclusion_prob, inclusion_prob])

                num_LMs = np.sum(solution)
                if num_LMs >= 2:
                    population[i] = solution
                    i += 1

            print("Initial population with weighted inclusion is complete")
            return population

        # Create an initial population composed of random chromosomes and seeds created from single-objective heuristics
        def created_seeded_population(population_size, num_genes):
            #num_seeds = 1
            # Create the initial population, then randomly choose chromosomes to replace with seeds
            initial_population = np.random.choice([0, 1], size=(population_size, num_genes), p=[0.5, 0.5])

            # Create chromosomes that are guaranteed to have optimal scores in single-objective problems
            seeds = []
            seed1 = np.zeros(num_genes, dtype=int)
            for i in range(num_genes):
                if matching_scores[i] == 1.0: seed1[i] = 1
            seeds.append(seed1)

            seed2 = np.zeros(num_genes, dtype=int)
            for i in range(num_genes):
                if lm_CTML_score[i] >= 3.25: seed2[i] = 1
            seeds.append(seed2)

            seed3 = np.zeros(num_genes, dtype=int)
            for i in range(num_genes):
                if LM_overall_preference_score[i] >= 0.75: seed3[i] = 1
            seeds.append(seed3)

            # Add seeds to the population at unique random locations
            num_seeds_to_add = min(len(seeds), population_size)
            available_indices = list(range(population_size))  # List of available indices
            for seed in seeds[:num_seeds_to_add]:
                if available_indices:  # check to make sure there are still open spots
                    random_index = np.random.choice(available_indices)  # Choose from available indices
                    initial_population[random_index] = seed
                    available_indices.remove(random_index)  # Remove the used index

            return initial_population

        # Note - the PLP function is getting caught in local minima and is having trouble converging because using the rubric is too stringent.
        # The right path forward is to normalize the categories between 0 and 1, where 1 is good and 0 is bad, and this would allow
        # The GA to explore the space more thoroughly. We need it to be the case that minor changes in the Chromosomes create measurably different PLPs.
        # If we do this and we get a pareto front, we could then grade the pareto front according to the rubric and return the front with the best
        # average metrics to the learner.

        # Consider going to single objective and returning the average Rubric score inside the fitness function.

        def fitness_func2(ga_instance, solution, solution_idx):
            solution = np.round(solution).astype(int)  # Round and cast to int
            included_indices = np.where(solution == 1)[0]
            num_LMs = len(included_indices)

            #Ensure a minimum of two Learning Materials
            if num_LMs <= 2:
                difficulty_matching_average = -1
                CTML_average_normalized = -1
                media_matching_average = -1
                max_time_compliance = -1
                min_time_compliance = -1
                normalized_average_coherence = -1
                normalized_MDIP = -1
                normalized_segmenting = -1
                normalized_average_balanced_cover = -1
                average_cohesiveness = -1
                return [difficulty_matching_average, CTML_average_normalized, media_matching_average, max_time_compliance,
                        min_time_compliance, normalized_average_coherence, normalized_MDIP, normalized_segmenting,
                        normalized_average_balanced_cover, average_cohesiveness]

            difficulty_matching_average = np.sum(solution*matching_scores) / num_LMs
            difficulty_matching_average = max(0.0, min(1.0, difficulty_matching_average))

            CTML_average = np.sum(solution*lm_CTML_score) / num_LMs
            CTML_average_normalized = (CTML_average - 1) / (4.0 - 1.0)
            CTML_average_normalized = max(0.0, min(1.0, CTML_average_normalized))

            media_matching_average = np.sum(solution*LM_overall_preference_score) / num_LMs
            media_matching_average = max(0.0, min(1.0, media_matching_average))

            total_duration = np.sum(solution*lm_time_taken)
            min_time_compliance = 0.0
            max_time_compliance = 0.0

            if min_time <= total_duration <= max_time:
                min_time_compliance = 1.0
                max_time_compliance = 1.0
            elif total_duration < min_time:  # Now within rubric range but below min_time
                min_time_compliance = 1.0 - (abs(min_time - total_duration) / abs(Rubric_min_time - min_time)) if Rubric_min_time != min_time else 0.0
                max_time_compliance = 1 # Punish the solution for being below min_time
            elif total_duration > max_time:  # Now within rubric range but above max_time
                max_time_compliance = 0.25 - abs(max_time - total_duration) / abs(Rubric_max_time - max_time) if Rubric_max_time != max_time else 0.0
                min_time_compliance = 1 # Punish the solution for exceeding max_time

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



            # Try reducing to three objectives to simplify problem
            #objective 1 - LM fitness for the learner
            objective1 = (difficulty_matching_average + CTML_average_normalized + media_matching_average + average_cohesiveness) / 4
            #objective 2 - LM coverage of the KNs
            objective2 = (normalized_segmenting + normalized_MDIP + normalized_average_coherence + normalized_average_balanced_cover) / 4
            #objective 3 - Honoring of learner time constraints
            objective3 = 2 / ((0.5 / max_time_compliance) + (0.5 / min_time_compliance)) if max_time_compliance > 0 and min_time_compliance > 0 else 0

            return [objective1, objective2, objective3]



        def fitness_func(ga_instance, solution, solution_idx):
            solution = np.round(solution).astype(int)  # Round and cast to int
            included_indices = np.where(solution == 1)[0]
            num_LMs = len(included_indices)


            # Check for complete set cover - otherwise invalidate potential solution
            kn_coverage = {kn: False for kn in KS_names}

            for i, lm_active in enumerate(solution):
                if lm_active:
                    for kn in LM_KNs_Covered[i]:
                        if kn in kn_coverage:
                            kn_coverage[kn] = True

            if not all(kn_coverage.values()):
                return [0, 0]  # Not a valid set cover

            difficulty_matching_average = np.sum(solution*matching_scores) / num_LMs
            difficulty_matching_average = max(0.0, min(1.0, difficulty_matching_average))

            CTML_average = np.sum(solution*lm_CTML_score) / num_LMs
            CTML_average_normalized = (CTML_average - 1) / (4.0 - 1.0)
            CTML_average_normalized = max(0.0, min(1.0, CTML_average_normalized))

            media_matching_average = np.sum(solution*LM_overall_preference_score) / num_LMs
            media_matching_average = max(0.0, min(1.0, media_matching_average))

            total_duration = np.sum(solution*lm_time_taken)
            min_time_compliance = 0.0
            max_time_compliance = 0.0

            # if min_time <= total_duration <= max_time:
            #     min_time_compliance = 1.0
            #     max_time_compliance = 1.0
            # elif total_duration < min_time:  # Now within rubric range but below min_time
            #     min_time_compliance = 1.0 - (abs(min_time - total_duration) / abs(Rubric_min_time - min_time)) if Rubric_min_time != min_time else 0.0
            #     max_time_compliance = 1
            # elif total_duration > max_time:  # Now within rubric range but above max_time
            #     max_time_compliance = 0.25 - abs(max_time - total_duration) / abs(Rubric_max_time - max_time) if Rubric_max_time != max_time else 0.0
            #     min_time_compliance = 1

            if min_time <= total_duration <= max_time:
                min_time_compliance = 1.0
                max_time_compliance = 1.0
            elif total_duration < min_time:
                min_time_compliance = 1.0 - (abs(min_time - total_duration) / min_time)
                max_time_compliance = min_time_compliance
            elif total_duration > max_time:  # Now within rubric range but above max_time
                max_time_compliance = 0.25 - abs(max_time - total_duration) / abs(Rubric_max_time - max_time) if Rubric_max_time != max_time else 0.0
                min_time_compliance = max_time_compliance

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

            if min_time < total_duration < max_time:
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
            #time_compliance = ((max_time_compliance * 0.5 + min_time_compliance * 0.5) * 3) + 1
            # Try reducing to three objectives to simplify problem
            #objective 1 - LM fitness for the learner
            normalized_score = (((difficulty_matching_average + CTML_average_normalized + media_matching_average +
                                average_cohesiveness + normalized_segmenting + normalized_MDIP +
                                normalized_average_coherence + normalized_average_balanced_cover + max_time_compliance + min_time_compliance) / 10)*3)+1

            #time_compliance = 2 / ((0.5 / max_time_compliance) + (0.5 / min_time_compliance)) if max_time_compliance > 0 and min_time_compliance > 0 else 0
            #if time_compliance > 0: time_compliance = (time_compliance * 3)+ 1

            return [rubric_scores["Rubric Average"], normalized_score]


            #return [difficulty_matching_average, CTML_average_normalized, media_matching_average, max_time_compliance,
            #        min_time_compliance, normalized_average_coherence, normalized_MDIP, normalized_segmenting, normalized_average_balanced_cover, average_cohesiveness]

        # GA Parameters
        # We need to add the requirement of a set cover to the genetic algorithm. All KNs in KS_names need to be covered by at least 1 LM for a solution to be valid
        # This necessitates a change to the population

        num_seeds = 10 # Indicates numbers of seeds to be included in population per category
        num_generations = 50
        num_parents_mating = 25
        sol_per_pop = 100
        num_genes = len(LM_database)
        #At 0.5 incusion_probability is strong for high time-frame students.
        inclusion_probability = 0.5
        # Consider a heuristic where we do a sweep of low and high-density chromosomes.
        inclusion_probability_non_KS = 0.05
        inclusion_probability_KS = 0.3

        # Define Rubric Parameters that will be used in the GA Fitness Function
        # Rubric Parameters Rubric says max_time by 1.2 and min time by 0.8
        #Rubric_max_time = math.ceil(max_time * 1.5)
        Rubric_max_time = global_max_time
        Rubric_min_time = math.floor(min_time * 0.5)
        Rubric_max_coherence = 1.1
        max_average_balanced_cover = 5
        Rubric_max_MDIP = 4
        Rubric_max_segmenting = 5

        # Use NGSA for multi-objective problems
        #ga_instance = pygad.GA(num_generations=num_generations,
        #                       num_parents_mating=num_parents_mating,
        #                       sol_per_pop=sol_per_pop,
        #                       num_genes=num_genes,
        #                       gene_space=gene_space,
        #                       fitness_func=fitness_func,
        #                       parent_selection_type='nsga2')

        ga_instance = pygad.GA(num_generations=num_generations,  # Increased generations
                               num_parents_mating=num_parents_mating,  # Increased parents
                               sol_per_pop=sol_per_pop,  # Increased population size
                               num_genes=num_genes,
                               gene_space=gene_space,
                               #initial_population = generate_initial_population_sliding_probability(sol_per_pop, num_genes, KS_names, LM_KNs_Covered, 1),
                               #initial_population=generate_initial_population_weighted(sol_per_pop, num_genes, inclusion_probability_non_KS, inclusion_probability_KS, KS_names, LM_KNs_Covered),
                               #initial_population=generate_initial_population(sol_per_pop, num_genes, inclusion_probability),
                               initial_population = generate_initial_population_seeds(sol_per_pop, num_genes, lm_time_taken,
                                                                                      min_time, max_time, num_seeds, matching_scores, LM_overall_preference_score, LM_KNs_Covered, KS_names),
                               fitness_func=fitness_func,
                               #parallel_processing=["process", 24],
                               #parent_selection_type='nsga2',  # Changed parent selection
                               parent_selection_type="nsga2",
                               mutation_type='random',  # Changed mutation type
                               mutation_probability=0.5,  # Adjust mutation probability
                               crossover_type='two_points',  # Change crossover type
                               crossover_probability=0.2,  # adjust crossover probability
                               keep_elitism=2
                               #save_solutions=True
                               )


        ga_instance.run()
        ga_instance.plot_fitness(label = ['Rubric Average', 'Normalized Average'])
        # filename = "student_" + str(student_profile_id) + ".png"
        #
        # experiment_dir = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Experiment_Results/"
        # filepath = os.path.join(experiment_dir, filename)
        #ga_instance.plot_fitness(label = ['Rubric Average', 'Normalized Average'], plot_type = "plot", title = "Student_" + str(student_profile_id))
        # plt.draw()
        # plt.savefig(filepath)
        # plt.close(fig)
        #ga_instance.plot_genes()
        #ga_instance.plot_new_solution_rate()
        #ga_instance.plot_fitness(label=['LM Difficulty Matching', 'CTML Principle', 'Media Matching', 'Max Time Compliance',
        #                                'Min Time Compliance', 'Normalized Average Coherence', 'Normalized MDIP', 'Normalized Segmenting',
        #                                'Normalized Balanced Cover', 'Average Cohesiveness'])
        #ga_instance.plot_new_solution_rate()
        #ga_instance.plot_fitness(["objective1", "objective2", "objective3"])
        print("Finished running GA")

        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        solution = np.round(solution).astype(int)  # Round and cast to int
        #print(solution)
        #print(f"Fitness of best solution: {solution_fitness}")  # Print the fitness value
        num_LMs = np.sum(solution)
        #print("The number of LMs is:", num_LMs)

        #difficulty_matching_average = np.sum(solution * matching_scores)
        #CTML_average = np.sum(solution * lm_CTML_score) / num_LMs
        #media_matching_average = np.sum(solution * LM_overall_preference_score) /num_LMs
        #total_duration = np.sum(solution * lm_time_taken)

        # Confirm that there is at least 1 LM
       #if num_LMs > 0:
       #     difficulty_matching_average = difficulty_matching_average / num_LMs
           # CTML_average = CTML_average / num_LMs
            #media_matching_average = media_matching_average / num_LMs
       # else:
         #   difficulty_matching_average = 0
            #CTML_average = 0
            #media_matching_average = 0

        #total_duration = np.sum(solution * lm_time_taken)

        #print(solution)
        #print(f"Difficulty Matching Average based on the best solution : {difficulty_matching_average}")
        #print(f"CTML Average based on the best solution : {CTML_average}")
        #print(f"Media Matching Average based on the best solution : {media_matching_average}")
        #print(f"Time duration average based on the best solution : {total_duration}")
        #print(f"Parameters of the best solution : {solution}")
        #print(f"Fitness value of the best solution = {solution_fitness}")

        # Print the personalized learning path
        # print("Personalized Learning Path:", personalized_learning_path)

        # Print the scores
        #for i, score in enumerate(matching_scores):
        #    if score != 0: print(f"LM {i+1} has a matching score of {score:.3f}")

        #pareto_fronts = ga_instance.pareto_fronts
        #print("The length of the pareto front is: ", len(pareto_fronts))
        # solutions = ga_instance.population
        # #
        # all_unique_solutions = []
        # for candidate_solution in solutions:
        #     candidate_solution = np.round(candidate_solution).astype(int)
        #
        #     # Check for uniqueness against the *master* list:
        #     is_unique = True
        #     for existing_solution in all_unique_solutions:
        #         if np.array_equal(candidate_solution, existing_solution):
        #             is_unique = False
        #             break
        #
        #     if is_unique:
        #         all_unique_solutions.append(candidate_solution)
        #
        # #print(len(all_unique_solutions))
        # #print(all_unique_solutions)
        #
        # print("Evaluating GA population")
        # best_solution = []
        # best_score = 0
        # best_rubric_scores = {}
        # best_raw_data = {}
        #
        # best solution
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
        #print("Average Balanced Cover is: ", balanced_average)
        # End Balanced Cover Section

        # Calculate average matching score
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
            "Personalized Learning Path": str(solution),
            "Total number of LMs": num_included_LMs,
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

        #print("Rubric average is:", rubric_scores["Rubric Average"])

        # if (rubric_scores["Rubric Average"]) >= best_score:
        #     best_score = rubric_scores["Rubric Average"]
        #     best_solution = candidate_solution
        #     best_raw_data = raw_data
        #     best_rubric_scores = rubric_scores
        #
        print("Best score for student: ", student_profile_id, " is ",  rubric_scores["Rubric Average"])
        combined_data = {**raw_data, **rubric_scores}
        combined_data_df = pd.DataFrame(combined_data, index=[0])
        experiment_df = pd.concat([experiment_df, combined_data_df], ignore_index=True)


    #print("Best Solution is:", best_solution)
    #print("The scores of the best solution is:")
    #for name, score in best_rubric_scores.items():
    #    print(f"Rubric {name}: {score}")


    run_test = False

    if run_test:
        personalized_learning_path = solution

        # In run coherence test we only include LMs that exclusively cover student goal nodes.

        # Note, we need to check all KNs to make sure that data is consistent.
        run_coherence_test = False
        if run_coherence_test:
            # Select LMs that only cover goal KNs
            for i in range(num_LMs):
                # print(f"Checking LM_KNs_Covered[{i}]: {LM_KNs_Covered[i]}")
                if any(kn in KS_names for kn in LM_KNs_Covered[i]):
                    # print(LM_KNs_Covered[i])
                    personalized_learning_path[i] = 1
                else:
                    personalized_learning_path[i] = 0

        # Initialize variables
        total_difficulty_score = 0
        # total_media_score is the average of LM_media_match and flipped_preference_score
        total_LM_media_match = 0
        total_flipped_preference_score = 0
        total_media_score = 0
        total_CTML_score = 0
        total_time = 0
        count = 0
        total_covering_non_goals = 0
        total_covering_goals = 0

        # Convert KS to a list of strings based on KNs
        # LM_KNs_Covered - a list of lists of the Knowledge Nodes covered by each Learning Material
        # Goal - iterate through all LMs and add up the number of coverings of KNs performed by LMs that are not goal nodes.

        # Iterate through the candidate learning path and match scores
        for i in range(len(personalized_learning_path)):
            if personalized_learning_path[i] == 1:
                # Add up the number of non-goal KNs covered by learning material
                for kn in LM_KNs_Covered[i]:
                    if kn not in KS_names:
                        total_covering_non_goals += 1
                    else:
                        total_covering_goals += 1
                total_media_score += LM_overall_preference_score[i]
                total_LM_media_match += LM_media_match[i]
                total_flipped_preference_score += flipped_preference_score[i]
                total_difficulty_score += matching_scores[i]
                total_CTML_score += lm_CTML_score[i]
                total_time += lm_time_taken[i]
                count += 1

        # Begin Balanced Cover Section
        # number of goal Knowledge Nodes in KS
        # print("Total number of non-goal coverings", total_covering_non_goals)
        # print("Total number of goal coverings", total_covering_goals)
        # print("Number of goal KNs", len(KS_names))

        balanced_cover_total = 0

        for kn in KS_names:
            num_coverings = 0
            for i in range(len(personalized_learning_path)):
                if personalized_learning_path[i] == 1:
                    if kn in LM_KNs_Covered[i]: num_coverings += 1
            balanced_cover_total += abs(num_coverings - total_covering_goals / len(KS_names))
        balanced_average = balanced_cover_total / len(KS_names)
        print("Average Balanced Cover is: ", balanced_average)
        # End Balanced Cover Section

        if total_time > max_time:
            print("PLP exceeds max time by", ((total_time - max_time) / max_time) * 100, "percent")
        elif total_time < min_time:
            print("PLP is less than min time by", ((min_time - total_time) / min_time) * 100, "percent")
        else:
            print("PLP is within the student time constraints.")

        # Calculate average matching score
        average_difficulty_matching_score = total_difficulty_score / count if count > 0 else 0
        average_media_preference_score = total_media_score / count if count > 0 else 0
        average_CTML_score = total_CTML_score / count if count > 0 else 0
        average_LM_media_match = total_LM_media_match / count if count > 0 else 0
        average_preference = total_flipped_preference_score / count if count > 0 else 0
        average_coherence = total_covering_non_goals / count if count > 0 else 0
        multiple_document_principle_average = total_covering_goals / len(KS_names)
        average_segmenting_principle = (total_covering_non_goals + total_covering_goals) / count if count > 0 else 0

        #print("The number of LMs is:", num_LMs)
        # calculate the average cohesiveness of the PLP
        # Filter the embeddings of included learning materials
        included_embeddings = [LM_database['embeddings'][i] for i in range(len(LM_database)) if
                               personalized_learning_path[i] == 1]

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

        # End section to calculate cohesiveness

        # Print the personalized learning path and the average matching score
        print("Personalized Learning Path:", personalized_learning_path)
        print("Total number of LMs is:", num_included_LMs)
        print("Average Difficulty Matching Score:", average_difficulty_matching_score)
        print("Average Overall Media Matching Score", average_media_preference_score)
        print("Average CTML Score", average_CTML_score)
        print("Average Cohesiveness:", average_cohesiveness)
        print("Total Time Taken", total_time)
        print("Average Coherence", average_coherence)
        print("Average Segmenting", average_segmenting_principle)
        print("Multiple Document Integration Score", multiple_document_principle_average)

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

        if average_difficulty_matching_score >= 0.75: rubric_scores["LM_Difficulty_Matching"] = 4
        elif average_difficulty_matching_score >= 0.5: rubric_scores["LM_Difficulty_Matching"] = 3
        elif average_difficulty_matching_score >= 0.25: rubric_scores["LM_Difficulty_Matching"] = 2

        if average_CTML_score >= 3.25: rubric_scores["CTML_Principle"] = 4
        elif average_CTML_score >= 2.5: rubric_scores["CTML_Principle"] = 3
        elif average_CTML_score >= 1.75: rubric_scores["CTML_Principle"] = 2

        if average_media_preference_score >= 0.75: rubric_scores["media_matching"] = 4
        elif average_media_preference_score >= 0.5: rubric_scores["media_matching"] = 3
        elif average_media_preference_score >= 0.25: rubric_scores["media_matching"] = 2

        if min_time < total_time < max_time: rubric_scores["time_interval_score"] = 4
        else:
            if total_time < min_time:
                if 0 < abs(total_time - min_time) / min_time <= 0.1: rubric_scores["time_interval_score"] = 3
                elif 0.1 < abs(total_time - min_time) / min_time <= 0.2: rubric_scores["time_interval_score"] = 2
            if total_time > max_time:
                if 0 < abs(total_time - max_time) / max_time <= 0.1: rubric_scores["time_interval_score"] = 3
                elif 0.1 < abs(total_time - max_time) / max_time <= 0.2: rubric_scores["time_interval_score"] = 2

        if average_coherence <= 0.25: rubric_scores["coherence_principle"] = 4
        elif average_coherence <= 0.5: rubric_scores["coherence_principle"]  = 3
        elif average_coherence <= 1.0: rubric_scores["coherence_principle"]  = 2

        if average_segmenting_principle <= 2: rubric_scores["segmenting_principle"]  = 4
        elif average_segmenting_principle <= 3: rubric_scores["segmenting_principle"]  = 3
        elif average_segmenting_principle <= 4: rubric_scores["segmenting_principle"]  = 2

        if average_cohesiveness >= 0.75: rubric_scores["cohesiveness"]  = 4
        elif average_cohesiveness >= 0.5: rubric_scores["cohesiveness"] = 3
        elif average_cohesiveness >= 0.25: rubric_scores["cohesiveness"] = 2

        if balanced_average <= 1: rubric_scores["balance"] = 4
        elif balanced_average <= 2.9: rubric_scores["balance"] = 3
        elif balanced_average <= 4.9: rubric_scores["balance"] = 2

        if multiple_document_principle_average >= 4: rubric_scores["MDIP"] = 4
        elif multiple_document_principle_average >= 3: rubric_scores["MDIP"]  = 3
        elif multiple_document_principle_average >= 2: rubric_scores["MDIP"]  = 2

        rubric_scores["Rubric Average"] = sum(rubric_scores.values()) / (len(rubric_scores) - 1)

        # Optional: Print individual rubric scores
        for name, score in rubric_scores.items():
            print(f"Rubric {name}: {score}")

        data = {
            "Student_id": int(student_profile_id),
            "Personalized Learning Path": str(personalized_learning_path),
            "Total number of LMs": num_included_LMs,
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

        data = pd.DataFrame(data, index=[0])
        experiment_df = pd.concat([experiment_df, data], ignore_index=True)
        Experiment = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Experiment_Results/GA_Results.csv"
        experiment_df.to_csv(Experiment)

Experiment = "/home/sean/Desktop/PhD_Work/PhD_Work/Rubric_project/Experiment_Results/GA_Results.csv"
experiment_df.to_csv(Experiment)

### End Test ###########################

        #

        # # Use rubric to convert to integer scores
        # LM_Difficulty_Matching = 1
        # CTML_Principle = 1
        # media_matching = 1
        # time_interval_score = 1
        #
        # if difficulty_matching_average >= 0.75: LM_Difficulty_Matching = 4
        # elif difficulty_matching_average >= 0.5: LM_Difficulty_Matching = 3
        # elif difficulty_matching_average >= 0.25: LM_Difficulty_Matching = 2
        #
        # if CTML_average >= 3.25: CTML_Principle = 4
        # elif CTML_average >= 2.5: CTML_Principle = 3
        # elif CTML_average >= 1.75: CTML_Principle = 2
        #
        # if media_matching_average >= 0.75: media_matching = 4
        # elif media_matching_average >= 0.5: media_matching = 3
        # elif media_matching_average >= 0.25: media_matching = 2

        # coherence_principle = 1
        # if average_coherence <= 0.25: coherence_principle = 4
        # elif average_coherence <= 0.5: coherence_principle = 3
        # elif average_coherence <= 1.0: coherence_principle = 2

        # segmenting_principle = 1
        # if average_segmenting_principle <= 2: segmenting_principle = 4
        # elif average_segmenting_principle <= 3: segmenting_principle = 3
        # elif average_segmenting_principle <= 4: segmenting_principle = 2

        #if min_time < total_duration < max_time: time_interval_score = 1.0
        #elif total_duration < min_time:
        #    time_interval_score = total_duration - min_time
        #else: time_interval_score = max_time - total_duration
        #else: time_interval_score = -abs(total_duration - (min_time + max_time) / 2)  # Negative absolute difference from midpoint

        # if min_time < total_duration < max_time: time_interval_score = 4
        # else:
        #     if total_duration < min_time:
        #         if 0 < abs(total_duration - min_time) / min_time <= 0.1: time_interval_score = 3
        #         elif 0.1 < abs(total_duration - min_time) / min_time <= 0.2: time_interval_score = 2
        #         else: time_interval_score = 1
        #     if total_duration > max_time:
        #         if 0 < abs(total_duration - max_time) / max_time <= 0.1: time_interval_score = 3
        #         elif 0.1 < abs(total_duration - max_time) / max_time <= 0.2: time_interval_score = 2
        #         else: time_interval_score = 1
    # Example lookup
    #KN = 'Speech Processing'
    #cog_level = KN_cog_level_dict.get(KN, "KN not found")
    #print(f"Cognitive level for {KN}: {cog_level}")


    #print(LM_difficulty_int)
    # Print the students goal nodes.
    #for node in KS:
    #    print(KG.vs[node]['name'])



    # Visualize the Graph
    #layout = KG.layout_fruchterman_reingold(grid="nogrid")
    #fig, ax = plt.subplots()

    # Adjust font size and color for better readability
    #ig.plot(KG, target=ax, layout=layout, vertex_size=12,
    #        vertex_label=KG.vs['name'], edge_curved=True,
    #        vertex_label_size=5, vertex_label_color='black')  # Set label color to black

    #plt.show()
    #plt.close()






# Compute cosine distance matrix
#distance_matrix = cosine_distances(embeddings)

#k_high = 80

#silhouette_scores = []
#for num_clusters in range(2, k_high):
#    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#    cluster_labels = kmeans.fit_predict(distance_matrix)
#    silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
#    silhouette_scores.append(silhouette_avg)

# Plot the Silhouette Scores
#plt.figure(figsize=(10, 6))
#plt.plot(range(2, k_high), silhouette_scores, marker='o')
#plt.xlabel('Number of Clusters')
#plt.ylabel('Silhouette Score')
#plt.title('Silhouette Score for Optimal Number of Clusters')
#plt.grid(True)
#plt.show()
#plt.savefig('Cohesiveness_Silhoutte.png', dpi=300)
# Apply K-Means Clustering
#num_clusters = 5  # You can adjust the number of clusters
#kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#cluster_labels = kmeans.fit_predict(distance_matrix)

# Assign cluster labels to the DataFrame
#LM_database['cluster'] = cluster_labels

# Print the cluster assignments
#print(LM_database[['Description', 'cluster']])


# Initialize the minimum similarity value and the pair of least similar descriptions
#min_value = float('inf')
#max_value = -1
#least_similar_pair = (None, None)
#most_similar_pair = (None, None)
#similarity_threshold = 0.5
#above_threshold_pairs = []
#below_threshold_count = 0


#num_descriptions = len(LM_database['Description'])
#running_tally = 0

# Calculate similarities
#for i in range(num_descriptions):
#    for j in range(i + 1, num_descriptions):
#        running_tally += 1
#        embedding1 = LM_database['embeddings'][i]
#        embedding2 = LM_database['embeddings'][j]
#        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        # Cosine similarity is in the range [-1,1], we want to normalize it to [0,1]
        # Therefore add 1 and divide by 2
        # This has the impact that a score of 1 is completely similar, a score of 0.5 is unrelated (orthogonal), and a score of 0 means completely opposite
#        similarity_score = (similarity_score + 1)/2

#        if similarity_score < similarity_threshold:
#            above_threshold_pairs.append((LM_database['Description'][i],
#                                          LM_database['Description'][j],
#                                          similarity_score))

#        if similarity_score < min_value:
#            min_value = similarity_score
#            least_similar_pair = (LM_database['Description'][i], LM_database['Description'][j])

#        if similarity_score > max_value:
#            max_value = similarity_score
#            most_similar_pair = (LM_database['Description'][i], LM_database['Description'][j])


#        if similarity_score < similarity_threshold:
#            below_threshold_count += 1

#print(f"Least similar texts have a similarity score of {min_value:.2f}")
#print("Text 1:", least_similar_pair[0])
#print("Text 2:", least_similar_pair[1])
#print(f"Most similar texts have a similarity score of {max_value:.2f}")
#print("Text 1:", most_similar_pair[0])
#print("Text 2:", most_similar_pair[1])
#print(f"Total number of pairs: {running_tally}")
#print(f"Number of pairs with similarity score below {similarity_threshold}: {below_threshold_count}")

# Print results
#print(f"Pairs with similarity score above {similarity_threshold}:")
#for pair in above_threshold_pairs:
#    print(f"Text 1: {pair[0]}")
#    print(f"Text 2: {pair[1]}")
#    print(f"Similarity Score: {pair[2]:.2f}")
#    print("-" * 20)

# Convert all descriptions to Doc objects and store vectors
#LM_database['nlp_desc'] = LM_database['Description'].apply(nlp)

#min_value = float('inf')
#least_similar_pair = (None, None)
#num_descriptions = LM_database.shape[0]
#similarity_threshold = 0.75
#below_threshold_count=0
#running_tally = 0
# Calculate similarities
#for i in range(num_descriptions):
#    for j in range(i + 1, num_descriptions):
#        running_tally += 1
#        text1 = LM_database['nlp_desc'][i]
#        text2 = LM_database['nlp_desc'][j]
#        value = text1.similarity(text2)

#        if value < min_value:
#            min_value = value
#            least_similar_pair = (LM_database['Description'][i], LM_database['Description'][j])

#        if value < similarity_threshold:
#            below_threshold_count += 1

#print(f"Least similar texts have similarity score of {min_value:.2f}")
#print("Text 1:", least_similar_pair[0])
#print("Text 2:", least_similar_pair[1])
#print(f"Total number of pairs: {running_tally}")
#print(f"Number of pairs with similarity score below {similarity_threshold}: {below_threshold_count}")

# In this section we determine the CTML score of each LM. This score does not depend on individual learners so we can accomplish this task before loading the learner profile

# Capture the LM parameters based on the Cognitive Theory of Multimedia Learning





