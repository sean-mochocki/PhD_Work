import igraph as ig
import pandas as pd
import ast
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder
import spacy
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import pygad

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

# Get Sentence Transformer model to be used to calculate cohesiveness
model = SentenceTransformer('all-mpnet-base-v2')
LM_database['embeddings'] = LM_database['Description'].apply(lambda x: model.encode(x, convert_to_tensor=True))

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
student_profile_id = 0
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

# Determine the number of candidate learning materials
num_LMs = len(LM_database)

run_test = False

if run_test:
    # Create a random personalized learning path with 0's and 1's
    personalized_learning_path = np.random.choice([0, 1], size=num_LMs, p=[0.5, 0.5])

    # In run coherence test we only include LMs that exclusively cover student goal nodes.

    # Note, we need to check all KNs to make sure that data is consistent.
    run_coherence_test = True
    if run_coherence_test:
        # Select LMs that only cover goal KNs
        for i in range(num_LMs):
            #print(f"Checking LM_KNs_Covered[{i}]: {LM_KNs_Covered[i]}")
            if any(kn in KS_names for kn in LM_KNs_Covered[i]):
                #print(LM_KNs_Covered[i])
                personalized_learning_path[i] = 1
            else:
                personalized_learning_path[i] = 0

    # Initialize variables
    total_difficulty_score = 0
    #total_media_score is the average of LM_media_match and flipped_preference_score
    total_LM_media_match = 0
    total_flipped_preference_score = 0
    total_media_score = 0
    total_CTML_score =0
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
                if kn not in KS_names: total_covering_non_goals += 1
                else: total_covering_goals += 1
            total_media_score += LM_overall_preference_score[i]
            total_LM_media_match += LM_media_match[i]
            total_flipped_preference_score += flipped_preference_score[i]
            total_difficulty_score += matching_scores[i]
            total_CTML_score += lm_CTML_score[i]
            total_time += lm_time_taken[i]
            count += 1

    # Begin Balanced Cover Section
    # number of goal Knowledge Nodes in KS
    #print("Total number of non-goal coverings", total_covering_non_goals)
    #print("Total number of goal coverings", total_covering_goals)
    #print("Number of goal KNs", len(KS_names))

    balanced_cover_total = 0

    for kn in KS_names:
        num_coverings = 0
        for i in range(len(personalized_learning_path)):
            if personalized_learning_path[i] == 1:
                if kn in LM_KNs_Covered[i]: num_coverings +=1
        #print("number of coverings for ", kn, "is:", num_coverings)
        balanced_cover_total += abs(num_coverings - total_covering_goals / len(KS_names))
        #print("balanced score ", kn, " is ", abs(num_coverings - total_covering_goals / len(KS_names)))
    #    for j in range(len(personalized_learning_path)):
    #print("Balanced Cover is: ", balanced_cover_total)
    print("Average Balanced Cover is: ", balanced_cover_total / len(KS_names))
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

    # calculate the average cohesiveness of the PLP
    # Filter the embeddings of included learning materials
    included_embeddings = [LM_database['embeddings'][i] for i in range(num_LMs) if personalized_learning_path[i] == 1]

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
    #print("Personalized Learning Path:", personalized_learning_path)
    print("Total number of LMs is:", num_included_LMs)
    print("Average Difficulty Matching Score:", average_difficulty_matching_score)
    print("Average Overall Media Matching Score", average_media_preference_score)
    print("Average LM Media Matching Score", average_LM_media_match)
    print("Average LM Preference Matching Score", average_preference)
    print("Average CTML Score", average_CTML_score)
    print("Average Cohesiveness:", average_cohesiveness)
    print("Total Time Taken", total_time)
    print("Average Coherence", average_coherence)
    print("Average Segmenting", average_segmenting_principle)
    print("Multiple Document Integration Score", multiple_document_principle_average)

#print(matching_scores)
run_GA = True
if run_GA:
    gene_space = {'low': 0, 'high': 1}  # Each gene is either 0 or 1

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

    def fitness_func(ga_instance, solution, solution_idx):
        solution = np.round(solution).astype(int)  # Round and cast to int
        included_indices = np.where(solution == 1)[0]
        num_LMs = len(included_indices)

        difficulty_matching_average = np.sum(solution*matching_scores)
        CTML_average = np.sum(solution*lm_CTML_score)
        media_matching_average = np.sum(solution*LM_overall_preference_score)
        total_duration = np.sum(solution*lm_time_taken)

        # Confirm that there is at least 1 LM
        if num_LMs > 0:
            difficulty_matching_average = difficulty_matching_average / num_LMs
            CTML_average = CTML_average / num_LMs
            media_matching_average = media_matching_average / num_LMs
        else:
            difficulty_matching_average = 0
            CTML_average = 0
            media_matching_average = 0

        # Use rubric to convert to integer scores
        LM_Difficulty_Matching = 1
        CTML_Principle = 1
        media_matching = 1
        time_interval_score = 1

        if difficulty_matching_average >= 0.75: LM_Difficulty_Matching = 4
        elif difficulty_matching_average >= 0.5: LM_Difficulty_Matching = 3
        elif difficulty_matching_average >= 0.25: LM_Difficulty_Matching = 2

        if CTML_average >= 3.25: CTML_Principle = 4
        elif CTML_average >= 2.5: CTML_Principle = 3
        elif CTML_average >= 1.75: CTML_Principle = 2

        if media_matching_average >= 0.75: media_matching = 4
        elif media_matching_average >= 0.5: media_matching = 3
        elif media_matching_average >= 0.25: media_matching = 2

        total_covering_goals, total_covering_non_goals= calculate_kn_coverage(solution, LM_KNs_Covered, KS_names)

        average_coherence = total_covering_non_goals / num_LMs if num_LMs > 0 else 0

        coherence_principle = 1
        if average_coherence <= 0.25: coherence_principle = 4
        elif average_coherence <= 0.5: coherence_principle = 3
        elif average_coherence <= 1.0: coherence_principle = 2

        multiple_document_principle_average = total_covering_goals / len(KS_names)
        average_segmenting_principle = (total_covering_non_goals + total_covering_goals) / num_LMs if num_LMs > 0 else 0

        segmenting_principle = 1
        if average_segmenting_principle <= 2: segmenting_principle = 4
        elif average_segmenting_principle <= 3: segmenting_principle = 3
        elif average_segmenting_principle <= 4: segmenting_principle = 2

        #if min_time < total_duration < max_time: time_interval_score = 1.0
        #elif total_duration < min_time:
        #    time_interval_score = total_duration - min_time
        #else: time_interval_score = max_time - total_duration
        #else: time_interval_score = -abs(total_duration - (min_time + max_time) / 2)  # Negative absolute difference from midpoint

        if min_time < total_duration < max_time: time_interval_score = 4
        else:
            if total_duration < min_time:
                if 0 < abs(total_duration - min_time) / min_time <= 0.1: time_interval_score = 3
                elif 0.1 < abs(total_duration - min_time) / min_time <= 0.2: time_interval_score = 2
                else: time_interval_score = 1
            if total_duration > max_time:
                if 0 < abs(total_duration - max_time) / max_time <= 0.1: time_interval_score = 3
                elif 0.1 < abs(total_duration - max_time) / max_time <= 0.2: time_interval_score = 2
                else: time_interval_score = 1

        return [LM_Difficulty_Matching, CTML_Principle, media_matching, time_interval_score, coherence_principle, segmenting_principle]

    # GA Parameters
    num_generations = 50
    num_parents_mating = 50
    sol_per_pop = 250
    num_genes = num_LMs


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
                           initial_population=created_seeded_population(sol_per_pop, num_genes),
                           fitness_func=fitness_func,
                           parent_selection_type='nsga2',  # Changed parent selection
                           mutation_type='scramble',  # Changed mutation type
                           mutation_probability=0.5,  # Adjust mutation probability
                           crossover_type='two_points',  # Change crossover type
                           crossover_probability=0.8  # adjust crossover probability
                           )


    ga_instance.run()
    ga_instance.plot_fitness(label=['LM Difficulty Matching', 'CTML Principle', 'Media Matching', 'Time Interval Score',
                                    'Coherence Principle', 'Segmenting Principle'])

    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

    solution = np.round(solution).astype(int)  # Round and cast to int

    num_LMs = np.sum(solution)
    #print(solution)
    difficulty_matching_average = np.sum(solution * matching_scores)
    CTML_average = np.sum(solution * lm_CTML_score) / num_LMs
    media_matching_average = np.sum(solution * LM_overall_preference_score) /num_LMs
    total_duration = np.sum(solution * lm_time_taken)

    # Confirm that there is at least 1 LM
    if num_LMs > 0:
        difficulty_matching_average = difficulty_matching_average / num_LMs
       # CTML_average = CTML_average / num_LMs
        #media_matching_average = media_matching_average / num_LMs
    else:
        difficulty_matching_average = 0
        #CTML_average = 0
        #media_matching_average = 0

    #total_duration = np.sum(solution * lm_time_taken)

    #print(solution)
    #print(f"Difficulty Matching Average based on the best solution : {difficulty_matching_average}")
    print(f"CTML Average based on the best solution : {CTML_average}")
    print(f"Media Matching Average based on the best solution : {media_matching_average}")
    print(f"Time duration average based on the best solution : {total_duration}")
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")

    # Print the personalized learning path
    # print("Personalized Learning Path:", personalized_learning_path)

    # Print the scores
    #for i, score in enumerate(matching_scores):
    #    if score != 0: print(f"LM {i+1} has a matching score of {score:.3f}")

    pareto_front = ga_instance.best_solutions
    print("The length of the pareto front is: ", len(pareto_front))


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





