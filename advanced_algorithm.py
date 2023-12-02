import pickle
import pandas as pd
from knowledge_graph import KnowledgeGraph
import os
import numpy as np
import ast
import time
import signal

# First load the saved knowledge graph
knowledge_graph_file = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures/knowledge_graph.pkl"
kg = None
with open(knowledge_graph_file, "rb") as f:
    kg = pickle.load(f)

# Load the student profile and learning objects data structures into pandas dataframes
data_structures = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures"
learning_objects_df = pd.read_csv(os.path.join(data_structures, "learning_objects.csv"))
profiles_df = pd.read_csv(os.path.join(data_structures, "profiles.csv"))

# Note, this section is copied from xcsp3_creator. This indicates refactoring, thought should be put into
# how this code might not be repeated
#Define constants for the knowledge graph
num_kn = 20
num_kp = 25
# Define the dataframe to store the experimental data
experiment_df = pd.DataFrame(columns=["Student Profile id", "Algorithm Time Limit", "KP Length", "LO Length", "KP", "solution", "LP Total Time", "LP Score", "Algorithm Run Time", "Max Time", "Max time per session"])
# # Specify the data types of the columns using the astype() method
# experiment_df = experiment_df.astype({"Student Profile id": int, "KP Indices": list, "LO Indices": list})

for student_profile_id in range(len(profiles_df)):
    print("student profile id: ", student_profile_id)

    student_start_node = profiles_df["goal1"][student_profile_id]
    student_end_node = profiles_df["goal2"][student_profile_id]

    #Pull values off of the first user profile
    max_time = int(profiles_df['max_time'][student_profile_id])*100
    time_per_session = int(profiles_df['time_per_session'][student_profile_id])*100

    # Get the first element of the column as a string
    cog_levels_str = profiles_df['cognitive_levels'][student_profile_id]

    # Evaluate the string as a Python expression and convert it into a list of integers
    # These are ranked from CL 1 to 3, where 3 is most advanced
    cog_levels_list = list(ast.literal_eval(cog_levels_str))

    # Capture the users rankings for preferred content type from 1-9, where 1 is most preferred
    research = int(profiles_df['research'][student_profile_id])
    website = int(profiles_df['website'][student_profile_id])
    discussion = int(profiles_df['discussion'][student_profile_id])
    educational = int(profiles_df['educational'][student_profile_id])
    news_article = int(profiles_df['news_article'][student_profile_id])
    diy = int(profiles_df['diy'][student_profile_id])
    lecture = int(profiles_df['lecture'][student_profile_id])
    powerpoint = int(profiles_df['powerpoint'][student_profile_id])
    textbook_excerpt = int(profiles_df['textbook_excerpt'][student_profile_id])

    # Capture the users preferred media type
    preferred_media = profiles_df['preferred_media'][student_profile_id]

    #Capture the learning object variables of interest
    LO_difficulty = learning_objects_df['Knowledge Density (Subjective)']
    # Define a function that maps strings to integers
    def difficulty_to_int(difficulty):
        if difficulty == "Easy":
            return 1
        elif difficulty == "Medium":
            return 2
        elif difficulty == "Hard":
            return 3
        else:
            return 0 # default value for unknown strings

    # Define LO_difficulty_int as integers between 1 and 3 where 1 is easy and 3 is hard
    LO_difficulty_int = LO_difficulty.map(difficulty_to_int)

    # Convert the dataframe into a list of lists to transfer to a CSP
    knowledge_nodes_covered = learning_objects_df["knowledge_node_covered"].values.tolist()

    # Next define parameters for set covering problem
    lo_time_taken = learning_objects_df['Time to Complete'].to_numpy()
    lo_time_taken = [int(x*100) for x in lo_time_taken]

    # Identify where the student cognitive level matches Learning Object Difficulty
    # Initialize an empty list to store the matches
    LO_cognitive_matches = []

    # Loop through the knowledge nodes covered by the learning objects with their indices
    for i, kn in enumerate(knowledge_nodes_covered):
        # Get the value of the cognitive level list at the same index
        cl = cog_levels_list[kn]
        # Compare the cognitive level with the difficulty level of the learning object
        if cl == LO_difficulty_int[i]:
            # If they match, append 1 to the list
            LO_cognitive_matches.append(1)
        else:
            # If they don't match, append 0 to the list
            LO_cognitive_matches.append(0)

    #Decide where the student's preferred_media matches the LO media type
    LO_media_match = [1 if x == preferred_media else 0 for x in learning_objects_df['Engagement Type']]

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

    # Create a list that scores each LO based on the dictionary
    LO_preferredcontent_score = [content_ranking[x] for x in learning_objects_df['Content Type']]
    # Now we need to flip the score to rank preferred content more highly
    flipped_score = [10 - x for x in LO_preferredcontent_score]

    #Combine these three scores so that they can be combined into the adaptivity_score
    Alpha = 20
    Beta = 5
    Zeta = 1

    consolidated_score = [Alpha*c + Beta*m + Zeta*f for c, m, f in zip(LO_cognitive_matches, LO_media_match, flipped_score)]
    knowledge_path = kg.find_random_paths(student_start_node, student_end_node, num_kp, 42)

    # For each student profile we are working with the following data structures
    # consolidated_score is the score of the LOs
    # lo_time_taken is the amount of time taken by each LO
    # kn_covered_by_lo

    # Here are the student requirements
    # max_time
    # time_per_session
    # knowledge_nodes_covered

    # Throw out all LOs that violate the time_per_session requirements
    time_violation = []
    for index, lo in enumerate(lo_time_taken):
        if lo >time_per_session: time_violation.append(index)

    lo_time_time = np.delete(lo_time_taken, time_violation)
    consolidated_score = np.delete(consolidated_score, time_violation)
    knowledge_nodes_covered = np.delete(knowledge_nodes_covered, time_violation)

    # Define a function to identify the LOs covered by each knowledge node
    def return_kn_covered_by_lo (num_kn, knowledge_nodes_covered):
        # Loop through each knowledge node and identify which LOs cover it
        kn_covered_by_lo = []
        for n in range(num_kn):
            # Initialize an empty list to store the indices of the learning objects that cover the current knowledge node
            lo_indices = []
            # Loop through each learning object
            for i in range(len(knowledge_nodes_covered)):
                # Check if the current learning object covers the current knowledge node
                if knowledge_nodes_covered[i] == n:
                    # If yes, append the index of the current learning object to the lo_indices list
                    lo_indices.append(i)
            # After the inner loop, append the lo_indices list to the kn_covered_by_lo list
            kn_covered_by_lo.append(lo_indices)
        return kn_covered_by_lo

    # Now that we've deleted LOs that violate time constraint, create the list of the LOs that cover knowledge nodes
    # Initialize an empty list to store the list of lists of kn_covered_by_lo
    kn_covered_by_lo = return_kn_covered_by_lo(num_kn, knowledge_nodes_covered)

    # Go through kn_covered_by_lo and order by consolidated score
    # Define a custom key function that returns the score from consolidated_score
    def get_score(x):
        return consolidated_score[x]

    # Sort each sublist of kn_covered_by_lo by the score in descending order
    for sublist in kn_covered_by_lo:
        sublist.sort(key=get_score, reverse=True)

    # Next iterate through the sorted kn_covered_by_lo and identify unnecessary LOs
    redundant_lo = []

    #print(kn_covered_by_lo)

    for sublist in kn_covered_by_lo:
        # Set min_time_so_far equal to the first value in the list
        min_time_so_far = lo_time_taken[sublist[0]]
        for lo_index in sublist:
            lo_time = lo_time_taken[lo_index]
            if lo_time < min_time_so_far:
                redundant_lo.append(lo_index)
                min_time_so_far = lo_time

    # The identified LOs cannot be in optimal solutions because they take longer than other LOs that cover the same KNs but have better scores
    # Delete these learning objects
    lo_time_taken = np.delete(lo_time_taken, redundant_lo)
    consolidated_score = np.delete(consolidated_score, redundant_lo)
    knowledge_nodes_covered = np.delete(knowledge_nodes_covered, redundant_lo)
    # Recalculate the knowledge nodes covered by LOs
    kn_covered_by_lo = return_kn_covered_by_lo(num_kn, knowledge_nodes_covered)

    print("Total number of LOs under consideration: ", len(lo_time_taken))
    print("Total number of LOs deleted due to time violations: ", + len(time_violation))
    print("Total number of LOs deleted due to knowledge node coverage redundancy", + len(redundant_lo))

