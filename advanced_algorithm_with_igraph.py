from contextlib import nullcontext

import igraph as ig
import pandas as pd
import os
import numpy as np
import ast
import time
from operator import itemgetter

#from google.protobuf.struct_pb2 import NULL_VALUE
from tensorflow.python.framework.test_ops import none_eager_fallback

knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/support_files/knowledge_nodes_without_commas.txt"
knowledge_nodes_edges = "/home/sean/Desktop/PhD_Work/PhD_Work/support_files/knowledge_nodes_edges_igraph.txt"

kg = ig.Graph()
#Replace this next value later with the number of vertices in the graph
num_kn = 20
node_names = []
with open(knowledge_nodes) as f:
    for line in f:
        name = line.strip()
        if name:
            node_names.append(name)

kg = ig.Graph.Read_Ncol(knowledge_nodes_edges, names = True, directed=True)

# Load the student profile and learning objects data structures into pandas dataframes
data_structures = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures"
learning_objects_df = pd.read_csv(os.path.join(data_structures, "learning_objects.csv"))
profiles_df = pd.read_csv(os.path.join(data_structures, "consolidated_profiles.csv"))

experiment_df = pd.DataFrame(columns=["Student_id", "Best_LP", "Best_AS", "LP_Time", "Alg_time", "Num_KP_Explored", "Num_total_KP"])

for student_profile_id in range(len(profiles_df)):
#for student_profile_id in range(1):
    #student_profile_id = 11
    print("student profile id: ", student_profile_id)
    # Change this once done with testing
    # student_profile_id = 3
    student_start_node = profiles_df["goal1"][student_profile_id]
    student_end_node = profiles_df["goal2"][student_profile_id]

    # Pull values off of the first user profile
    max_time = int(profiles_df['max_time'][student_profile_id]) * 100

    # Test code, delete after
    #max_time = 1100
    #print("Fake Max Time is: ", max_time)

    time_per_session = int(profiles_df['time_per_session'][student_profile_id]) * 100

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

    # Capture the learning object variables of interest
    LO_difficulty = learning_objects_df['Knowledge Density (Subjective)']
    LO_titles = learning_objects_df['Title']
    LO_concepts = learning_objects_df['Concept']

    # Define a function that maps strings to integers
    def difficulty_to_int(difficulty):
        if difficulty == "Easy":
            return 1
        elif difficulty == "Medium":
            return 2
        elif difficulty == "Hard":
            return 3
        else:
            return 0  # default value for unknown strings


    # Define LO_difficulty_int as integers between 1 and 3 where 1 is easy and 3 is hard
    LO_difficulty_int = LO_difficulty.map(difficulty_to_int)

    # Convert the dataframe into a list of lists to transfer to a CSP
    knowledge_nodes_covered = learning_objects_df["knowledge_node_covered"].values.tolist()

    # Next define parameters for set covering problem
    lo_time_taken = learning_objects_df['Time to Complete'].to_numpy()
    lo_time_taken = [int(x * 100) for x in lo_time_taken]

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

    # Decide where the student's preferred_media matches the LO media type
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
    flipped_score = [round(((10 - x) * 0.1) + 0.1, 1) for x in LO_preferredcontent_score]

    # Combine these three scores so that they can be combined into the adaptivity_score
    Alpha = 1.0
    Beta = 1.0
    Zeta = 1.0

    consolidated_score = [Alpha * c + Beta * m + Zeta * f for c, m, f in
                          zip(LO_cognitive_matches, LO_media_match, flipped_score)]

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
        if lo > time_per_session: time_violation.append(index)

    lo_time_taken = np.delete(lo_time_taken, time_violation)
    consolidated_score = np.delete(consolidated_score, time_violation)
    knowledge_nodes_covered = np.delete(knowledge_nodes_covered, time_violation)
    LO_titles = np.delete(LO_titles, time_violation)
    LO_concepts = np.delete(LO_concepts, time_violation)


    # Define a function to identify the LOs covered by each knowledge node
    def return_kn_covered_by_lo(num_kn, knowledge_nodes_covered):
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

    for sublist in kn_covered_by_lo:
        # Set min_time_so_far equal to the first value in the list
        min_time_so_far = lo_time_taken[sublist[0]]
        for i, lo_index in enumerate(sublist):
            lo_time = lo_time_taken[lo_index]
            # Skip the first value by checking the index
            if i == 0:
                continue
            # Use >= instead of > for the rest of the values
            if lo_time >= min_time_so_far:
                redundant_lo.append(lo_index)
                min_time_so_far = lo_time

    # The identified LOs cannot be in optimal solutions because they take longer than other LOs that cover the same KNs but have better scores
    # Delete these learning objects
    lo_time_taken = np.delete(lo_time_taken, redundant_lo)
    consolidated_score = np.delete(consolidated_score, redundant_lo)
    knowledge_nodes_covered = np.delete(knowledge_nodes_covered, redundant_lo)
    LO_titles = np.delete(LO_titles, redundant_lo)
    LO_concepts = np.delete(LO_concepts, redundant_lo)
    # Recalculate the knowledge nodes covered by LOs
    kn_covered_by_lo = return_kn_covered_by_lo(num_kn, knowledge_nodes_covered)

    print("Total number of LOs under consideration: ", len(lo_time_taken))
    print("Total number of LOs deleted due to time violations: ", + len(time_violation))
    print("Total number of LOs deleted due to knowledge node coverage redundancy", + len(redundant_lo))

    #Get all of the unique paths between the two student goal nodes
    paths = kg.get_all_simple_paths(str(student_start_node), str(student_end_node))
    named_paths = [kg.vs[path]["name"] for path in paths]
    print("Number of total paths = ", len(named_paths))

    path_sets = [frozenset(named_path) for named_path in named_paths]
    unique_paths = [list(s) for s in set(path_sets)]

    # New code to delete paths with identical members and different sequences
    path_dict = {} # dictionary to store paths as keys and sequences as values
    for named_path in named_paths:
      path_list = sorted(named_path) # convert path to a sorted list
      if tuple(path_list) not in path_dict: # check if the list is already in the dictionary
        path_dict[tuple(path_list)] = named_path # if not, add the list as a key and the path as a value

    final_paths = [] # list to store the final paths
    for path_list, path in path_dict.items(): # iterate over the dictionary
      final_paths.append(path) # append the first path for each set of members

    knowledge_path = [[int(x) for x in path] for path in final_paths]
    num_kp = len(knowledge_path)
    print("Number of unique knowledge paths = ", len(knowledge_path))

    start_time = time.time()

    # Define a custom key function that returns the score from consolidated_score
    def get_time(x):
        return lo_time_taken[x]


    # Sort each sublist of kn_covered_by_lo by the time_taken in ascending order (meaning that fast LOs are first)
    for sublist in kn_covered_by_lo:
        sublist.sort(key=get_time, reverse=False)

    # Record the minimum possible time for each knowledge path
    path_times = []
    for path in knowledge_path:
        path_time = 0
        for kn in path:
            path_time += lo_time_taken[kn_covered_by_lo[kn][0]]
        path_times.append(path_time)

    # print("Path times are: ", path_times)

    kp_time_violation = []
    for index, path_time in enumerate(path_times):
        if path_time > max_time:
            kp_time_violation.append(index)

    print("Number of knowledge paths with no valid solution: ", len(kp_time_violation))

    # Create a new list with only the elements that are not in kp_time_violation
    knowledge_path = [sublist for i, sublist in enumerate(knowledge_path) if i not in kp_time_violation]
    print("Number of knowledge paths of interest: ", len(knowledge_path))

    # Sort the Knowledge Paths according to the highest possible score of their individual elements
    # Sort each sublist of kn_covered_by_lo by the score in descending order
    for sublist in kn_covered_by_lo:
        sublist.sort(key=get_score, reverse=True)

    # Record the maximum possible score for each knowledge path
    path_max_score = []
    for path in knowledge_path:
        score = 0.0
        for kn in path:
            score += consolidated_score[kn_covered_by_lo[kn][0]]
        path_max_score.append(round(score, 1))
        # path_max_score.append(score)

    zipped = list(zip(path_max_score, knowledge_path))

    sorted_zipped = sorted(zipped, key=itemgetter(0), reverse=True)
    sorted_path_max_score, sorted_knowledge_path = zip(*sorted_zipped)

    sorted_path_max_score = list(sorted_path_max_score)
    sorted_knowledge_path = list(sorted_knowledge_path)

    #print("Max score of kps are: ", sorted_path_max_score)


    # Create function that performs back-tracking and forward checking COP

    # Call a search algorithm with the best knowledge path found so far

    def check_Max_score(kp, kn_coverage, lo_scores, lo_times, max_time):
        Best_LP = []
        time_taken = 0.0
        best_score = 0.0
        # First check if the maximum score is a valid LP. If so, return that LO
        for kn in kp:
            Best_LP.append(kn_coverage[kn][0])
            time_taken += lo_times[kn_coverage[kn][0]]
            best_score += lo_scores[kn_coverage[kn][0]]
        if time_taken <= max_time:
            print("Best LP is equivalent to maximal score")
            return Best_LP, time_taken, best_score
        else:
            print("Best LP is not equivalent to maximal score")
            return [], 0, 0


    def Deterministic_COP_Algorithm(kp, kn_coverage, lo_scores, lo_times, max_time, max_global_score_so_far=0):
        """
        :param kp: An array of knowledge nodes that need to be covered
        :param kn_coverage: A list of lists of which LOs covered which knowledge nodes. They are assumed to be sorted in
        descending order by adaptivity score and time taken
        :param lo_scores: An array of adaptivity scores for learning objects
        :param lo_times: An array of times for learning objects
        :param max_time: an integer values that shows the maximum desired time for a student LP
        :param best_score_so_far: This is the global value of the best score discovered so far - use this to limit search space
        :return: Best LP, time_taken, best_score
        """
        Best_LP = []
        time_taken = 0
        best_score = 0.0

        # As an additional heuristic, calculate the best possible score remaining in the LP
        max_remaining_score = []
        total_score = 0
        # total_score = sum([lo_scores[sublist[0]] for sublist in kn_coverage])
        # total_score = sum([lo_scores[sublist[0]] for sublist in kn_coverage[1:]])
        for kn in kp:
            total_score += lo_scores[kn_coverage[kn][0]]

        # for i in range(0, len(kp)):
        #     current_score = lo_scores[kn_coverage[i][0]]
        #     max_remaining_score.append(round(total_score - current_score,1))
        #     total_score -= current_score

        for kn in kp:
            current_score = lo_scores[kn_coverage[kn][0]]
            max_remaining_score.append(round(total_score - current_score, 1))
            total_score -= current_score

        def is_goal(state):
            """"
            This function asks if we have identified a valid solution
            The state is a list of learning objects chosen, the time so far, and the score so far
            """
            partial_LP, time_so_far, score_so_far = state[0], state[1], state[2]

            if len(partial_LP) != len(kp):
                return False
            if time_so_far > max_time:
                return False

            return True

        def forward_check(state, action, best_score):
            # Check if the action is consistent with the constraints
            # The action is a learning object number
            # The state is a list of selected learning objects
            # Get the index of the next node to be covered
            # This is a valid approach, because if the length is 0, 1, or 2, this is the next node to be covered
            partial_LP, time_so_far, score_so_far = state[0], state[1], state[2]

            next_node = len(partial_LP)

            # Check to see if the LO under consideration covers the next state
            if action not in kn_coverage[kp[next_node]]:
                return False

            # Check to see if the move is valid based on the time of the new action
            total_time_inside = time_so_far + lo_times[action]

            if total_time_inside > max_time:
                return False

            # Check to see if the best prospective LP can exceed the best LP discovered so far (locally and globally)
            if partial_LP:
                best_potential_score = score_so_far + max_remaining_score[len(partial_LP) - 1]
                if best_potential_score <= max_global_score_so_far:
                    return False
                if best_potential_score <= best_score:
                    return False

            return True

        def DFS(Best_LP, time_taken, best_score):
            stack = []
            # Put the initial state onto the stack
            stack.append(([], 0, 0.0))
            # The state is a tuple of the LP so far, the time_so_far, and the score_so_far

            while stack:
                state = stack.pop()
                if is_goal(state):
                    if state[2] > best_score:
                        Best_LP = state[0]
                        time_taken = state[1]
                        best_score = round(state[2], 1)
                        continue
                    continue

                action_index = len(state[0])
                # Determine the possible actions from the kn_covered_by_lo list of lists
                # actions is a list of los that cover the next node of interest in the knowledge path
                actions = list(kn_coverage[kp[action_index]])
                # Filter out invalid actions
                actions = [action for action in actions if forward_check(state, action, best_score)]

                for action in actions:
                    new_state = (
                    state[0] + [action], state[1] + lo_times[action], state[2] + round(lo_scores[action], 1))
                    stack.append(new_state)
            # return Best_LP, time_taken, best_score
            return Best_LP, time_taken, round(best_score, 1)

        # return DFS(Best_LP, time_taken, best_score)
        return DFS(Best_LP, time_taken, round(best_score, 1))


    LP, path_time, score = check_Max_score(sorted_knowledge_path[0], kn_covered_by_lo, consolidated_score,
                                           lo_time_taken, max_time)

    # If the score is 0, it means that the best score is not equivalent to the max score
    # if score == 0:

    num_knowledge_paths_explored = 0
    for path, max_score in zip(sorted_knowledge_path, sorted_path_max_score):
        print("Next max score is: ", max_score)
        if score >= max_score:
            break
        print("Searching Knowledge path: ", path)
        print("Run Deterministic COP Algorithm")
        temp_LP, temp_time, temp_score = Deterministic_COP_Algorithm(path, kn_covered_by_lo, consolidated_score,
                                                                     lo_time_taken, max_time, score)
        if temp_score > score:
            score = temp_score
            LP = temp_LP
            path_time = temp_time
        print("Best score so far is: ", score)
        num_knowledge_paths_explored = num_knowledge_paths_explored + 1

    total_time = time.time() - start_time
    print("Best Learning Path is: ", LP)
    print("The time to complete the best Learning Path is: ", path_time)
    print("The adaptivity score of the best learning path is: ", round(score, 1))
    print("The total function took: ", total_time, " seconds")

    learning_path_titles = [LO_titles[i] for i in LP]
    learning_path_concepts = [LO_concepts[i] for i in LP]

    data = {
        "Student_id": int(student_profile_id),
        "Best_LP": str(LP),
        "Best_AS": round(score, 1),
        "LP_Time": path_time,
        "Alg_time": total_time,
        "Num_KP_Explored": int(num_knowledge_paths_explored),
        "Num_total_KP": int(num_kp),
        "Learning_path_titles": str(learning_path_titles),
        "Learning_path_concepts": str(learning_path_concepts)
    }

    data = pd.DataFrame(data, index=[0])
    experiment_df = pd.concat([experiment_df, data], ignore_index=True)

Experiment = "/home/sean/Desktop/PhD_Work/PhD_Work/Experiment/LP_Experiment_Synthetic_Data.csv"
experiment_df.to_csv(Experiment)