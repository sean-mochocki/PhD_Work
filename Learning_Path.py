import pickle
import pandas as pd
from knowledge_graph import KnowledgeGraph
import os
import numpy as np
import ast
import time
import signal

# First load the saved knowledge graph
knowledge_graph_file = "/remote_home/PhD_Project/data_structures/knowledge_graph.pkl"
kg = None
with open(knowledge_graph_file, "rb") as f:
    kg = pickle.load(f)

# Load the student profile and learning objects data structures into pandas dataframes
data_structures = "/remote_home/PhD_Project/data_structures"
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

    # Initialize an empty list to store the list of lists of kn_covered_by_lo
    kn_covered_by_lo = []

    # Loop through each knowledge node and identify which LOs cover it
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

    knowledge_path = kg.find_random_paths(student_start_node , student_end_node, num_kp, 42)
    time_limit = [1, 2, 3]

    for tl in time_limit:
        print("tl: ", tl)
        for kp in knowledge_path:       
            # Define a custom exception class for timeout
            class TimeoutException(Exception):
                pass

            # Define a handler function that raises the exception
            def handler(signum, frame):
                raise TimeoutException("Time limit exceeded")

            # Register the handler for the SIGALRM signal
            signal.signal(signal.SIGALRM, handler)

            def learning_path(knowledge_path, lo_time_taken, max_time, time_per_session, kn_covered_by_lo, consolidated_score, time_limit=None):
                # Define a dictionary to map the values in kn_covered_by_lo to the indices in lo_time_taken
                value_to_index = {v: i for i, v in enumerate(range(len(lo_time_taken)))}
                
                # Start the clock
                start_time = time.time()

                # Define a function to check if a partial assignment is valid
                def is_valid(assignment):
                    # Check if the total time is within the max_time limit
                    total_time = 0
                    for index in assignment:
                        total_time += lo_time_taken[index]
                    if total_time > max_time:
                        return False
                    # Check if each session time is within the time_per_session limit
                    session_time = 0
                    for i in range(len(assignment)):
                        session_time += lo_time_taken[assignment[i]]
                        if i < len(assignment) - 1 and assignment[i] // 3 != assignment[i + 1] // 3:
                            # A new session starts
                            if session_time > time_per_session:
                                return False
                            session_time = 0
                    # Check the last session time
                    if session_time > time_per_session:
                        return False
                    # All checks passed
                    return True

                # Define a variable to store the best solution and its score
                best_solution = None
                best_score = 0


                # Define a function to find a solution using DFS
                def dfs(knowledge_path):
                    # Initialize the stack with an empty assignment and the variables in the knowledge path
                    stack = [([], [var for var in knowledge_path if var in range(len(kn_covered_by_lo))])]

                    # Loop until the stack is empty or a solution is found
                    nonlocal best_solution, best_score
                    while stack:
                        # Pop the top element from the stack
                        assignment, variables = stack.pop()

                        # Check if the assignment is complete
                        if not variables:
                            # Calculate the score of the solution
                            score = sum(consolidated_score[i] for i in assignment)

                            # Compare the score with the best score
                            if score > best_score:
                                # Update the best solution and its score
                                best_solution = assignment
                                best_score = score

                            # Continue the search
                            continue

                        # Get the next variable to be assigned
                        var = variables[0]

                        # Loop through the possible values for the variable
                        for val in kn_covered_by_lo[var]:
                            # Get the index of the value in lo_time_taken using the dictionary
                            index = value_to_index[val]

                            # Append the index to the assignment
                            assignment.append(index)

                            # Check if the assignment is valid
                            if is_valid(assignment):
                                # Push the new assignment and the remaining variables to the stack
                                stack.append((assignment.copy(), variables[1:]))

                            # Remove the index from the assignment
                            assignment.pop()

                    # Return the best solution
                    return best_solution

                # Check if the time_limit is not None
                if tl is not None:
                    # Set the alarm for the time limit
                    signal.alarm(tl)

                # Try to find a solution using dfs
                try:
                    solution = dfs(knowledge_path)
                except TimeoutException:
                    # The time limit was exceeded, return the best solution so far
                    solution = best_solution

                # Check if the time_limit is not None
                if tl is not None:
                    # Cancel the alarm
                    signal.alarm(0)

                # Calculate the total time of the solution
                total_time = sum(lo_time_taken[i] for i in solution) if solution else 0

                # Return the best solution, the total time, and the highest score
                return solution, total_time, best_score, time.time()-start_time

            
            solution, total_time, total_score, running_time = learning_path(kp, lo_time_taken, max_time, time_per_session, kn_covered_by_lo, consolidated_score, tl)

            # print("Student Profile id: ", student_profile_id)
            # print("KP Length:", len(kp))
            # # print("Algorithm Time Limit: ", tl)
            # print("LO Length", len(solution))
            # print("LP Total Time:", total_time)
            # print("LP Score:", total_score)
            # print("Algorithm Run Time:", running_time)
            # print("Max Time: ", max_time)
            # print("Max time per session: ", time_per_session)

            # Create the row of data as a dictionary
            # Create the row of data as a dictionary
            data = {
                "Student Profile id": int(student_profile_id),
                "Algorithm Time Limit": tl,
                "KP Length": len(kp),
                # Remove the try-except block from the dictionary
                "kp": str(kp),
                "solution": str(solution),
                "LP Total Time": total_time,
                "LP Score": total_score,
                "Algorithm Run Time": running_time,
                "Max Time": max_time,
                "Max time per session": time_per_session
            }

            # Use the try-except block to handle the TypeError
            try:
                # Assign the value of the LO Length key using len(solution)
                data["LO Length"] = len(solution)
            except TypeError:
                # Set the LO Length key to zero if solution is None
                data["LO Length"] = 0
            # Convert the dictionary to a dataframe
            data = pd.DataFrame(data, index=[0])
            experiment_df = pd.concat([experiment_df, data], ignore_index=True)

Experiment = "/remote_home/PhD_Project/Experiment/dfs_experiment.csv"
experiment_df.to_csv(Experiment)