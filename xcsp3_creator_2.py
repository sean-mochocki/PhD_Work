import pickle
import pandas as pd
from knowledge_graph import KnowledgeGraph
import os
import numpy as np
from pycsp3 import *
from pycsp3 import ACE
import ast
from operator import itemgetter

# First load the saved knowledge graph
knowledge_graph_file = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures/knowledge_graph.pkl"
kg = None
with open(knowledge_graph_file, "rb") as f:
    kg = pickle.load(f)

# Load the student profile and learning objects data structures into pandas dataframes
data_structures = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures/"
learning_objects_df = pd.read_csv(os.path.join(data_structures, "learning_objects.csv"))
profiles_df = pd.read_csv(os.path.join(data_structures, "consolidated_profiles.csv"))

#Define constants for the knowledge graph
num_kn = 20
num_kp = 1
student_profile_id = 8

print("student profile id: ", student_profile_id)

student_start_node = profiles_df["goal1"][student_profile_id]
student_end_node = profiles_df["goal2"][student_profile_id]

# Pull values off of the first user profile
max_time = int(profiles_df['max_time'][student_profile_id]) * 100
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
Alpha = 1
Beta = 1
Zeta = 1

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

# print(kn_covered_by_lo)

# for sublist in kn_covered_by_lo:
#     # Set min_time_so_far equal to the first value in the list
#     min_time_so_far = lo_time_taken[sublist[0]]
#     for lo_index in sublist:
#         lo_time = lo_time_taken[lo_index]
#         if lo_time > min_time_so_far:
#             redundant_lo.append(lo_index)
#             min_time_so_far = lo_time

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
# Recalculate the knowledge nodes covered by LOs
kn_covered_by_lo = return_kn_covered_by_lo(num_kn, knowledge_nodes_covered)

print("Total number of LOs under consideration: ", len(lo_time_taken))
print("Total number of LOs deleted due to time violations: ", + len(time_violation))
print("Total number of LOs deleted due to knowledge node coverage redundancy", + len(redundant_lo))

#print("number of knowledge paths is: ", num_kp)
#knowledge_path = kg.find_random_paths(student_start_node, student_end_node, 20, 42)
#knowledge_path = kg.find_unique_paths(student_start_node, student_end_node, 20)
# # Assign minimum times to knowledge paths to make sure that all have valid solutions
#
# # Define a custom key function that returns the score from consolidated_score
# def get_time(x):
#     return lo_time_taken[x]
#
# # Sort each sublist of kn_covered_by_lo by the time_taken in ascending order (meaning that fast LOs are first)
# for sublist in kn_covered_by_lo:
#     sublist.sort(key=get_time, reverse=False)
#
# # Record the minimum possible time for each knowledge path
# path_times = []
# for path in knowledge_path:
#     path_time = 0
#     for kn in path:
#         path_time += lo_time_taken[kn_covered_by_lo[kn][0]]
#     path_times.append(path_time)
#
# #print("Path times are: ", path_times)
#
# kp_time_violation = []
# for index, path_time in enumerate(path_times):
#     if path_time > max_time:
#         kp_time_violation.append(index)
#
# print("Number of knowledge paths with no valid solution: ", len(kp_time_violation))
#
# # Create a new list with only the elements that are not in kp_time_violation
# knowledge_path = [sublist for i, sublist in enumerate(knowledge_path) if i not in kp_time_violation]
# print("Number of knowledge paths of interest: ", len(knowledge_path))
#
# #Sort the Knowledge Paths according to the highest possible score of their individual elements
# # Sort each sublist of kn_covered_by_lo by the score in descending order
# for sublist in kn_covered_by_lo:
#     sublist.sort(key=get_score, reverse=True)
#
# # Record the maximum possible score for each knowledge path
# path_max_score = []
# for path in knowledge_path:
#     score = 0
#     for kn in path:
#         score += consolidated_score[kn_covered_by_lo[kn][0]]
#     path_max_score.append(score)
#
# zipped = list(zip(path_max_score, knowledge_path))
#
# sorted_zipped = sorted(zipped, key=itemgetter(0), reverse = True)
# sorted_path_max_score, sorted_knowledge_path = zip(*sorted_zipped)
#
# sorted_path_max_score = list(sorted_path_max_score)
# sorted_knowledge_path = list(sorted_knowledge_path)
#
# print("Max score of kps are: ", sorted_path_max_score)
# #Create function that performs back-tracking and forward checking COP
#
# # Call a search algorithm with the best knowledge path found so far
#
# def check_Max_score(kp, kn_coverage, lo_scores, lo_times, max_time):
#     Best_LP = []
#     time_taken = 0
#     best_score = 0
#     #First check if the maximum score is a valid LP. If so, return that LO
#     for kn in kp:
#         Best_LP.append(kn_coverage[kn][0])
#         time_taken += lo_times[kn_coverage[kn][0]]
#         best_score += lo_scores[kn_coverage[kn][0]]
#     if time_taken <= max_time:
#         print("Best LP is equivalent to maximal score")
#         return Best_LP, time_taken, best_score
#     else:
#         print("Best LP is not equivalent to maximal score")
#         return [], 0, 0
sorted_knowledge_path_file = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures/student_profile_" + str(student_profile_id) + "_top_200_KPs.pkl"
sorted_knowledge_path = None
with open(sorted_knowledge_path_file, "rb") as f:
    sorted_knowledge_path = pickle.load(f)

# Delete all elements of sorted knowledge path except for first. This is the one that will be encoded
sorted_knowledge_path = sorted_knowledge_path[0:10]

#All necessary variables for XCSP3 are defined, begin creating XCSP3 file
# Create a variable that defines the knowledge nodes to be covered
kn = VarArray(size=num_kn, dom={0,1})
# Create a variable that defines the available Learning Objects
lo = VarArray(size = len(lo_time_taken), dom={0,1})
# Create a variable that defines the knowledge paths available
kp = VarArray(size=[len(sorted_knowledge_path)], dom = {0,1})
satisfy (sum(kp) == 1)

# # If a given knowledge path is chosen, this implies that the associated knowledge nodes are chosen
satisfy(*(imply(kp[i] == 1, kn[sorted_knowledge_path[i][j]] == 1) for i in range(len(kp)) for j in range(len(sorted_knowledge_path[i]))))

# If a given knowledge path is chosen, this implies that the knowledge nodes not associated with the knowledge path are not chosen
if len(sorted_knowledge_path[0]) != num_kn:
   satisfy(*(imply(kp[i] == 1, conjunction(kn[j] == 0 for j in range(num_kn) if j not in sorted_knowledge_path[i])) for i in range(len(kp))))
else:
    satisfy(*(imply(kp[i] == 1, conjunction(kn[j] == 0 for j in range(num_kn) if j not in sorted_knowledge_path[i])) for i in range(1, len(kp))))

# This line creates a disjunction for every knowledge node, if it is 1, then one of the LOs that cover it must be 1
for i in range(len(kn)): satisfy(If(kn[i]==1, Then=disjunction([lo[kn_covered_by_lo[i][j]] for j in range(len(kn_covered_by_lo[i]))]), Else=kn[i]==0))

# Make a requirement that the total number of learning objects equals the total number of knowledge nodes
satisfy(Sum(lo) == Sum(kn))

lo_time_taken = [int(x*100) for x in lo_time_taken]
# Make the requirement that the sum of all learning objects cannot exceed max time
satisfy(
    # The sum of the learning objects cannot exceed the max time
    Sum(lo * lo_time_taken) <= max_time*100
)

# # The goal of this next section is to define an adaptivity score that we can maximize.
adaptivity_score = Var(dom=range(5000))
consolidated_score = [int(x*10) for x in consolidated_score]
satisfy(adaptivity_score == Sum(lo*consolidated_score))

maximize(adaptivity_score)