import pickle
import pandas as pd
from knowledge_graph import KnowledgeGraph
import os
import numpy as np
from pycsp3 import *
from pycsp3 import ACE
import ast

# First load the saved knowledge graph
knowledge_graph_file = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures/knowledge_graph.pkl"
kg = None
with open(knowledge_graph_file, "rb") as f:
    kg = pickle.load(f)

# Load the student profile and learning objects data structures into pandas dataframes
data_structures = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures"
learning_objects_df = pd.read_csv(os.path.join(data_structures, "learning_objects.csv"))
profiles_df = pd.read_csv(os.path.join(data_structures, "profiles.csv"))

#Define constants for the knowledge graph
num_kn = 20
num_kp = 25
student_profile_id = 4

#Pull values off of the first user profile
max_time = int(profiles_df['max_time'][student_profile_id])
time_per_session = int(profiles_df['time_per_session'][student_profile_id])

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

# Create a variable that defines the knowledge nodes to be covered
kn = VarArray(size=num_kn, dom={0,1})

# Create a variable that defines the available Learning Objects
lo = VarArray(size = learning_objects_df.shape[0], dom={0,1})

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

# Create the unique knowledge paths that can possibly be covered
paths = kg.find_random_paths(0,19,num_kp, 42)

# Create the variable that holds the various knowledge paths
# This variable will be composed of 0s and 1s, where a 1 indicates a chosen learning path
kp = VarArray(size=[len(paths)], dom = {0,1})

# First require that at least 1 knowledge path is chosen
# satisfy (disjunction(*(kp[i] == 1 for i in range(len(kp)))))

# Next require that only one knowledge path is chosen
satisfy (sum(kp) == 1)

# # If a given knowledge path is chosen, this implies that the associated knowledge nodes are chosen
satisfy(*(imply(kp[i] == 1, kn[paths[i][j]] == 1) for i in range(len(kp)) for j in range(len(paths[i]))))

# If a given knowledge path is chosen, this implies that the knowledge nodes not associated with the knowledge path are not chosen
satisfy(*(imply(kp[i] == 1, conjunction(kn[j] == 0 for j in range(num_kn) if j not in paths[i])) for i in range(len(kp))))

# satisfy(*(kp[i] == 1 == kn[paths[i][j]] == 1 for i in range(len(kp)) for j in range(len(paths[i]))))

# satisfy(*((kp[i] == 1) & (kn[paths[i][j]] == 1) for i in range(len(kp)) for j in range(len(paths[i]))))

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


# This line creates a disjunction for every knowledge node, if it is 1, then one of the LOs that cover it must be 1
for i in range(len(kn)): satisfy(If(kn[i]==1, Then=disjunction([lo[kn_covered_by_lo[i][j]] for j in range(len(kn_covered_by_lo[i]))]), Else=kn[i]==0))

# Make a requirement that the total number of learning objects equals the total number of knowledge nodes
satisfy(Sum(lo) == Sum(kn))

# Next define parameters for set covering problem
lo_time_taken = learning_objects_df['Time to Complete'].to_numpy()
lo_time_taken = [int(x*100) for x in lo_time_taken]

# Make the requirement that the sum of all learning objects cannot exceed max time
satisfy(
    # The sum of the learning objects cannot exceed the max time
    Sum(lo * lo_time_taken) <= max_time*100
)

# Make the requirement that individual learning objects to not exceed time-per-session
satisfy(    
    # Each individual learning object cannot exceed max time
    lo[i] * lo_time_taken[i] <= time_per_session*100 for i in range(len(lo))
)

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

# Post a constraint that adds 20 points to the adaptivity score if lo == 1 and LO_cognitive_matches == 1
# Add 5 point for each LO_media_match. This goes to 0 if the student has no preference
# Add points for the students preferred content type
# This adaptivity score will need to be optimized over time.

# # The goal of this next section is to define an adaptivity score that we can maximize.
adaptivity_score = Var(dom=range(100000))

#Combine these three scores so that they can be combined into the adaptivity_score
Alpha = 20
Beta = 5
Zeta = 1

consolidated_score = [Alpha*c + Beta*m + Zeta*f for c, m, f in zip(LO_cognitive_matches, LO_media_match, flipped_score)]

satisfy(adaptivity_score == Sum(lo*consolidated_score))

# satisfy(
#     adaptivity_score == 20*Sum((lo * LO_cognitive_matches)) + 5*Sum(lo*LO_media_match)+ Sum(lo*flipped_score)
# )

# satisfy(
#     adaptivity_score == (20*Sum((lo * LO_cognitive_matches) + 5*Sum([lo[i] * LO_media_match[i] for i in range(len(lo))])))
# )

# satisfy(
#     adaptivity_score == Sum(20*(lo * LO_cognitive_matches) + 5*(lo * LO_media_match))
# )

# satisfy(
#     adaptivity_score == Sum(20*Sum(lo[i] * LO_cognitive_matches[i] for i in range(len(lo))) + 5*(lo[i] * LO_media_match[i]) for i in range(len(lo)))
# )

# satisfy(
#     adaptivity_score == 20*Sum([lo[i] * LO_cognitive_matches[i] for i in range(len(lo))]) + 
#     5*Sum([lo[i] * LO_media_match[i] for i in range(len(lo))]) + 
#     Sum([lo[i] * flipped_score[i] for i in range(len(lo))])
# )


maximize(adaptivity_score)








