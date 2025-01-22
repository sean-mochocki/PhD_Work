import igraph as ig
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Data/Knowledge_Nodes.txt"
knowledge_graph_edges = "/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Data/Knowledge_Graph_Edges.txt"
learner_profile = "/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Data/Learner_Profile_8_Jan_2025.xlsx"
learning_materials = "/home/sean/Desktop/PhD_Work/PhD_Work/PLP_Rubric_Project/Data/Learning_Materials_Base_set.xlsx"

# This is the section where we create the Knowledge Graph

# Identify the names of the knowledge nodes
with open(knowledge_nodes, 'r') as file:
    # Read the lines into a list
    KNs = file.read().splitlines()

# Read the edges from the file
with open(knowledge_graph_edges, "r") as file:
    edges = [line.strip().split(" -> ") for line in file]

KG = ig.Graph(directed=True)
KG.add_vertices(KNs)

# Add Nodes from the edges (ensure all nodes are added)
#all_nodes = set([node for edge in edges for node in edge])
KG.add_edges(edges)

#path = KG.get_shortest_paths("Generation of natural language", "AI in General")[0]
#print(path)
#node_names = [KG.vs[node_index]['name'] for node_index in path]
#print(node_names)
#print(KG)
# Create the set of LMs from the file
LM_database = pd.read_excel(learning_materials)
# Convert the 'KNs Covered' column to a list of strings
LM_database['KNs Covered'] = LM_database['KNs Covered'].str.split(',')
# Convert 'Time to Complete' to decimal format (from MM:SS format)
LM_database['Time to Complete'] = LM_database['Time to Complete'].str.split(":").apply(lambda x: int(x[0]) + int(x[1]) / 60)

# In this section we determine the CTML score of each LM. This score does not depend on individual learners so we can accomplish this task before loading the learner profile

# Capture the LM parameters based on the Cognitive Theory of Multimedia Learning
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

# Determine the Difficulty of each LM. This variable is not dependent on the learner
LM_difficulty = LM_database['Knowledge Density (Subjective)']
# Define a function that maps strings to integers
def difficulty_to_int(difficulty):
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

# Define LM_difficulty_int as integers between 1 and 4 where 1 is low and 4 is very high
LM_difficulty_int = LM_difficulty.map(difficulty_to_int)

#Create the learner profile from the file
profile_database = pd.read_excel(learner_profile)

# Convert string representation of lists to actual lists using a lambda function
#profile_database['cognitive_levels'] = profile_database['cognitive_levels'].apply(lambda x: ast.literal_eval(x))
profile_database['goals'] = profile_database['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])


# Define the duration of the LMs - this does not depend on individual learners
lm_time_taken = LM_database['Time to Complete'].to_numpy()
lm_time_taken = [int(x * 100) for x in lm_time_taken]

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

# Multiply by 100 for convenience when comparing student learning goal times to LM times
max_time = int(profile_database['maximum_time'][student_profile_id]) * 100
min_time = int(profile_database['minimum_time'][student_profile_id]) * 100

# Get the first element of the column as a string
cog_levels_str = profile_database['cognitive_levels'][student_profile_id]

# Evaluate the string as a Python expression and convert it into a list of integers
# These are ranked from CL 1 to 4, where 4 is most advanced
cog_levels_list = list(ast.literal_eval(cog_levels_str))
#print(cog_levels_list)

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
print(flipped_preference_score)

# Capture the users preferred media type and score LMs accordingly. In the case that the learner's preference is No-preference we won't use preferred_media as a factor in the fitness function
preferred_media = profile_database['preferred_media'][student_profile_id]
print(preferred_media)
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
print(LM_overall_preference_score)

# Capture the learning object variables of interest

LM_titles = LM_database['Title']
LM_KNs_Covered = LM_database['KNs Covered']





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








