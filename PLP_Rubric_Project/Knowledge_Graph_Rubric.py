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



#Create the learner profile from the file
profile_database = pd.read_excel(learner_profile)

# Convert string representation of lists to actual lists using a lambda function
#profile_database['cognitive_levels'] = profile_database['cognitive_levels'].apply(lambda x: ast.literal_eval(x))
profile_database['goals'] = profile_database['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

# This is the point in the code where we start solving problems for individual learners.
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
# These are ranked from CL 1 to 3, where 3 is most advanced
cog_levels_list = list(ast.literal_eval(cog_levels_str))
print(cog_levels_list)

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

# Capture the users preferred media type
preferred_media = profile_database['preferred_media'][student_profile_id]

# Capture the learning object variables of interest
LM_difficulty = LM_database['Knowledge Density (Subjective)']
LM_titles = LM_database['Title']
LM_KNs_Covered = LM_database['KNs Covered']
# Next define parameters for set covering problem
lm_time_taken = LM_database['Time to Complete'].to_numpy()
lm_time_taken = [int(x * 100) for x in lm_time_taken]
print(lm_time_taken)

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








