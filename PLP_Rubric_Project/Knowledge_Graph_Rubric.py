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
#LM_database['Avolve Keywords'] = LM_database['Avolve Keywords'].apply(lambda x: ast.literal_eval(x))
#print(LM_database.columns)

#Create the learner profile from the file
profile_database = pd.read_excel(learner_profile)

# Convert string representation of lists to actual lists using a lambda function
profile_database['cognitive_levels'] = profile_database['cognitive_levels'].apply(lambda x: ast.literal_eval(x))
profile_database['goals'] = profile_database['goals'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

# This is the point in the code where we start solving problems for individual learners.
goal_nodes = profile_database['goals'][0]

KS = []
for goals in goal_nodes:
    if goals != 1:
        paths = KG.get_shortest_paths(goals-1, 0)
        for path in paths:
            KS.extend(path)  # Append individual nodes from each path
    #if goals != 1: KS.extend(KG.get_shortest_paths(goals-1, 0))

# Delete duplicates from the knowledge set
KS = list(set(KS))



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








