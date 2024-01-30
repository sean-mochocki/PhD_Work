import igraph as ig

knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/support_files/knowledge_nodes_without_commas.txt"
knowledge_nodes_edges = "/home/sean/Desktop/PhD_Work/PhD_Work/support_files/knowledge_nodes_edges_igraph.txt"

kg = ig.Graph()

node_names = []
with open(knowledge_nodes) as f:
    for line in f:
        name = line.strip()
        if name:
            node_names.append(name)

kg = ig.Graph.Read_Ncol(knowledge_nodes_edges, names = True, directed=True)
print(kg.vs["name"])
# kg.vs["topic"] = node_names
# print(kg.vs["topic"])
# Print the vertex names and topics
# for v in kg.vs:
#     print(v["name"], v["topic"])

# Your code so far
print(kg)
paths = kg.get_all_simple_paths("0", "19")
named_paths = [kg.vs[path]["name"] for path in paths]
print(len(named_paths))
#print(named_paths)

path_sets = [frozenset(named_path) for named_path in named_paths]
unique_paths = [list(s) for s in set(path_sets)]

print(len(unique_paths))

# # New code to delete paths with identical members and different sequences
# path_dict = {} # dictionary to store paths as keys and sequences as values
# for path in paths:
#   path_set = frozenset(path) # convert path to a set
#   if path_set not in path_dict: # check if the set is already in the dictionary
#     path_dict[path_set] = path # if not, add the set as a key and the path as a value

# New code to delete paths with identical members and different sequences
path_dict = {} # dictionary to store paths as keys and sequences as values
for named_path in named_paths:
  path_list = sorted(named_path) # convert path to a sorted list
  if tuple(path_list) not in path_dict: # check if the list is already in the dictionary
    path_dict[tuple(path_list)] = named_path # if not, add the list as a key and the path as a value

final_paths = [] # list to store the final paths
for path_list, path in path_dict.items(): # iterate over the dictionary
  final_paths.append(path) # append the first path for each set of members

# final_paths = [] # list to store the final paths
# for path_set, path in path_dict.items(): # iterate over the dictionary
#   final_paths.append(path) # append the first path for each set of members
final_paths = [[int(x) for x in path] for path in final_paths]
print(len(final_paths)) # print the number of final paths
print(final_paths[0:5])

#print(kg)
#print(final_paths) # print the final paths

# def create_nodes(self, names_file):
#     with open(names_file) as f:  # Open the names file
#         number = 0  # Initialize the node number
#         for line in f:  # For each line in the file
#             name = line.strip()  # Remove any whitespace from the line
#             if name:  # If the line is not empty
#                 node = Node(number, name)  # Create a new node with the number and name
#                 self.nodes.append(node)  # Add the node to the list of nodes
#                 number += 1  # Increment the node number

# def create_edges(self, edges_file):
#     with open(edges_file) as f:  # Open the edges file
#         for line in f:  # For each line in the file
#             parts = line.split(":")  # Split the line by colon
#             if len(parts) == 2:  # If the line has two parts
#                 source = int(parts[0])  # The first part is the source node number
#                 targets = parts[1].split(",")  # The second part is a list of target node numbers separated by comma
#                 for target in targets:  # For each target node number
#                     target = int(target)  # Convert it to an integer
#                     if source < len(self.nodes) and target < len(
#                             self.nodes):  # If both source and target are valid node numbers
#                         self.nodes[source].add_edge(self.nodes[target])  # Add an edge from source to target