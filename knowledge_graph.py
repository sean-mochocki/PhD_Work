import pickle
import random

# A class to represent a node in the knowledge graph
class Node:
    # Initialize the node with a number and a name
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.edges = [] # A list of nodes that this node points to

    # Add an edge from this node to another node
    def add_edge(self, node):
        self.edges.append(node)

    # Return a string representation of the node
    def __str__(self):
        return f"{self.number}:{self.name}"

class KnowledgeGraph:
    # Initialize the graph with a file of names and a file of edges
    def __init__(self, names_file, edges_file):
        self.nodes = [] # A list of nodes in the graph
        self.create_nodes(names_file) # Create the nodes from the names file
        self.create_edges(edges_file) # Create the edges from the edges file

    # Create the nodes from the names file
    def create_nodes(self, names_file):
        with open(names_file) as f: # Open the names file
            number = 0 # Initialize the node number
            for line in f: # For each line in the file
                name = line.strip() # Remove any whitespace from the line
                if name: # If the line is not empty
                    node = Node(number, name) # Create a new node with the number and name
                    self.nodes.append(node) # Add the node to the list of nodes
                    number += 1 # Increment the node number
                #print(name)

    # Create the edges from the edges file
    def create_edges(self, edges_file):
        with open(edges_file) as f: # Open the edges file
            for line in f: # For each line in the file
                parts = line.split(":") # Split the line by colon
                if len(parts) == 2: # If the line has two parts
                    source = int(parts[0]) # The first part is the source node number
                    targets = parts[1].split(",") # The second part is a list of target node numbers separated by comma
                    for target in targets: # For each target node number
                        target = int(target) # Convert it to an integer
                        if source < len(self.nodes) and target < len(self.nodes): # If both source and target are valid node numbers
                            self.nodes[source].add_edge(self.nodes[target]) # Add an edge from source to target

    # Print the graph with the nodes and their edges
    def print_graph(self):
        for node in self.nodes: # For each node in the graph
            print(node) # Print the node
            print("->", end=" ") # Print an arrow to indicate its edges
            for edge in node.edges: # For each edge of the node
                print(edge, end=" ") # Print the edge node
            print() # Print a new line
    
    def print_topic_names(self):
        for node in self.nodes: print(node.name)

    # A function to find all possible paths between two nodes in the graph
    # with an optional parameter to limit the maximum path length
    def find_all_paths(self, start, end, max_length=None):
        # Initialize an empty list to store the paths
        paths = []
        # Initialize a queue to store the current path and node
        queue = []
        # Enqueue the first node and path
        queue.append((start, [start]))
        # Loop until the queue is empty
        while queue:
            # Dequeue the current node and path
            node, path = queue.pop(0)
            # If the current node is the end node, add the path to the list of paths
            if node == end:
                paths.append(path)
            # Otherwise, iterate over the adjacent nodes of the current node
            else:
                for adjacent in self.nodes[node].edges:
                    # If the adjacent node is not in the current path, and the extended path does not exceed the maximum length (if given), enqueue it with the extended path
                    if adjacent.number not in path and (max_length is None or len(path) < max_length):
                        queue.append((adjacent.number, path + [adjacent.number]))
        # Return the list of paths
        return paths

    # Define a function that only returns unique paths
    def find_unique_paths(self, start, end, max_length=None):
        # Call the find_all_paths function to get the list of all possible paths
        paths = self.find_all_paths(start, end, max_length)
        # Initialize an empty list to store the unique paths
        unique_paths = []
        # Loop through each path in the list of paths
        for path in paths:
            # Sort the path to make it easier to compare
            sorted_path = sorted(path)
            # Check if the sorted path is already in the unique paths list
            if sorted_path not in unique_paths:
                # If not, add it to the unique paths list
                unique_paths.append(sorted_path)
        # Return the unique paths list
        return unique_paths

        # Define a function that finds the number of unique paths between all pairs of nodes

    # Define a function that finds the number of unique paths between all pairs of nodes
    def count_unique_paths(self, max_length=10):
        # Initialize an empty dictionary to store the number of unique paths
        count = {}
        # Loop through all pairs of nodes in the graph
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                # Get the start and end nodes
                start = i
                end = j
                # Call the find_unique_paths function for the pair
                print("starting: ", i, " ", j)
                unique_paths = self.find_unique_paths(start, end, max_length)
                # Get the number of unique paths
                num = len(unique_paths)
                print(i, " ", j, " = ", num)
                # Store the number in the dictionary, using the pair as the key
                count[(start, end)] = num
        # Return the dictionary
        return count

    # A function to check if a list of integers is a valid knowledge path
    def check_if_real_path(self, path):
        # If the path is empty or has only one node, return 1
        if len(path) <= 1:
            return 0
        # Loop through the path from the second node
        for i in range(1, len(path)):
            # Get the current and previous node numbers
            current = path[i]
            previous = path[i-1]
            # If the current or previous node number is invalid, return 0
            if current >= len(self.nodes) or previous >= len(self.nodes):
                return 0
            # Get the list of edges from the previous node
            edges = self.nodes[previous].edges
            # If the current node is not in the list of edges, return 0
            if self.nodes[current] not in edges:
                return 0
        # If the loop finishes without returning 0, return 1
        return 1


    # # Define a sub function that finds one random path from start to end
    # def find_one_random_path(self, start, end, visited=None):
    #     # Initialize the visited set to keep track of the visited nodes
    #     if visited is None:
    #         visited = set()
    #     # Add the start node to the visited set
    #     visited.add(self.nodes[start]) # Add the node object to the visited set
    #     # Initialize an empty list to store the path
    #     path = []
    #     # Append the start node to the path
    #     path.append(start)
    #     # Initialize a variable to store the current node
    #     current = self.nodes[start] # Access the node object from the self.nodes list
    #     # Loop until the current node is the end node or there are no more edges to follow
    #     while current.number != end and self.nodes[current.number].edges: # Use the node number as an index for the self.nodes list
    #         # Choose a random edge from the current node
    #         edge = random.choice(self.nodes[current.number].edges) # Use the node number as an index for the self.nodes list
    #         # Check if the edge node is not visited
    #         if self.nodes[edge.number] not in visited: # Check the node object in the visited set
    #             # Add the edge node to the visited set
    #             visited.add(self.nodes[edge.number]) # Add the node object to the visited set
    #             # Append the edge node to the path
    #             path.append(edge)
    #             # Update the current node to the edge node
    #             current = self.nodes[edge.number] # Use the node number as an index for the self.nodes list
    #         else:
    #             # Remove the edge from the current node's edges to avoid looping
    #             self.nodes[current.number].edges.remove(edge) # Use the node number as an index for the self.nodes list
    #     # Return the path
    #     return path

    import random

    import random

    def find_one_random_path(self, start, end, stack=None, visited=None):
        # Initialize the stack and the visited set if they are None
        if stack is None:
            stack = []
        if visited is None:
            visited = set()
        # Push the start node to the stack and add it to the visited set
        stack.append(start)
        visited.add(self.nodes[start])
        # While the stack is not empty
        while stack:
            # Pop the top node from the stack and assign it to current
            current = stack.pop()
            # If the current node is the end node, return the stack as the path
            if current == end:
                return stack
            # Otherwise, get the list of adjacent nodes of the current node and shuffle it randomly
            adj_nodes = self.nodes[current].edges.copy()
            random.shuffle(adj_nodes)
            # For each adjacent node
            for adj_node in adj_nodes:
                # If the adjacent node is not in the visited set
                if self.nodes[adj_node.number] not in visited:
                    # Make a copy of the stack and the visited set
                    stack_copy = stack.copy()
                    visited_copy = visited.copy()
                    # Push the adjacent node to the stack copy and add it to the visited set copy
                    stack_copy.append(adj_node.number)
                    visited_copy.add(self.nodes[adj_node.number])
                    # Recursively call the function with the adjacent node, the end node, the stack copy and the visited set copy as parameters and assign the result to path
                    path = self.find_one_random_path(adj_node.number, end, stack_copy, visited_copy)
                    # If the path is not None, return the path
                    if path is not None:
                        return path
        # If the stack is empty, return None
        return None



    # Define a function that returns unique random learning paths
    def find_random_paths(self, start, end, num_paths, seed=None):
        # Check if the seed is None
        if seed is not None:
            # Set the random seed for reproducibility
            random.seed(seed)
        # Initialize an empty set to store the random paths
        random_paths = set()
        # Loop until the desired number of paths is reached
        while len(random_paths) < num_paths:
            # Call the find_one_random_path function to get one random path
            path = self.find_one_random_path(start, end)
            # Convert the path to a tuple of node numbers
            path_tuple = tuple([node.number if isinstance(node, Node) else node for node in path])
            # Check if the path tuple is already in the random paths set
            if path_tuple not in random_paths:
                # If not, add it to the random paths set
                random_paths.add(path_tuple)
        # Return the list of random paths
        random_paths = list(random_paths)
        random_paths = [list(t) for t in random_paths]
        return random_paths


    

# Save the Knowledge graph
knowledge_nodes = "/home/sean/Desktop/PhD_Work/PhD_Work/support_files/knowledge_nodes.txt"
knowledge_nodes_edges = "/home/sean/Desktop/PhD_Work/PhD_Work/support_files/knowledge_nodes_edges.txt"
knowledge_graph_file = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures/knowledge_graph.pkl"
knowledge_graph_folder = "/home/sean/Desktop/PhD_Work/PhD_Work/data_structures"
# Create a knowledge graph from names.txt and edges.txt files
kg = KnowledgeGraph(knowledge_nodes, knowledge_nodes_edges)

# random_paths = kg.find_random_paths(0,19,100, 42)
# print(random_paths)

# path_checker = [0, 6, 2, 7, 12, 8, 19, 17, 13, 6, 3]
# print(kg.check_if_real_path(path_checker))

with open(knowledge_graph_file, "wb") as f:
    pickle.dump(kg, f)
