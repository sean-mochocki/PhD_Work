import time

# Define the data structures
kn_covered_by_lo = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

lo_time_taken = [80, 10, 30, 50, 90, 40, 60, 20, 70]
consolidated_score = [40, 20, 100, 50, 30, 90, 80, 60, 70]

max_time = 120
time_per_session = 200

# Define the knowledge path
knowledge_path = [1, 0, 2]

def learning_path(knowledge_path, lo_time_taken, max_time, time_per_session, kn_covered_by_lo, consolidated_score):
    # Define a dictionary to map the values in kn_covered_by_lo to the indices in lo_time_taken
    value_to_index = {v: i for i, v in enumerate(range(1, len(lo_time_taken) + 1))}
    
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

    # Call the dfs function with the knowledge path
    solution = dfs(knowledge_path)

    # Calculate the total time of the solution
    total_time = sum(lo_time_taken[i] for i in solution) if solution else 0

    # Return the best solution, the total time, and the highest score
    return solution, total_time, best_score, time.time()-start_time

solution, total_time, best_score, duration = learning_path(knowledge_path, lo_time_taken, max_time, time_per_session, kn_covered_by_lo, consolidated_score)

print("knowledge path is: ", knowledge_path)
print("Solution is: ", solution)
print("total time is: ", total_time)
print("best score is: ", best_score)
print("Algorithm took: ", duration, " seconds" )
