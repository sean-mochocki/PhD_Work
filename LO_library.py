# Import pandas module
import pandas as pd
import os

# Define a function that loads an xlsx file from a specified folder into a pandas dataframe
def load_xlsx(filename, folder):
    # Use the join method of os.path to construct the full path of the file
    full_path = os.path.join(folder, filename)
    # Use the read_excel function of pandas to load the xlsx file as a dataframe
    df = pd.read_excel(full_path)
    # Return the dataframe
    return df

# Define a function that loads a text file from a folder and assigns numbers to the concepts
def load_text_and_assign_numbers(filename, folder, df):
    # Use the join method of os.path to construct the full path of the file
    full_path = os.path.join(folder, filename)
    # Use the open function to read the text file
    with open(full_path, "r") as f:
        # Read the text file as a single string
        text = f.read()
        # Split the text by commas and strip any whitespace
        concepts = [c.strip() for c in text.split(",")]
        # Convert the concepts to lowercase
        concepts = [c.lower() for c in concepts]
        # Create a dictionary that maps each concept to its index
        concept_dict = {c: i for i, c in enumerate(concepts)}
        # Create a new column for the dataframe that assigns numbers to the concepts
        # Convert the concept column to lowercase before mapping
        df["knowledge_node_covered"] = df["Concept"].str.lower().map(concept_dict)
        # Print the concept column where the knowledge_node_covered column is nan
        print(df[df["knowledge_node_covered"].isna()]["Concept"])
        # Convert the 'knowledge_node_covered' column to integer
        df["knowledge_node_covered"] = df["knowledge_node_covered"].astype(int)
    # Return the dataframe
    return df

#Load the learning objects file into a pandas dataframe
filename = "Learning_Objects_Specialty.xlsx"
folder_name = "/remote_home/PhD_Project/support_files"
df = load_xlsx(filename, folder_name)

knowledge_node_filename = "knowledge_nodes.txt"
knowledge_nodes_file_location = "/remote_home/PhD_Project/support_files"

df = load_text_and_assign_numbers(knowledge_node_filename, knowledge_nodes_file_location, df)
#print(df[['knowledge_node_covered']].to_string(index=False))

# Define a function that creates a new column for difficulty level
def create_difficulty_level(df):
    # Create a dictionary that maps each category to a number
    difficulty_dict = {"Easy": 1, "Medium": 2, "Hard": 3}
    # Create a new column for the dataframe that assigns numbers to the categories
    df["difficulty_level"] = df["Knowledge Density (Subjective)"].map(difficulty_dict)
    # Return the dataframe
    return df

# Define a function that counts the number of occurrences for each topic
def count_topics(df):
    # Use the value_counts method of the dataframe to count the frequency of each topic in the knowledge_node_covered column
    topic_counts = df["knowledge_node_covered"].value_counts()
    topic_counts = topic_counts.sort_index()
    # Print the topic counts
    print(topic_counts)

df = create_difficulty_level (df)
#print(df["difficulty_level"])

# Next, convert the time column to minutes and decimals of minutes
df['Time to Complete'] = df['Time to Complete'].str.split(':').apply(lambda x: int(x[0]) + round(int(x[1])/60,2))
# df['Time to Complete'] = pd.to_timedelta(df['Time to Complete'], unit='m')
# df['time_in_minutes'] = df['Time to Complete'].dt.components['hours']*60 +df['Time to Complete'].dt.components['minutes']+df['Time to Complete'].dt.components['seconds']/60

# Define a function that saves a pandas dataframe to a specific location
def save_dataframe(df, filename, location):
    # Use the join method of os.path to construct the full path of the file
    full_path = os.path.join(location, filename)
    # Use the to_csv method of the dataframe to save it to a csv file
    df.to_csv(full_path, index=False)

save_dataframe(df, "learning_objects.csv", "/remote_home/PhD_Project/data_structures")