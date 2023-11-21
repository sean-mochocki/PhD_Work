import pandas as pd
import random

# Define a class for student profile
class StudentProfile:
    # Initialize the object with the given parameters
    def __init__(self, cognitive_levels, goal1, goal2, preferred_media, research, website, discussion, educational, news_article, diy, lecture, powerpoint, textbook_excerpt, max_time, time_per_session):
        self.cognitive_levels = cognitive_levels # An array of 20 values between 1 and 3
        self.goal1 = goal1 # An integer between 0 and 19
        self.goal2 = goal2 # An integer between 0 and 19, not equal to goal1
        self.preferred_media = preferred_media # A string of Video, Written, or No_Preference
        self.research = research # An integer between 1 and 9
        self.website = website # An integer between 1 and 9, not equal to research
        self.discussion = discussion # An integer between 1 and 9, not equal to research and website
        self.educational = educational # An integer between 1 and 9, not equal to research, website, and discussion
        self.news_article = news_article # An integer between 1 and 9, not equal to research, website, discussion, and educational
        self.diy = diy # An integer between 1 and 9, not equal to research, website, discussion, educational, and news_article
        self.lecture = lecture # An integer between 1 and 9, not equal to research, website, discussion, educational, news_article, and diy
        self.powerpoint = powerpoint # An integer between 1 and 9, not equal to research, website, discussion, educational, news_article, diy, and lecture
        self.textbook_excerpt = textbook_excerpt # An integer between 1 and 9, not equal to research, website, discussion, educational, news_article, diy, lecture, and powerpoint
        self.max_time = max_time # An integer between 2 and 8
        self.time_per_session = time_per_session # An integer of 30, 60, 90, 120, 150, or 180

    # Define a method to print the object attributes
    def print(self):
        print("Cognitive levels:", self.cognitive_levels)
        print("Goal 1:", self.goal1)
        print("Goal 2:", self.goal2)
        print("Preferred media:", self.preferred_media)
        print("Research:", self.research)
        print("Website:", self.website)
        print("Discussion:", self.discussion)
        print("Educational:", self.educational)
        print("News article:", self.news_article)
        print("DIY:", self.diy)
        print("Lecture:", self.lecture)
        print("Powerpoint:", self.powerpoint)
        print("Textbook excerpt:", self.textbook_excerpt)
        print("Max time:", self.max_time)
        print("Time per session:", self.time_per_session)

# Define a function to randomly initialize a student profile object
def random_student_profile():
    # Generate a random array of 20 values between 1 and 3 for cognitive levels
    cognitive_levels = [random.randint(1, 3) for i in range(20)]
    # Generate two random integers between 0 and 19 for goals, ensuring they are not equal
    goal1 = random.randint(0, 19)
    goal2 = random.randint(0, 19)
    while goal2 == goal1:
        goal2 = random.randint(0, 19)
    # Generate a random string of Video, Written, or No_Preference for preferred media
    preferred_media = random.choice(["Video", "Written", "No_Preference"])
    # Generate a random permutation of integers from 1 to 9 for the nine categories
    categories = random.sample(range(1, 10), 9)
    # Assign the categories to the corresponding attributes
    research = categories[0]
    website = categories[1]
    discussion = categories[2]
    educational = categories[3]
    news_article = categories[4]
    diy = categories[5]
    lecture = categories[6]
    powerpoint = categories[7]
    textbook_excerpt = categories[8]
    # Generate a random integer between 2 and 8 for max time
    max_time = random.randint(40, 480)
    # Generate a random integer of 30, 60, 90, 120, 150, or 180 for time per session
    time_per_session = random.choice([30, 60, 90, 120, 150, 180])
    # Create a student profile object with the generated parameters
    student_profile = StudentProfile(cognitive_levels, goal1, goal2, preferred_media, lecture, website, discussion, educational, news_article, diy, lecture, powerpoint, textbook_excerpt, max_time, time_per_session)
    # Return the student profile object
    return student_profile

# Define a function that generates a number of student profiles and returns a pandas dataframe
def generate_student_profiles(n):
    # Initialize an empty list to store the profiles
    profiles = []
    # Loop n times
    for i in range(n):
        # Generate a random student profile using the previous function
        profile = random_student_profile()
        # Append the profile to the list
        profiles.append(profile)
    # Convert the list of profiles to a pandas dataframe
    df = pd.DataFrame([vars(p) for p in profiles])
    # Return the dataframe
    return df

# Define a function that saves a pandas dataframe to a csv file
def save_dataframe(df, filename):
    # Use the to_csv method of the dataframe to save it to a csv file
    df.to_csv(filename, index=False)

# Generate 10 student profiles and store them in a dataframe
df = generate_student_profiles(100)
# Print the dataframe
#print(df)

profiles = "/remote_home/PhD_Project/data_structures/profiles.csv"
save_dataframe(df, "profiles.csv")