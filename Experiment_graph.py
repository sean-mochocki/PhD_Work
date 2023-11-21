import pandas as pd
import numpy as np
import os
# Import matplotlib.pyplot library
import matplotlib.pyplot as plt

experiment = "/remote_home/PhD_Project/Experiment/"
data_folder = "/remote_home/PhD_Project/Data/"

experiment_df = pd.read_csv(os.path.join(experiment, "dfs_experiment_100.csv"))
experiment_0_df = pd.read_excel(os.path.join(experiment, "Student_0_ACE_Data_100.xlsx"))
experiment_1_df = pd.read_excel(os.path.join(experiment, "Student_1_ACE_Data_100.xlsx"))
experiment_2_df = pd.read_excel(os.path.join(experiment, "Student_2_ACE_Data_100.xlsx"))
experiment_3_df = pd.read_excel(os.path.join(experiment, "Student_3_ACE_Data_100.xlsx"))
experiment_4_df = pd.read_excel(os.path.join(experiment, "Student_4_ACE_Data_100.xlsx"))

experiment_0 = experiment_0_df.to_numpy()
experiment_1 = experiment_1_df.to_numpy()
experiment_2 = experiment_2_df.to_numpy()
experiment_3 = experiment_3_df.to_numpy()
experiment_4 = experiment_4_df.to_numpy()

# Filter the dataframe by Student Profile
experiment_df_0 = experiment_df[experiment_df["Student Profile id"] == 0]
experiment_df_1 = experiment_df[experiment_df["Student Profile id"] == 1]
experiment_df_2 = experiment_df[experiment_df["Student Profile id"] == 2]
experiment_df_3 = experiment_df[experiment_df["Student Profile id"] == 3]
experiment_df_4 = experiment_df[experiment_df["Student Profile id"] == 4]

fig, axs = plt.subplots(2,3)

# Student 0 id
# Group by Algorithm Time Limit and get the maximum LP Score
max_lp_score = experiment_df_0.groupby("Algorithm Time Limit")["LP Score"].max()

# Convert the result to a numpy array
max_lp_score_array = max_lp_score.to_numpy()

# Group by Algorithm Time Limit and get the sum of Algorithm Run Time
sum_run_time = experiment_df.groupby("Algorithm Time Limit")["Algorithm Run Time"].sum()

# Convert the result to a numpy array
sum_run_time_array = sum_run_time.to_numpy()

dfs_array = np.column_stack((sum_run_time_array, max_lp_score_array))

# Plot the sum_run_time_array with red dots and label "Depth First Search"
axs[0,0].scatter(dfs_array[:, 0], dfs_array[:, 1], color="red", marker="o", label="Depth First Search")

# Plot the experiment_0 array with blue dots and label "ACE"
axs[0,0].scatter(experiment_0[:, 0], experiment_0[:, 1], color="blue", marker="o", label="ACE")

# Set the labels of the x and y axes
axs[0,0].set_xlabel("Time in Seconds")
axs[0,0].set_ylabel("Adaptivity Score")

# Show the legend
#axs[0,0].legend()
axs[0,0].set_title("Student 0")
#axs[0,0].savefig(data_folder + "experiment_0.png")



# Student ID 1
# Group by Algorithm Time Limit and get the maximum LP Score
max_lp_score = experiment_df_1.groupby("Algorithm Time Limit")["LP Score"].max()

# Convert the result to a numpy array
max_lp_score_array = max_lp_score.to_numpy()

# Group by Algorithm Time Limit and get the sum of Algorithm Run Time
sum_run_time = experiment_df.groupby("Algorithm Time Limit")["Algorithm Run Time"].sum()

# Convert the result to a numpy array
sum_run_time_array = sum_run_time.to_numpy()

dfs_array = np.column_stack((sum_run_time_array, max_lp_score_array))

# Plot the sum_run_time_array with red dots and label "Depth First Search"
axs[0,1].scatter(dfs_array[:, 0], dfs_array[:, 1], color="red", marker="o", label="Depth First Search")

# Plot the experiment_0 array with blue dots and label "ACE"
axs[0,1].scatter(experiment_1[:, 0], experiment_1[:, 1], color="blue", marker="o", label="ACE")

# Set the labels of the x and y axes
axs[0,1].set_xlabel("Time in Seconds")
axs[0,1].set_ylabel("Adaptivity Score")

# Show the legend
#axs[0,1].legend()
axs[0,1].set_title("Student 1")
# plt.savefig(data_folder + "experiment_1.png")

# plt.close()



# Student ID 2
# Group by Algorithm Time Limit and get the maximum LP Score
max_lp_score = experiment_df_2.groupby("Algorithm Time Limit")["LP Score"].max()

# Convert the result to a numpy array
max_lp_score_array = max_lp_score.to_numpy()

# Group by Algorithm Time Limit and get the sum of Algorithm Run Time
sum_run_time = experiment_df.groupby("Algorithm Time Limit")["Algorithm Run Time"].sum()

# Convert the result to a numpy array
sum_run_time_array = sum_run_time.to_numpy()

dfs_array = np.column_stack((sum_run_time_array, max_lp_score_array))

# Plot the sum_run_time_array with red dots and label "Depth First Search"
axs[0,2].scatter(dfs_array[:, 0], dfs_array[:, 1], color="red", marker="o", label="Depth First Search")

# Plot the experiment_0 array with blue dots and label "ACE"
axs[0,2].scatter(experiment_2[:, 0], experiment_2[:, 1], color="blue", marker="o", label="ACE")

# Set the labels of the x and y axes
axs[0,2].set_xlabel("Time in Seconds")
axs[0,2].set_ylabel("Adaptivity Score")

# Show the legend
#axs[0,2].legend()
axs[0,2].set_title("Student 2")
# axs[0,2].savefig(data_folder + "experiment_2.png")

# plt.close()

# Student ID 3
# Group by Algorithm Time Limit and get the maximum LP Score
max_lp_score = experiment_df_3.groupby("Algorithm Time Limit")["LP Score"].max()

# Convert the result to a numpy array
max_lp_score_array = max_lp_score.to_numpy()

# Group by Algorithm Time Limit and get the sum of Algorithm Run Time
sum_run_time = experiment_df.groupby("Algorithm Time Limit")["Algorithm Run Time"].sum()

# Convert the result to a numpy array
sum_run_time_array = sum_run_time.to_numpy()

dfs_array = np.column_stack((sum_run_time_array, max_lp_score_array))

# Plot the sum_run_time_array with red dots and label "Depth First Search"
axs[1,0].scatter(dfs_array[:, 0], dfs_array[:, 1], color="red", marker="o", label="Depth First Search")

# Plot the experiment_0 array with blue dots and label "ACE"
axs[1,0].scatter(experiment_3[:, 0], experiment_3[:, 1], color="blue", marker="o", label="ACE")

# Set the labels of the x and y axes
axs[1,0].set_xlabel("Time in Seconds")
axs[1,0].set_ylabel("Adaptivity Score")

# Show the legend
#axs[1,0].legend()
axs[1,0].set_title("Student 3")
# plt.savefig(data_folder + "experiment_3.png")

# plt.close()

# Student ID 4
# Group by Algorithm Time Limit and get the maximum LP Score
max_lp_score = experiment_df_4.groupby("Algorithm Time Limit")["LP Score"].max()

# Convert the result to a numpy array
max_lp_score_array = max_lp_score.to_numpy()

# Group by Algorithm Time Limit and get the sum of Algorithm Run Time
sum_run_time = experiment_df.groupby("Algorithm Time Limit")["Algorithm Run Time"].sum()

# Convert the result to a numpy array
sum_run_time_array = sum_run_time.to_numpy()
print(sum_run_time_array)
dfs_array = np.column_stack((sum_run_time_array, max_lp_score_array))

# Plot the sum_run_time_array with red dots and label "Depth First Search"
axs[1,1].scatter(dfs_array[:, 0], dfs_array[:, 1], color="red", marker="o", label="Depth First Search")

# Plot the experiment_0 array with blue dots and label "ACE"
axs[1,1].scatter(experiment_4[:, 0], experiment_4[:, 1], color="blue", marker="o", label="ACE")

# Set the labels of the x and y axes
axs[1,1].set_xlabel("Time in Seconds")
axs[1,1].set_ylabel("Adaptivity Score")

# Show the legend
#axs[1,1].legend()
axs[1,1].set_title("Student 4")

fig.tight_layout()

# Create a list of labels for the legend
labels = ["Depth First Search", "ACE"]

# Create a legend for the figure
fig.legend(labels=labels, loc="lower right", bbox_to_anchor=(1, 0.1))
# Set the title of the figure
fig.suptitle("Adaptivity Score vs Time for 100 Knowledge Paths", fontsize=16)
fig.subplots_adjust(top=0.85)
# Delete the bottom right axis
fig.delaxes(axs[1, 2])

fig.savefig(data_folder + "combined_plot_100_paths.png")

# plt.savefig(data_folder + "experiment_4.png")

# plt.close()

