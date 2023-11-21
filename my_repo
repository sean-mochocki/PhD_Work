import pandas as pd
import numpy as np
import os
# Import matplotlib.pyplot library
import matplotlib.pyplot as plt

experiment = "/remote_home/PhD_Project/Experiment/"
data_folder = "/remote_home/PhD_Project/Data/"

experiment_0_100_df = pd.read_excel(os.path.join(experiment, "Student_0_ACE_Data_100.xlsx"))
experiment_1_100_df = pd.read_excel(os.path.join(experiment, "Student_1_ACE_Data_100.xlsx"))
experiment_2_100_df = pd.read_excel(os.path.join(experiment, "Student_2_ACE_Data_100.xlsx"))
experiment_3_100_df = pd.read_excel(os.path.join(experiment, "Student_3_ACE_Data_100.xlsx"))
experiment_4_100_df = pd.read_excel(os.path.join(experiment, "Student_4_ACE_Data_100.xlsx"))

experiment_0_50_df = pd.read_excel(os.path.join(experiment, "Student_0_ACE_Data_50.xlsx"))
experiment_1_50_df = pd.read_excel(os.path.join(experiment, "Student_1_ACE_Data_50.xlsx"))
experiment_2_50_df = pd.read_excel(os.path.join(experiment, "Student_2_ACE_Data_50.xlsx"))
experiment_3_50_df = pd.read_excel(os.path.join(experiment, "Student_3_ACE_Data_50.xlsx"))
experiment_4_50_df = pd.read_excel(os.path.join(experiment, "Student_4_ACE_Data_50.xlsx"))

experiment_0_25_df = pd.read_excel(os.path.join(experiment, "Student_0_ACE_Data_25.xlsx"))
experiment_1_25_df = pd.read_excel(os.path.join(experiment, "Student_1_ACE_Data_25.xlsx"))
experiment_2_25_df = pd.read_excel(os.path.join(experiment, "Student_2_ACE_Data_25.xlsx"))
experiment_3_25_df = pd.read_excel(os.path.join(experiment, "Student_3_ACE_Data_25.xlsx"))
experiment_4_25_df = pd.read_excel(os.path.join(experiment, "Student_4_ACE_Data_25.xlsx"))

experiment_0_100 = experiment_0_100_df.to_numpy()
experiment_1_100 = experiment_1_100_df.to_numpy()
experiment_2_100 = experiment_2_100_df.to_numpy()
experiment_3_100 = experiment_3_100_df.to_numpy()
experiment_4_100 = experiment_4_100_df.to_numpy()

experiment_0_50 = experiment_0_50_df.to_numpy()
experiment_1_50 = experiment_1_50_df.to_numpy()
experiment_2_50 = experiment_2_50_df.to_numpy()
experiment_3_50 = experiment_3_50_df.to_numpy()
experiment_4_50 = experiment_4_50_df.to_numpy()

experiment_0_25 = experiment_0_25_df.to_numpy()
experiment_1_25 = experiment_1_25_df.to_numpy()
experiment_2_25 = experiment_2_25_df.to_numpy()
experiment_3_25 = experiment_3_25_df.to_numpy()
experiment_4_25 = experiment_4_25_df.to_numpy()

#build plot for student 0
plt.scatter(experiment_0_25[:, 0], experiment_0_25[:, 1], color="red", marker="o", label="ACE - 25 KPs")
plt.scatter(experiment_0_50[:, 0], experiment_0_50[:, 1], color="green", marker="o", label="ACE - 50 KPs")
plt.scatter(experiment_0_100[:, 0], experiment_0_100[:, 1], color="blue", marker="o", label="ACE - 100 KPs")
plt.title("Student 0 - ACE Time vs Adaptivity Score")
plt.xlabel("Time in Seconds")
plt.ylabel("Adaptivity Score")
plt.legend()
plt.savefig(data_folder + "Student_0_ACE.png")
plt.close()

plt.scatter(experiment_1_25[:, 0], experiment_1_25[:, 1], color="red", marker="o", label="ACE - 25 KPs")
plt.scatter(experiment_1_50[:, 0], experiment_1_50[:, 1], color="green", marker="o", label="ACE - 50 KPs")
plt.scatter(experiment_1_100[:, 0], experiment_1_100[:, 1], color="blue", marker="o", label="ACE - 100 KPs")
plt.title("Student 1 - ACE Time vs Adaptivity Score")
plt.xlabel("Time in Seconds")
plt.ylabel("Adaptivity Score")
plt.legend()
plt.savefig(data_folder + "Student_1_ACE.png")
plt.close()

plt.scatter(experiment_2_25[:, 0], experiment_2_25[:, 1], color="red", marker="o", label="ACE - 25 KPs")
plt.scatter(experiment_2_50[:, 0], experiment_2_50[:, 1], color="green", marker="o", label="ACE - 50 KPs")
plt.scatter(experiment_2_100[:, 0], experiment_2_100[:, 1], color="blue", marker="o", label="ACE - 100 KPs")
plt.title("Student 2 - ACE Time vs Adaptivity Score")
plt.xlabel("Time in Seconds")
plt.ylabel("Adaptivity Score")
plt.legend()
plt.savefig(data_folder + "Student_2_ACE.png")
plt.close()

plt.scatter(experiment_3_25[:, 0], experiment_3_25[:, 1], color="red", marker="o", label="ACE - 25 KPs")
plt.scatter(experiment_3_50[:, 0], experiment_3_50[:, 1], color="green", marker="o", label="ACE - 50 KPs")
plt.scatter(experiment_3_100[:, 0], experiment_3_100[:, 1], color="blue", marker="o", label="ACE - 100 KPs")
plt.title("Student 3 - ACE Time vs Adaptivity Score")
plt.xlabel("Time in Seconds")
plt.ylabel("Adaptivity Score")
plt.legend()
plt.savefig(data_folder + "Student_3_ACE.png")
plt.close()

plt.scatter(experiment_4_25[:, 0], experiment_4_25[:, 1], color="red", marker="o", label="ACE - 25 KPs")
plt.scatter(experiment_4_50[:, 0], experiment_4_50[:, 1], color="green", marker="o", label="ACE - 50 KPs")
plt.scatter(experiment_4_100[:, 0], experiment_4_100[:, 1], color="blue", marker="o", label="ACE - 100 KPs")
plt.title("Student 4 - ACE Time vs Adaptivity Score")
plt.xlabel("Time in Seconds")
plt.ylabel("Adaptivity Score")
plt.legend()
plt.savefig(data_folder + "Student_4_ACE.png")
plt.close()

# fig, axs = plt.subplots(2,3)

# # Plot the ACE - KP Population = 25
# axs[0,0].scatter(experiment_0_25[:, 0], experiment_0_25[:, 1], color="red", marker="o", label="ACE - 25 KPs")

# # Plot the experiment_0 array with blue dots and label "ACE"
# axs[0,0].scatter(experiment_0_50[:, 0], experiment_0_50[:, 1], color="green", marker="o", label="ACE - 50 KPs")

# # Plot the experiment_0 array with blue dots and label "ACE"
# axs[0,0].scatter(experiment_0_100[:, 0], experiment_0_100[:, 1], color="blue", marker="o", label="ACE - 100 KPs")

# # Set the labels of the x and y axes
# axs[0,0].set_xlabel("Time in Seconds")
# axs[0,0].set_ylabel("Adaptivity Score")

# # Show the legend
# #axs[0,0].legend()
# axs[0,0].set_title("Student 0")


# fig.tight_layout()

# # Create a list of labels for the legend
# labels = ["ACE - 25 KPs", "ACE - 50 KPs", "ACE - 100 KPs"]

# # Create a legend for the figure
# fig.legend(labels=labels, loc="lower right", bbox_to_anchor=(1, 0.1))
# # Set the title of the figure
# fig.suptitle("ACE Adaptivity Score vs Time for KP Populations of 25, 50, 100", fontsize=16)
# fig.subplots_adjust(top=0.85)
# # Delete the bottom right axis
# fig.delaxes(axs[1, 2])

# fig.savefig(data_folder + "combined_plot_ACE.png")