import pandas as pd
import numpy as np
import os
# Import matplotlib.pyplot library
import matplotlib.pyplot as plt
import seaborn as sns

experiment = "/home/sean/Desktop/PhD_Work/PhD_Work/Experiment/"
data_folder = "/home/sean/Desktop/PhD_Work/PhD_Work/Data/"

experiment_df = pd.read_csv(os.path.join(experiment, "dfs_experiment_baseline.csv"))
experiment_df = experiment_df.drop('Best_LP', axis=1)

experiment_optimal_df = pd.read_csv(os.path.join(experiment, "dfs_experiment_run2.csv"))
experiment_optimal_df = experiment_optimal_df.drop('Best_LP', axis=1)

# Plot the scatterplot with colors and labels by student_id
for student, group in experiment_df.groupby("Student_id"):
    plt.plot(group["Num_total_KP"], group["Best_AS"], label="B_S " +str(student))

# Plot the scatterplot with colors for experiment_optimal_df
# Plot the scatterplot with colors and labels by student_id
for student, group in experiment_optimal_df.groupby("Student_id"):
    plt.plot(group["Num_total_KP"], group["Best_AS"], label="O_S " +str(student))

#sns.scatterplot(data=experiment_df, x="Num_total_KP", y="Best_AS", palette="bright", hue="Student_id")
# Add labels and title
plt.xlabel("Number of total KPs")
plt.ylabel("Optimal Adaptivity Score")
plt.title("Optimal Adaptivity Score vs Number of KPs by student_id")
plt.legend(bbox_to_anchor=(1.14,0), loc='lower right')
# Save plot as image file
plt.savefig(data_folder+"adaptivity_score_vs_numKPs_baseline_comparison.png")
plt.close()
#
# # Plot the scatterplot with colors and labels by student_id
# for student, group in experiment_df.groupby("Student_id"):
#     plt.plot(group["Num_total_KP"], group["Alg_time"], label="Student " +str(student))
#
# plt.xlabel("Number of  total KPs")
# plt.ylabel("Time to calculate optimal score in seconds")
# plt.title("Time to calculate vs Number of KPs by student_id")
# plt.legend()
# # Save plot as image file
# plt.savefig(data_folder+"Alg_time_vs_numKPs_baseline.png")
# plt.close()
#
# # Plot the scatterplot with colors and labels by student_id
# for student, group in experiment_df.groupby("Student_id"):
#     plt.plot(group["Num_total_KP"], group["Num_KP_Explored"], label="Student " +str(student))
#
# plt.xlabel("Number of  total KPs")
# plt.ylabel("Number of KPs explored")
# plt.title("Number of KPs Explored vs Number of KPs by student_id")
# plt.legend()
# # Save plot as image file
# plt.savefig(data_folder+"Num_KPs_explored_vs_numKPs_baseline.png")
# plt.close()

# grouped_df = experiment_df.groupby("Student_id").mean()
#
# plt.scatter(grouped_df["Num_total_KP"], grouped_df["Best_AS"], c=grouped_df.index)
# plt.legend(grouped_df.index, title="Student_id")
#
# # Step 5: Add labels and title
# plt.xlabel("Num_total_KP")
# plt.ylabel("Best_AS")
# plt.title("Scatterplot of Best_AS vs Num_total_KP by student_id")
# plt.plot()
# # Step 6: Save plot as image file
# plt.savefig(data_folder+"adaptivity_score_vs_numKPs.png")