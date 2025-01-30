import os
import csv
import pandas as pd

num_LMs = 100
num_KNs = 20

columns = ['LM_Name', 'Multimedia', 'Coherence', 'Segmenting', 'Worked_Example', 'Signaling', 'Spatial_Contiguity', 'Temporal_Contiguity', 'Modality', 'Redundancy',
           'Personalization', 'Voice', 'Sourcing', 'Difficulty', 'Duration', 'Media_Type', 'Content_type', 'Publication_Date', 'Popularity', 'KNs_Covered']

#for i in range(num_LMs):


#data = {
#        "Student_id": int(student_profile_id),
#        "Best_LP": str(LP),
#        "Best_AS": round(score, 1),
#        "LP_Time": path_time,
#        "Alg_time": total_time,
#        "Num_KP_Explored": int(num_knowledge_paths_explored),
#        "Num_total_KP": int(num_kp),
#    }

#data = pd.DataFrame(data, index=[0])
#experiment_df = pd.concat([experiment_df, data], ignore_index=True)
#Experiment = "/home/sean/Desktop/PhD_Work/PhD_Work/Experiment/dfs_experiment_consolidated_profile_student_11.csv"
#experiment_df.to_csv(Experiment)