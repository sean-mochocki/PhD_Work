#!/bin/bash
# create a text file to store the output
output_file="test_output_3.txt"
# loop through the four profiles
for profile in 2 3 4 5 7 8 11
do
  # construct the file name from the profile and the number
  file="xcsp3_creator_2_student_${profile}.xml" 
  echo "Running command for $file" >> $output_file
  # run the command and redirect both stdout and stderr to the output file
  java -jar ACE-2.1.jar $file -t=600s -varh=Wdeg -wt=chs >> $output_file 2>&1
  # append a separator line to the output file
  echo "----------------------------------------" >> $output_file
done
# print a message to the terminal when done
echo "All commands completed. Check the output file for results."

