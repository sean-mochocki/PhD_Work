#!/bin/bash
# create a text file to store the output
output_file="test_output.txt"
# loop through the five files and run the commands
for file in xcsp3_creator_profile_{0..4}_25_paths.xml
do
  # append the file name to the output file
  echo "Running command for $file" >> $output_file
  # run the command and redirect both stdout and stderr to the output file
  java -jar ACE-2.1.jar $file -t=600s >> $output_file 2>&1
  # append a separator line to the output file
  echo "----------------------------------------" >> $output_file
done
# print a message to the terminal when done
echo "All commands completed. Check the output file for results."
