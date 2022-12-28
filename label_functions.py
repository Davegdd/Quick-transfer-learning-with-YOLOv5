# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 19:56:38 2022

@author: David
"""

import os
import json



def check_keys(path):
    # Open json file outputted by Megadetector and display keys    
    with open(path, encoding='utf-8') as json_file:
        train_annotations =json.load(json_file)
    return train_annotations.keys()


def create_label_files(annotations, output_path):
    # Creation of label files in YOLOv5 format from Megadetector json
    for i in range(len(annotations['images'])):
        id = annotations['images'][i]['file'][:-4]

        # Extract the box(es) coordinates and write them to a file with the same name of the image
        try:
            with open(f'{output_path}/{id}.txt', 'w') as f:
                for j in range(len(annotations['images'][i]['detections'])):
                    bbox = annotations['images'][i]['detections'][j]['bbox']
                    # The integer below (“0”) has to be changed for each class: 0 oryx, 1 ostrich, 2 wildebeest…
                    f.write("0 " + str(bbox)[1:-1].replace(',', "") + '\n')

        # Error handling in case there are no detections
        except KeyError:
            print(f'{id} ignored.')


def transform_coordinates(directory):
    # Transformation from Megadetector to YOLOv5 coordinates
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Open the file and read its contents
            with open(os.path.join(directory, filename), 'r') as f:
                contents = f.read()

            # Check if the file is empty
            if not contents:
                print(f'Error: {filename} is empty')
                continue

            # Split the contents into a list of lines
            lines = contents.split('\n')

            # Iterate over the lines in the file
            for i, line in enumerate(lines):
                # Split the line into a list of words
                words = line.split()

                # Transform using the formulas: x = x + width/2 and y = y + height/2
                try:
                    words[1] = str(float(words[1]) + float(words[3]) / 2)
                    words[2] = str(float(words[2]) + float(words[4]) / 2)

                except (IndexError, ValueError):
                    # If the line does not have enough words or the words are not numbers, skip it
                    continue

                # Join the words back into a single string
                lines[i] = ' '.join(words)

            # Join the lines back into a single string
            new_contents = '\n'.join(lines)

            # Write the modified contents back to the file
            with open(os.path.join(directory, filename), 'w') as f:
                f.write(new_contents)

