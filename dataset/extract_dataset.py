#!/usr/bin/env python
# coding: utf-8


import os
import sys

def main():
    if len(sys.argv) != 4:
        print("Incorrect number of arguments. extract_dataset.py raw_dataset_path extract_dataset_path topic_label_file")
        return 1
    
    input_raw_dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    input_topic_file = sys.argv[3]

    with open(input_topic_file, "r") as topic_file:
        topic_name_all = {line.split(";")[0] for line in topic_file.readlines()}

    print("Finding all topic files")
    raw_file_paths = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(input_raw_dataset_path) for f in filenames])

    for topic_name in topic_name_all:
        topic_name_path = os.path.join(output_path, topic_name)

    raw_file_num_step = len(raw_file_paths) // 100
    print("Progress: ", end="")
    
    for i, raw_file_path in enumerate(raw_file_paths):
        if i % raw_file_num_step == 0:
            print("*", end="", flush=True)

        with open(raw_file_path, "r") as raw_file:
            label_line = raw_file.readline()
            text_line = raw_file.readline()    
            
        topic_name = label_line.split(";")[0]
        doc_id = int(label_line.split(";")[1])

        if not topic_name in topic_name_all:
            continue
            
        output_topic_path = os.path.join(output_path, topic_name)
        file_name = os.path.split(raw_file_path)[1]

        output_topic_docid_path = os.path.join(output_topic_path, "%03d" % (doc_id % 1000,))
        os.makedirs(output_topic_docid_path, exist_ok=True)

        output_file_path = os.path.join(output_topic_docid_path, file_name)
        
        with open(output_file_path, "w") as output_file:
            output_file.write(label_line)
            output_file.write(text_line)
    

    return 0




sys.exit(main())


