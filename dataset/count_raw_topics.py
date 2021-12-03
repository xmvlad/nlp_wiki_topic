#!/usr/bin/env python
# coding: utf-8


import os
import sys
import collections


def main():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments. count_raw_topics.py dataset_raw_path output_topic_count_file")
        return 1

    base_input_path = sys.argv[1]
    output_file_name = sys.argv[2]
    print("Finding all topic files.")
    all_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(base_input_path) for f in filenames])
    all_files_size = len(all_files)

    counter = collections.Counter()
    print("Topic count progress: ", end='')
    for i, file_path in enumerate(all_files):
        if i % (all_files_size//100) == 0:
            print("*", end='', flush=True)
        
        with open(file_path, "r") as file:
            topic_name = file.readline().split(";")[0]
            counter.update([topic_name])


    with open(output_file_name, "w") as topic_file:
        for topic_name, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            topic_file.write("%s;%d\n" % (topic_name, count))
    
    return 0;


sys.exit(main())

