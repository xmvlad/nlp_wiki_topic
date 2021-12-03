#!/usr/bin/env python
# coding: utf-8

import sys

def main():
    if len(sys.argv) != 4:
        print("Not enough parameters. extract_top_topics.py topic_raw_all_file topic_dataset_file top_n_topics")
        return 1

    input_topic_file = sys.argv[1]
    output_topic_file = sys.argv[2]
    select_top_n = int(sys.argv[3])

    with open(input_topic_file, "r") as topic_raw_file:
        all_lines = topic_raw_file.readlines()

    with open(output_topic_file, "w") as topic_dataset_file:
        for line in all_lines[0:select_top_n]:
            topic_dataset_file.write(line)


    return 0



sys.exit(main())




