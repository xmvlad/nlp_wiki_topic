#!/usr/bin/env python
# coding: utf-8


import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments. cleanup_dataset.py dataset_extract_path size_in_bytes")
        return 1

    base_input_path = sys.argv[1]
    size_in_bytes = int(sys.argv[2])

    print("Finding all topic files.")
    all_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(base_input_path) for f in filenames])
    all_files_step = len(all_files) // 100

    print("Progress: ", end="")
    for i, file_path in enumerate(all_files):
        if (i % all_files_step) == 0:
            print("*", end="", flush=True)
        if os.path.getsize(file_path) < size_in_bytes:
            os.remove(file_path)
    return 0

sys.exit(main())
