import sys
import os
import random
from shutil import copyfile

def main():
    if len(sys.argv) != 4:
        print("Incorrect number of arguments. split_train_test.py dataset_sample output_path sample_fraction")
        return 1

    random.seed(42)

    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    train_frac = float(sys.argv[3])

    topic_dirs = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

    for topic_name in topic_dirs:
        all_topic_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(dataset_path, topic_name)) for f in filenames])
        random.shuffle(all_topic_files)
        s = int(len(all_topic_files)*train_frac)
        train_files = all_topic_files[0:s]
        test_files = all_topic_files[s:]

        for file_path in train_files:
            output_topic_path = os.path.join(output_path, "train",  topic_name)
            os.makedirs(output_topic_path, exist_ok=True)
            copyfile(file_path, os.path.join(output_topic_path, os.path.basename(file_path)))

        for file_path in test_files:
            output_topic_path = os.path.join(output_path, "test",  topic_name)
            os.makedirs(output_topic_path, exist_ok=True)
            copyfile(file_path, os.path.join(output_topic_path, os.path.basename(file_path)))

sys.exit(main())
