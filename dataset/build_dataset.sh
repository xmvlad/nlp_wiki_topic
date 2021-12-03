#!/bin/bash

set -e
set -x

TOP_N_TOPICS=65
MIN_SIZE_BYTES=384
DATASET_SAMPLE_SIZE=12000
TRAIN_TEST_FRACTION=0.85

# Extract text form wikipedia xml dump
mkdir -p ./dump_text
python -m wikiextractor.WikiExtractor ./dump/*.xml --output ./dump_text

# Extract top level topics 
mkdir -p ./dataset_raw
python wiki_topic_extract.py ./dump_text ./dataset_raw

# Delete topic texts that smaller then MIN_SIZE_BYTES
python cleanup_dataset.py ./dataset_raw $MIN_SIZE_BYTES

# Count number of topic occurence in corpus
python count_raw_topics.py ./dataset_raw ./topic_count_raw.txt

# Select N most common topics
python get_top_topics.py ./topic_count_raw.txt ./topic_label.txt $TOP_N_TOPICS

# Extract most common topics from topic_label.txt
mkdir -p ./dataset_extract
python extract_dataset.py ./dataset_raw ./dataset_extract ./topic_label.txt

rm -rf ./dataset_extract/references
rm -rf ./dataset_extract/see_also
rm -rf ./dataset_extract/external_links

# Topics have significat class disbalance, therefore we undersample
mkdir -p ./dataset_sample
python sample_dataset.py ./dataset_extract ./dataset_sample $DATASET_SAMPLE_SIZE

# Split dataset into train an test parts
mkdir -p ./dataset_final
python split_train_test.py ./dataset_sample ./dataset_final $TRAIN_TEST_FRACTION

