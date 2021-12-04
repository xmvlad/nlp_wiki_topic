# Introduction
In this project dataset from Wikipedia was collected, it contains top level topic names from Wikipedia articles and related text. Few common NLP models was trained on this dataset.

# Download dataset
Clone https://github.com/xmvlad/nlp_wiki_topic_dataset that contains wiki_topic_dataset.tar.gz and unpack it to __dataset__ folder if you wan't to run models from this project.

# Build dataset
To build dataset you need to clone this repository, then get Wikipedia dump. English Wikipedia xml dumps located at https://dumps.wikimedia.org/enwiki/. To build current version of dataset this https://dumps.wikimedia.org/enwiki/20211201/enwiki-20211201-pages-articles-multistream.xml.bz2 (18.9GB) dump was used. Next you need to put this dump to directory __dataset/dump__ and unpack it using __"bzip2 -d -k dump_name.bz2"__ command. Finaly run __dataset/build_dataset.sh__ , this script contain comments about steps of data processing and parameters that can be tuned. To rebuild dataset you need aproximately 100GB to store initial Wikipedia dump + 30-40GB for preprocessing, it takes approximately 3-4 hours on decent desktop with SSD.

# Dataset build parameters
This parameters can be changed in __dataset/build_dataset.sh__ :

- TOP_N_TOPICS=65 - number of topics selected from list sorted by number of samples 
- MIN_SIZE_BYTES=384 - minimum number of topic text size
- DATASET_SAMPLE_SIZE=12000 - number of samples per topic in dataset, top topics was heavily unbalanced, so undersampling was used to tackle this problem
- TRAIN_TEST_FRACTION=0.85 - train/test fraction split for final dataset

# Model Results

| Model              | Accuracy | F1 score |
|------------------- |----------|----------|
| LogReg TF-IDF      |   0.626  |  0.621   |
| BERT fine-tuned    |   0.778  |  0.777   |
| RoBERTa fine-tuned |   0.784  |  0.784   |
