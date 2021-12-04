import torch
import random
import os
import sys
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME="roberta-base"

class WikiTopicfDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        topic_dirs = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
        self.class_to_idx = {c : i for i, c in enumerate(topic_dirs) }
        self.idx_to_class = {i : c for c, i in self.class_to_idx.items() }

        self.all_files = []
        for topic_name in topic_dirs:
            all_topic_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(dataset_path, topic_name)) for f in filenames])
            self.all_files.extend(all_topic_files)


    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        with open(file_path, "r") as f:
            lines = f.readlines()

        text = lines[1].strip()
        text_tokens = self.tokenizer([text], padding="max_length", truncation=True)
        item = {key: torch.tensor(val[0]) for key, val in text_tokens.items()}
        item["labels"] = self.class_to_idx[lines[0].split(";")[0]]
        return item
    
    def __len__(self):
        return len(self.all_files)

def compute_metric(eval_pred):
    logits, labels = eval_pred    
    pred = logits.argmax(1)
    return {"accuracy": accuracy_score(labels, pred),
            "f1_score": f1_score(labels, pred, average="macro") }



random.seed(42)
torch.manual_seed(42)

train_dataset = WikiTopicfDataset("../dataset/dataset_final/train")
test_dataset = WikiTopicfDataset("../dataset/dataset_final/test")

torch.cuda.empty_cache() 

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(train_dataset.class_to_idx))

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    gradient_accumulation_steps=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)

trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  compute_metrics=compute_metric)

trainer.train()





