import os
import re
from collections import Counter, OrderedDict
import math
import torch
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score


class TfIdfVectorizer:
    def __init__(self, max_vocab=0, lower=True, tokenizer_pattern=r"(?i)\b[a-z]{2,}\b"):
        self.lower = lower
        self.tokenizer_pattern = re.compile(tokenizer_pattern)
        self.max_vocab = max_vocab
        self.vocab_df = OrderedDict()
        
    def __tokenize(self, text):
        terms = self.tokenizer_pattern.findall(text.lower() if self.lower else text)
        return terms
    
    def fit(self, texts):
        term_id = 0
        for doc_idx, doc in enumerate(texts):
            tokenized = self.__tokenize(doc)
            for term in tokenized:
                if term not in self.vocab_df:
                    self.vocab_df[term] = {}
                    self.vocab_df[term]["doc_ids"] = {doc_idx}
                    self.vocab_df[term]["doc_count"] = 1
                    self.vocab_df[term]["id"] = term_id
                    self.vocab_df[term]["term_num"] = 1
                    term_id += 1
                elif doc_idx not in self.vocab_df[term]["doc_ids"]:
                    self.vocab_df[term]["doc_ids"].add(doc_idx)
                    self.vocab_df[term]["doc_count"] += 1

                if term in self.vocab_df:
                    self.vocab_df[term]["term_num"] += 1

        for term in self.vocab_df.keys():
            del self.vocab_df[term]["doc_ids"]

        texts_len = len(texts)
        for term in self.vocab_df:
            self.vocab_df[term]["idf"] = math.log(texts_len / self.vocab_df[term]["doc_count"])

        if self.max_vocab > 0 and len(self.vocab_df) > self.max_vocab:
            min_term_num = sorted(self.vocab_df.items(), key=lambda x: x[1]["term_num"], reverse=True)[self.max_vocab][1]["term_num"]

            all_keys = [k for k in self.vocab_df]
            for term in all_keys:
                if self.vocab_df[term]["term_num"] <= min_term_num:
                    del self.vocab_df[term]

            for i, term in enumerate(self.vocab_df):
                self.vocab_df[term]["id"] = i

        print(f"Vocab size: {len(self.vocab_df)}")
        
        
    def transform(self, texts):
        values = []
        doc_indices = []
        term_indices = []
        for doc_idx, raw_doc in enumerate(texts):
            term_counter = {}
            for token in self.__tokenize(raw_doc):
                if token in self.vocab_df:
                    term = self.vocab_df[token]
                    term_idx = term["id"]
                    term_idf = term["idf"]
                    if term_idx not in term_counter:
                        term_counter[term_idx] = term_idf
                    else:
                        term_counter[term_idx] += term_idf
            term_indices.extend(term_counter.keys())
            values.extend(term_counter.values())
            doc_indices.extend([doc_idx] * len(term_counter))
        indices = torch.LongTensor([doc_indices, term_indices])
        values_tensor = torch.FloatTensor(values)
        tf_idf = torch.sparse.FloatTensor(indices, values_tensor, torch.Size([len(texts), len(self.vocab_df)])).to_dense()
        return tf_idf

    def get_term(self, term_idx):
        for k in vectorizer.vocab_df.keys():
            if vectorizer.vocab_df[k]["id"] == term_idx:
                return k


class WikiTopicTfIdfDataset(torch.utils.data.Dataset):
    def __init__(self, vectorizer, dataset_path):
        self.vectorizer = vectorizer
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

        item = {}
        item["vectors"] = self.vectorizer.transform([lines[1].strip()])[0, :]
        item["labels"] = self.class_to_idx[lines[0].split(";")[0]]
        return item
    
    def __len__(self):
        return len(self.all_files)

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

def get_all_texts(dataset_path):
    topic_dirs = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    all_texts = []
    for topic_name in topic_dirs:
        all_topic_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(dataset_path, topic_name)) for f in filenames])
        for file_path in all_topic_files:
            with open(file_path, "r") as f:
                text = f.readlines()[1].strip()
                all_texts.append(text)
    return all_texts

random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

train_dataset_path = "../dataset/dataset_final/train"
test_dataset_path = "../dataset/dataset_final/test"
batch_size = 64

print("Fitting tf-idf vectorizer.")
vectorizer = TfIdfVectorizer(max_vocab=5000)
vectorizer.fit(get_all_texts(train_dataset_path))

train_dataset = WikiTopicTfIdfDataset(vectorizer, train_dataset_path)
test_dataset = WikiTopicTfIdfDataset(vectorizer, test_dataset_path)

train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6)
test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=6)

class_num = len(train_dataset.class_to_idx)
model = LogisticRegressionModel(len(vectorizer.vocab_df), class_num)
model = model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5


for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    print(datetime.now())

    model.train()

    train_labels = []
    train_predict = []
    for data in train_data_loader:
        labels, vectors = data["labels"].to(DEVICE), data["vectors"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(vectors)
        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
        train_labels.extend(list(labels.detach().cpu().numpy()))
        train_predict.extend(list(predicted.detach().cpu().numpy()))

    train_labels_array = np.array(train_labels)
    train_predict_array = np.array(train_predict)

    train_accuracy = accuracy_score(train_labels_array, train_predict_array)
    train_f1_score = f1_score(train_labels_array, train_predict_array, average="macro")
    print(f"Train accuracy: {train_accuracy}")
    print(f"Train f1 score: {train_f1_score}")

    model.eval()
    test_labels = []
    test_predict = []
    for data in test_data_loader:
        labels, vectors = data["labels"].to(DEVICE), data["vectors"].to(DEVICE)
        
        outputs = model(vectors)
        _, predicted = torch.max(outputs.data, 1)

        test_labels.extend(list(labels.detach().cpu().numpy()))
        test_predict.extend(list(predicted.detach().cpu().numpy()))

    test_labels_array = np.array(test_labels)
    test_predict_array = np.array(test_predict)

    test_accuracy = accuracy_score(test_labels_array, test_predict_array)
    test_f1_score = f1_score(test_labels_array, test_predict_array, average="macro")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test f1 score: {test_f1_score}")
    