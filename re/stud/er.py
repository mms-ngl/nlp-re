import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

from transformers import AutoTokenizer, BertForTokenClassification, AdamW

from typing import List, Dict
import matplotlib.pyplot as plt


train_path = "../../data/train.jsonl"
dev_path = "../../data/dev.jsonl"

save_path = "../../model/er/"

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print("Using {}.".format(device))

batch_size = 8
training_number = 5


def read_dataset(path: str):
    dataset = []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    return dataset


# here we define our collate function
def collate_fn(batch) -> Dict[str, torch.Tensor]:

    input_batch = tokenizer(
        [sentence["tokens"] for sentence in batch],
        return_tensors="pt",
        padding=True,
        # We use this argument because the texts in our dataset are lists of words.
        is_split_into_words=True,
    )

    relations_batch = [sentence["relations"] for sentence in batch]

    labels = []

    for i, relations in enumerate(relations_batch):

        entities  = []
        for elem in relations:

            entity = list(range(elem["subject"]["start_idx"], elem["subject"]["end_idx"]))

            if entity not in entities:
                entities.append(entity)

            entity = list(range(elem["object"]["start_idx"], elem["object"]["end_idx"]))

            if entity not in entities:
                entities.append(entity)


        BI_entity = [elem for entity in entities for elem in entity]
        
        B_entity = [entity[0] for entity in entities]
        I_entity = list(set(BI_entity) - set(B_entity))

        # obtains the word_ids of the i-th sentence
        word_ids = input_batch.word_ids(batch_index=i)

        previous_word_idx = None

        label_ids = []

        for word_idx in word_ids:

            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)

            # We set the label for the first token of each word.
            # elif word_idx != previous_word_idx:
            elif word_idx != previous_word_idx:

                if word_idx in B_entity:

                    label_ids.append(label2id["B-ENT"])

                elif word_idx in I_entity:

                    label_ids.append(label2id["I-ENT"])
                else:
                    label_ids.append(label2id["O"])


            # For the other tokens in a word, we set the label to -100 so they are automatically
            # ignored in the loss function.
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    input_batch["labels"] = torch.as_tensor(labels)

    return input_batch


# Model Training
class Trainer():

    def __init__(self,
                 model,
                 optimizer):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """

        self.model = model
        self.optimizer = optimizer

    def train_model(self, train_dataset:Dataset, val_dataset:Dataset, epochs:int=1):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            val_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.
            epochs: the number of times to iterate over train_dataset.
        """

        training_loss, validation_loss = [], []

        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            train_loss = self.train(train_dataset)
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, train_loss))
            training_loss.append(train_loss)

            valid_loss = self.validate(val_dataset)
            print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch+1, valid_loss))
            validation_loss.append(valid_loss)

        return training_loss, validation_loss

    def train(self, train_dataset):
        """
        Args:
            train_dataset: the dataset to use to train the model.

        Returns:
            the average train loss over train_dataset.
        """

        train_loss = 0.0
        self.model.train()

        for batch in train_dataset:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()

            self.optimizer.step()

            train_loss += loss.tolist()

        return train_loss / len(train_dataset)

    def validate(self, val_dataset):
        """
        Args:
            val_dataset: the dataset to use to evaluate the model.

        Returns:
            the average validation loss over val_dataset.
        """

        valid_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in val_dataset:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss

                valid_loss += loss.tolist()

        return valid_loss / len(val_dataset)


if __name__ == "__main__":

    # Initializations and functions calls

    labels = {
        "O": 0,
        "B-ENT": 1,
        "I-ENT": 2
    }

    label2id = {n: i for i, n in enumerate(labels)}
    id2label = {i: n for n, i in label2id.items()}


    train_dataset  = read_dataset(train_path)
    dev_dataset  = read_dataset(dev_path)


    er_model = BertForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    er_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    trainer = Trainer(
      model = er_model,
      optimizer = AdamW(er_model.parameters(), lr=5e-5)
    )

    training_loss, validation_loss = trainer.train_model(train_dataloader, val_dataloader, training_number)


    final_save = save_path + str("final")
    os.mkdir(final_save)
    er_model.save_pretrained(final_save)
    tokenizer.save_pretrained(final_save)


    # losses graphs
    plt.plot(range(1, training_number+1), training_loss, label='Train')
    plt.plot(range(1, training_number+1), validation_loss, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()
