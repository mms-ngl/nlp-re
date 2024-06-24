import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

from transformers import AutoTokenizer, BertForSequenceClassification, AdamW

from typing import List, Dict
import matplotlib.pyplot as plt


train_path = "../../data/train.jsonl"
dev_path = "../../data/dev.jsonl"
relations_path = "../../data/relations2id.json"

save_path = "/../../model/rc/"

model_name = "textattack/bert-base-uncased-yelp-polarity"


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


def read_relations(path: str):

    with open(path) as json_file:
        data = json.load(json_file)

    return data


# here we define our collate function
def collate_fn(batch) -> Dict[str, torch.Tensor]:

    tokens_batch = [sentence["tokens"] for sentence in batch]

    relations_batch = [sentence["relations"] for sentence in batch]

    new_tokens_batch = []

    labels = []
    for i, relations in enumerate(relations_batch):

        # obtains the word_ids of the i-th sentence
        tokens = tokens_batch[i]

        for item in relations:

            new_tokens = []

            for idx, token in enumerate(tokens):
                if idx == item["subject"]["start_idx"]:

                    new_tokens.append("<SUBJ>")

                elif idx == item["subject"]["end_idx"]:

                    new_tokens.append("</SUBJ>")

                elif idx == item["object"]["start_idx"]:

                    new_tokens.append("<OBJ>")

                elif idx == item["object"]["end_idx"]:

                    new_tokens.append("</OBJ>")

                new_tokens.append(token)


            new_tokens_batch.append(new_tokens)

            label = label2id[item["relation"]]
            labels.append(label)

    input_batch = tokenizer(
        [sentence for sentence in new_tokens_batch],
        return_tensors="pt",
        padding=True,
        # We use this argument because the texts in our dataset are lists of words.
        is_split_into_words=True,
    )

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

            with open('train_losses.txt', 'w') as f:
                f.write(str(training_loss))

            with open('valid_losses.txt', 'w') as f:
                f.write(str(validation_loss))

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

    relations = read_relations(relations_path)
    label2id = relations
    id2label = {v:k for k,v in relations.items()}


    train_dataset  = read_dataset(train_path)
    dev_dataset  = read_dataset(dev_path)


    rc_model = BertForSequenceClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    rc_model.to(device)

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
      model = rc_model,
      optimizer = AdamW(rc_model.parameters(), lr=5e-5)
    )

    training_loss, validation_loss = trainer.train_model(train_dataloader, val_dataloader, training_number)


    final_save = save_path + str("final")
    os.mkdir(final_save)
    rc_model.save_pretrained(final_save)
    tokenizer.save_pretrained(final_save)


    # losses graphs
    plt.plot(range(1, training_number+1), training_loss, label='Train')
    plt.plot(range(1, training_number+1), validation_loss, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()

