import os
from typing import List, Dict
import torch

from model import Model
from transformers import pipeline, AutoTokenizer, BertForTokenClassification, BertForSequenceClassification, AutoModelForSequenceClassification


def build_model(device: str) -> Model:

    er_model = "../../model/er/"

    rc_model = "../../model/rc/"

    er_model_path = os.path.join(os.path.dirname(__file__), er_model)

    rc_model_path = os.path.join(os.path.dirname(__file__), rc_model)

    return StudentModel(er_model_path, rc_model_path, device)


class StudentModel(Model):
    def __init__(self, 
                er_model_path:str, 
                rc_model_path:str, 
                device:str="cpu"):

        self.er_model = BertForTokenClassification.from_pretrained(er_model_path)
        self.er_tokenizer = AutoTokenizer.from_pretrained(er_model_path)

        self.rc_model = BertForSequenceClassification.from_pretrained(rc_model_path)
        self.rc_tokenizer = AutoTokenizer.from_pretrained(rc_model_path)

    def predict(self, tokens: List[List[str]]) -> List[List[Dict]]:

        er_input_batch = self.er_tokenizer(
            [sentence for sentence in tokens],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
        )

        with torch.no_grad():
            output = self.er_model(**er_input_batch)
            logits = output.logits

        er_predictions = logits.argmax(dim=-1)

        # Creating tokens-candidates tuples
        tokens_candidates = [] 
        for i, sentence in enumerate(tokens):
            
            er_prediction = er_predictions[i]

            er_prediction = [self.er_model.config.id2label[elem.item()] for elem in er_prediction]

            word_ids = er_input_batch.word_ids(batch_index=i)

            filtered_er_prediction = []
            for token_id, token in enumerate(sentence):

                input_id = word_ids.index(token_id)

                pred = er_prediction[input_id]

                filtered_er_prediction.append(pred)

            candidates = self.relation_tuple_candidates(filtered_er_prediction)

            sample = {
                        "tokens": sentence,
                        "candidates": candidates
                    }

            tokens_candidates.append(sample)


        # Preparing batch for relation classification
        rc_batch = self.rc_batch(tokens_candidates)

        rc_input_batch = self.rc_tokenizer(
            [sentence for sentence in rc_batch],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
        )

        with torch.no_grad():
            output = self.rc_model(**rc_input_batch)
            logits = output.logits

        rc_probabilities = torch.softmax(logits, dim=-1)


        selects_batch = []
        idx = 0
        for sample in tokens_candidates:

            candidates = sample["candidates"]

            selects = []
            for candidate in candidates:

                probabilities = rc_probabilities[idx].tolist()

                max_prob = max(probabilities)

                if max_prob > 0.9:
                    max_prob_idx = probabilities.index(max_prob)

                    relation = self.rc_model.config.id2label[max_prob_idx]

                    select = {
                      "subject": candidate["subject"],
                      "relation": relation,
                      "object": candidate["object"]
                    }

                    selects.append(select)

                idx += 1

            selects_batch.append(selects)

        return selects_batch


    def relation_tuple_candidates(self, filtered_er_prediction):

        start_idx = None    
        end_idx = None        
        entities = []

        for label_id, label in enumerate(filtered_er_prediction):

            if label=="B-ENT":
                if start_idx != None:
                    end_idx = label_id

                    entity = {"start_idx": start_idx, "end_idx": end_idx}
                    entities.append(entity)

                start_idx = label_id

            elif label=="O":
                if start_idx != None:
                    end_idx = label_id

                    entity = {"start_idx": start_idx, "end_idx": end_idx}
                    entities.append(entity)
                
                    start_idx = None


        candidates = []
        # preparing input to entity r 
        for subj in entities:
            for obj in entities:
                if subj != obj:

                    candidate = {

                        "subject": 
                        {
                            "start_idx": subj["start_idx"],
                            "end_idx": subj["end_idx"]
                        },
                        "object":
                        {
                            "start_idx": obj["start_idx"],
                            "end_idx": obj["end_idx"]
                        }
                    }

                    candidates.append(candidate)

        return candidates



    def rc_batch(self, batch) -> Dict[str, torch.Tensor]:

        tokens_batch = [sentence["tokens"] for sentence in batch]

        candidates_batch = [sentence["candidates"] for sentence in batch]

        new_tokens_batch = []

        for i, candidates in enumerate(candidates_batch):

            # obtains the word_ids of the i-th sentence
            tokens = tokens_batch[i]

            for item in candidates:

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

        return new_tokens_batch