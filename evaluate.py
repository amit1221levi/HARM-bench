import argparse
import json
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_json(filename):
    """Load json file"""
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data


def save_dictlist_to_json(mydictlist, filename):
    """Save a list of dictionaries to json file"""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4) 
    f.close()


class TestDataset(Dataset):
    """Simple dataset module for testing the reward model"""
    def __init__(self, test_ds):
        self.ds = test_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ix):
        return self.ds[ix]


class RewardModel(torch.nn.Module):
    """
    Wrapper class for the reward model, 
    which handles loading the model and tokenizers, 
    and the forward pass for final predictions
    """
    def __init__(self, model_path):
        super().__init__()

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.config = self.model.config.from_pretrained(model_path)
        self.config = self.model.config

        # Set the label2id and id2label mappings for the model config
        self.model.config.label2id = {'negative': 0, 'positive': 1}
        self.model.config.id2label = {0: 'negative', 1: 'positive'}

    def threshold(self, n):
        """
        Threshold function to evaluate regression models, where scores > 0 
        are positive and scores <= 0 are negative,

        The actual grading evaluation scheme may not rely on such thresholding, 
        this serves more as an example.
        """
        if n > 0:
            return 1
        else:
            return 0

    def forward(self, encoded):
        """
        Forward pass of the model to get predictions from the model logits
        Args:
            encoded (dict): Dictionary of tensors containing the encoded input
        """
        # Get the model logits by passing the encoded input to the model
        outputs = self.model(**encoded)
        logits = outputs.logits.detach().cpu()
        
        # The code here converts model logits to predictions

        # (1) In this evaluation example, for regression models, 
        # we threshold the logits (scores) to get predictions,
        # where scores > 0 are positive and scores <= 0 are negative.
        # Your actual training / evaluation scheme doesn't need to do this.

        # (2) For classification models, we can take the argmax of the logits 
        # to get predictions, where the argmax is the index of the highest logit.

        if self.config.problem_type == "regression":        
            return [self.threshold(n) for n in logits] 
        elif self.config.problem_type == "classification":
            return  torch.argmax(logits, dim=1).tolist()
        else:
            raise ValueError("problem_type must be either 'regression' or 'classification'")

class Evaluator:
    def __init__(self, model_path, ds_test):
        # Load the model and dataset
        self.load_model(model_path)
        self.ds_test = ds_test
        self.verify_ds_format(ds_test)
        self.dataset = TestDataset(ds_test)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False)

    def verify_ds_format(self, ds):
        """Verify that the dataset has the correct format"""
        assert "chat" in ds[0].keys(), "The dataset must have a 'chat' key"
        assert "label" in ds[0].keys(), "The dataset must have a 'label' key"

    def load_model(self, model_path):
        """Load the reward model and tokenizer"""
        self.model = RewardModel(model_path)
        self.tokenizer = self.model.tokenizer
        assert self.model.config.problem_type in ["regression", "classification"], "The model must be either a regression or classification model"

    def predict(self, batch):
        """Get predictions from the model"""
        # Encode the batch using the tokenizer
        encoded = self.tokenizer(
                batch['chat'],
                return_tensors="pt",
                truncation=True,
                padding=True)
        
        # Get predictions from the model
        scores = self.model(encoded)
        return scores

    def compute_accuracy(self, all_pred_labels, all_target_labels):
        """Compute the accuracy of the model predictions"""
        assert len(all_pred_labels) == len(all_target_labels), "The number of predictions must be equal to the number of targets"
        acc = accuracy_score(all_pred_labels, all_target_labels)
        return acc
    
    def evaluate(self):
        """Evaluate the model on the test dataset"""
        all_pred_labels, all_target_labels = [], []

        for batch in tqdm(self.dataloader):
            with torch.no_grad():
                scores = self.predict(batch)
                all_pred_labels.extend(
                    [self.model.config.id2label[score] for score in scores])
                all_target_labels.extend(batch["label"])

        acc = self.compute_accuracy(all_pred_labels, all_target_labels)
        print(f"Evaluation Complete, Accuracy: {acc}")

def save_hf_model(hf_model, hf_tokenizer, model_path):
    """Save the model and tokenizer to the specified path"""
    hf_model.save_pretrained(model_path)
    hf_model.config.save_pretrained(model_path)
    hf_tokenizer.save_pretrained(model_path)

def verify_dataset(dataset):
    """Verifies the key names and value types of the dataset"""
    for mydict in dataset:
        assert len(set(["entry_id", "label", "chat"]) - (set(mydict.keys()))) == 0
        assert type(mydict["entry_id"]) == int
        assert (mydict["label"] == "negative" or mydict["label"] == "positive")
        assert type(mydict["chat"]) == str
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/reward-model",
        help="Path to the model")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="m2_reward_dataset_example.json",
        help="Path to the test dataset")
    args = parser.parse_args()

    hf_pretrained_model_name = "OpenAssistant/reward-model-deberta-v3-base"

    # Example code to load a pretrained reward model
    model = AutoModelForSequenceClassification.from_pretrained(hf_pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_name)
    
    # Example code to save your reward model
    save_hf_model(model, tokenizer, args.model_path)
    
    # Example code of how we will load your dataset
    reward_dataset = load_json(args.data_path)
    verify_dataset(reward_dataset)

    # Example of how we will load your model
    reward = RewardModel(args.model_path)
    evaluator = Evaluator(args.model_path, reward_dataset)
    evaluator.evaluate()