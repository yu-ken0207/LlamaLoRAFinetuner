from datasets import load_dataset

class DatasetLoader:
    def __init__(self, dataset_name, split="train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        return self.dataset

    def format_dataset(self, formatting_func, batched=True):
        if self.dataset:
            self.dataset = self.dataset.map(formatting_func, batched=batched)
        return self.dataset
