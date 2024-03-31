from datasets import load_dataset
from transformers import AutoTokenizer


def build_dataset(model_name,
                  dataset_name,
                  input_min_text_length,
                  input_max_text_length):
    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model_name (str): Tokenizer model name.
    - dataset_name (str): Name of the dataset to load.
    - input_min_text_length (int): Minimum length of the dialogues.
    - input_max_text_length (int): Maximum length of the dialogues.

    Returns:
    - dataset_splits (datasets.dataset_dict.DatasetDict): Preprocessed dataset containing train and test parts.
    """
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(
        lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length,
        batched=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    def tokenize(sample):
        prompt = f"""
                Summarize the following conversation.
                {sample["dialogue"]}
                Summary:
            """
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)

    return dataset_splits


def load_dataset(model_name, dataset_name):
    dataset = build_dataset(model_name=model_name,
                            dataset_name=dataset_name,
                            input_min_text_length=200,
                            input_max_text_length=1000)
    return dataset