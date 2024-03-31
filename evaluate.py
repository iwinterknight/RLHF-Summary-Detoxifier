import os
from transformers import AutoTokenizer
from util.evaluation_utils import create_toxicity_evaluator, evaluate_toxicity
from util.model_utils import create_peft_model, create_ppo_and_ref_model, create_reward_model_pipeline
from data.load_dataset import load_dataset
from trl.core import LengthSampler
import tqdm
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--dataset_name', type=str, default="./cache")
    parser.add_argument('--toxicity_model_name', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--saved_model_path', type=bool, default=False, help='')
    args = parser.parse_args()
    return args


def quantitative_evaluate(ppo_model, ref_model, toxicity_model_name, tokenizer, dataset):
    toxicity_evaluator = create_toxicity_evaluator(toxicity_model_name)

    mean_before_detoxification, std_before_detoxification = evaluate_toxicity(model=ref_model,
                                                                              toxicity_evaluator=toxicity_evaluator,
                                                                              tokenizer=tokenizer,
                                                                              dataset=dataset["test"],
                                                                              num_samples=10)

    mean_after_detoxification, std_after_detoxification = evaluate_toxicity(model=ppo_model,
                                                                            toxicity_evaluator=toxicity_evaluator,
                                                                            tokenizer=tokenizer,
                                                                            dataset=dataset["test"],
                                                                            num_samples=10)
    print(f'toxicity [mean, std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')
    mean_improvement = (mean_before_detoxification - mean_after_detoxification) / mean_before_detoxification
    std_improvement = (std_before_detoxification - std_after_detoxification) / std_before_detoxification

    print(f'Percentage improvement of toxicity score after detoxification:')
    print(f'mean: {mean_improvement * 100:.2f}%')
    print(f'std: {std_improvement * 100:.2f}%')


def qualitative_evaluation(ppo_model, ref_model, dataset, tokenizer, saved_model_path, toxicity_model_name):
    if not os.listdir(saved_model_path):
        print("Saved PPO and Reference Models not found!!")
        return
    tokenizer.load_pretrained(os.path.join(saved_model_path, "ppo_model"))
    ppo_model.load_pretrained(os.path.join(saved_model_path, "ppo_model"))
    ref_model.load_pretrained(os.path.join(saved_model_path, "ref_model"))

    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    batch_size = 20
    compare_results = {}

    df_batch = dataset["test"][0:batch_size]

    compare_results["query"] = df_batch["query"]
    prompt_tensors = df_batch["input_ids"]

    summary_tensors_ref = []
    summary_tensors = []

    generation_kwargs = {
        "min_length": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True
    }

    reward_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "none",  # You want the raw logits without softmax.
        "batch_size": 16
    }

    rm_pipe = create_reward_model_pipeline(toxicity_model_name, device)

    # Get response from ppo and base model.
    for i in tqdm(range(batch_size)):
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

        summary = ref_model.generate(
            input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
            **generation_kwargs
        ).squeeze()[-gen_len:]
        summary_tensors_ref.append(summary)

        summary = ppo_model.generate(
            input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
            **generation_kwargs
        ).squeeze()[-gen_len:]
        summary_tensors.append(summary)

    # Decode responses.
    compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
    compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

    # Sentiment analysis of query/response pairs before/after.
    not_hate_index = 0
    texts_before = [d + s for d, s in zip(compare_results["query"], compare_results["response_before"])]
    rewards_before = rm_pipe(texts_before, **reward_kwargs)
    compare_results["reward_before"] = [reward[not_hate_index]["score"] for reward in rewards_before]

    texts_after = [d + s for d, s in zip(compare_results["query"], compare_results["response_after"])]
    rewards_after = rm_pipe(texts_after, **reward_kwargs)
    compare_results["reward_after"] = [reward[not_hate_index]["score"] for reward in rewards_after]


def evaluate(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    toxicity_model_name = args.toxicity_model_name
    saved_model_path = args.saved_model_path

    dataset = load_dataset(model_name, dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    peft_model = create_peft_model(model_name)
    ppo_model, ref_model = create_ppo_and_ref_model(peft_model)

    qualitative_evaluation(ppo_model, ref_model, dataset, tokenizer, saved_model_path, toxicity_model_name)
    quantitative_evaluate(ppo_model, ref_model, toxicity_model_name, tokenizer, dataset)
