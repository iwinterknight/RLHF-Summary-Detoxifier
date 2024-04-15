from util.train_utils import collator, create_peft_model, create_ppo_and_ref_model,create_reward_model_pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, DPOTrainer
from data.load_dataset import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.core import LengthSampler
import os
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
    parser.add_argument('--alignment_method', type=str, default="ppo", help='')
    args = parser.parse_args()
    return args


def create_ppo_trainer(model_name, dataset, ppo_model, ref_model, tokenizer):
    learning_rate = 1.41e-5
    max_ppo_epochs = 1
    mini_batch_size = 4
    batch_size = 16

    config = PPOConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        ppo_epochs=max_ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size
    )

    ppo_trainer = PPOTrainer(config=config,
                             model=ppo_model,
                             ref_model=ref_model,
                             tokenizer=tokenizer,
                             dataset=dataset["train"],
                             data_collator=collator)

    return ppo_trainer


def create_dpo_trainer(dataset, ppo_model, ref_model, tokenizer):
    training_args = TrainingArguments(
        output_dir='./data/output',
        num_train_epochs=5,
        per_device_train_batch_size=1,
        fp16=False,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        prediction_loss_only=True,
        learning_rate=1.41e-5
    )

    dpo_trainer = DPOTrainer(
        ppo_model,
        model_ref=ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    return dpo_trainer


def train(args):
    alignment_method = args.alignment_method
    if alignment_method == "ppo":
        train_ppo(args)
    else:
        train_dpo(args)


def train_ppo(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    toxicity_model_name = args.toxicity_model_name
    saved_model_path = args.saved_model_path

    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    dataset = load_rlhf_dataset(model_name, dataset_name)

    peft_model = create_peft_model(model_name)
    ppo_model, ref_model = create_ppo_and_ref_model(peft_model)
    ppo_trainer = create_ppo_trainer(model_name, dataset, ppo_model, ref_model, tokenizer)
    rm_pipe = create_reward_model_pipeline(toxicity_model_name, device)

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

    max_ppo_steps = 10

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]

        # Get response from FLAN-T5/PEFT LLM.
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()

            generation_kwargs["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

            summary_tensors.append(summary.squeeze()[-max_new_tokens:])

        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Compute reward outputs.
        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = rm_pipe(query_response_pairs, **reward_kwargs)

        # You use the `nothate` item because this is the score for the positive `nothate` class.
        not_hate_index = 0
        reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]

        # Run PPO step.
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))

    tokenizer.save_pretrained(saved_model_path)
    ppo_model.save_pretrained(os.path.join(saved_model_path, "ppo_model"))
    ref_model.save_pretrained(os.path.join(saved_model_path, "ref_model"))


def train_dpo(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    toxicity_model_name = args.toxicity_model_name
    saved_model_path = args.saved_model_path

    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    dataset = load_dataset(model_name, dataset_name)

    peft_model = create_peft_model(model_name)
    dpo_model, ref_model = create_ppo_and_ref_model(peft_model)
    dpo_trainer = create_dpo_trainer(dataset, dpo_model, ref_model, tokenizer)
    rm_pipe = create_reward_model_pipeline(toxicity_model_name, device)

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

    max_ppo_steps = 10

    for step, batch in tqdm(enumerate(dpo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]

        # Get response from FLAN-T5/PEFT LLM.
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()

            generation_kwargs["max_new_tokens"] = max_new_tokens
            summary = dpo_trainer.generate(prompt_tensor, **generation_kwargs)

            summary_tensors.append(summary.squeeze()[-max_new_tokens:])

        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Compute reward outputs.
        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = rm_pipe(query_response_pairs, **reward_kwargs)

        # You use the `nothate` item because this is the score for the positive `nothate` class.
        not_hate_index = 0
        reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]

        # Run PPO step.
        stats = dpo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        dpo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'dpo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'dpo/policy/advantages_mean: {stats["dpo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))

    tokenizer.save_pretrained(saved_model_path)
    dpo_model.save_pretrained(os.path.join(saved_model_path, "dpo_model"))
    ref_model.save_pretrained(os.path.join(saved_model_path, "ref_model"))


if __name__ == "__main__":
    args = parse_config()
    train(args)
