import torch
import evaluate
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, \
    GenerationConfig, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def create_peft_model(model_name):
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                  torch_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(model,
                                           './peft-dialogue-summary-checkpoint-from-s3/',
                                           lora_config=lora_config,
                                           torch_dtype=torch.bfloat16,
                                           device_map="auto",
                                           is_trainable=True)

    print(f'PEFT model parameters to be updated:\n{print_number_of_trainable_model_parameters(peft_model)}\n')

    return peft_model


def create_ppo_and_ref_model(peft_model):
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,
                                                                   torch_dtype=torch.bfloat16,
                                                                   is_trainable=True)
    print(
        f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
    print(ppo_model.v_head)

    ref_model = create_reference_model(ppo_model)
    print(f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n')

    return ppo_model, ref_model


def prepare_reward_model(toxicity_model_name):
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name, device_map="auto")
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name, device_map="auto")
    print(toxicity_model.config.id2label)
    return toxicity_model, toxicity_tokenizer


def create_reward_model_pipeline(toxicity_model_name, device):
    rm_pipe = pipeline("sentiment-analysis",
                              model=toxicity_model_name,
                              device=device)
    reward_logits_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "none",  # Set to "none" to retrieve raw logits.
        "batch_size": 16
    }

    reward_probabilities_kwargs = {
        "top_k": None,  # Return all scores.
        "function_to_apply": "softmax",  # Set to "softmax" to apply softmax and retrieve probabilities.
        "batch_size": 16
    }
    return rm_pipe


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"