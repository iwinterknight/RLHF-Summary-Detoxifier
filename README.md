# RLHF-Summary-Detoxifier

This repo contains code to build a low rank adaptive T5-XL text summarization model with Reinforcement Learning from Human Feedback(RLHF) alignment using Proximal Policy Optimization(PPO).

# Motivation
Summarizing dialogue is a ubiquitous medium for knowledge extraction for a multitude of downstream business goals such as improving user experience, personalizing content recommendations, collecting user feedback, executing tasks(such as flight booking, restaurant reservations etc.). Unfortunately, user-facing systems can be fraught with language artifacts acquired from pre-training curricula of language models. This work is an attempt to generate dialogue summaries that align with the 3 core values in generated content, helpfulness, honesty and harmlessness. This work also attempts to remove inherent toxic content from the dialogue when generating high quality summaries, without losing factual information from conversational data. 

# High level architecture for conversation summary generator model training using RLHF(PPO)
<p align="center">
  <img width="789" alt="RLHF Model Architecture" src="https://github.com/iwinterknight/RLHF-Summary-Detoxifier/assets/37212007/efa1b1c9-7123-41a3-839b-b10c86a87c25">
</p>

The instruction fine-tuned version of the Encoder-Decoder style T5 architecture, FLAN T5 XL comprises 3B parameters pre-trained on various language tasks like translation, summarization, question-answering etc.
The encoder will be used to represent conversation data and the decoder will be provided a supervision signal from the ground truth summaries provided in the DialogSum dataset. The fine-tuning will be done in a low rank adaptation setting to meet the compute constraints.
Reward Model : Offensive Speech Classifier, Bias detection model, Sentiment Classifier with cumulative loss fusion through a joint training model.
To prevent Reward Hacking while aligning summaries to human preferences, a Kullback-Leibler divergence loss will be used in addition to the reinforcement learning alignment objective. This is to ensure retention of factual content from the dialogue, while redacting/transforming unsuitable content for summary alignment.

# Datasets and Models
# DialogSum
DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 dialogues with corresponding manually labeled summaries and topics, focusing on dialogues under rich real-life scenarios, including more diverse task-oriented dialogues.

Salients : 
1. Under rich real-life scenarios, including more diverse task-oriented scenarios.
2. Have clear communication patterns and intents, which is valuable to serve as summarization sources.
3. Have a reasonable length, which comforts the purpose of automatic summarization.

# Model
# Flan T5 XL
Flan T5 is the instruction fine-tuned version of the encoder-decoder style T5 model (Publication link : https://arxiv.org/pdf/2210.11416.pdf). The Flan T5 XL comprising of 3B parameters is first fine-tuned on the DialogSum dataset with low rank adaptive(LoRA) parameters, using the open-sourced `peft` transformer library. This reduces the trainable parameters to ~18M, viz. approximately 0.66% of the total trainable parameters. 
<p align="center">
  <img width="413" alt="LoRA" src="https://github.com/iwinterknight/RLHF-Summary-Detoxifier/assets/37212007/4a8ecc37-3035-4216-aba6-06fbb9360312">
</p>

PEFT fine-tuning causes a slight but reasonable drop in performance compared to the original instruction-tuned model, as seen below in the ROUGE metric values.
<p align="center">
  <img width="732" alt="peft_results" src="https://github.com/iwinterknight/RLHF-Summary-Detoxifier/assets/37212007/b5edb5dc-3b37-43a9-ac38-dc63a89be099">
</p>

# Training
The instruction tuned peft model is trained on the RLHF objective for alignment to produce summaries without offensive/bias tone and content. The RL objective uses Proximal Policy Optimization(PPO) to jointly train against weighted rewards scored from a sentiment scorer model and a bias detection model. In order to prevent reward hacking a Kullback–Leibler(KL) divergence penalty is added to the training objective.

# Installation
Clone the repository and install the required packages:
```
git clone https://github.com/iwinterknight/RLHF-Summary-Detoxifier.git
cd RLHF-Summary-Detoxifier
pip install -r requirements.txt
```
