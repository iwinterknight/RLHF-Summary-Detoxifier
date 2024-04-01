# RLHF-Summary-Detoxifier

This repo contains code to build a low rank adaptive T5-XL text summarization model with Reinforcement Learning from Human Feedback(RLHF) alignment using Proximal Policy Optimization(PPO) and Direct Preference Optimization(DPO).

# Motivation
Summarizing dialogue is a ubiquitous medium for knowledge extraction for a multitude of downstream business goals such as improving user experience, personalizing content recommendations, collecting user feedback, executing tasks(such as flight booking, restaurant reservations etc.). Unfortunately, user-facing systems can be fraught with language artifacts acquired from pre-training curricula of language models. This work is an attempt to generate dialogue summaries that align with the 3 core values in generated content, helpfulness, honesty and harmlessness. This work also attempts to remove inherent toxic content from the dialogue when generating high quality summaries, without losing factual information. It also aims to draw a comparison between alignment quality between Reinforcement Learning frameworks like Proximal Policy Optimization(PPO) and Direct Preference Optimization(DPO) for conversational data.

# High level architecture for conversation summary generator model training using RLHF(PPO)
<p align="center">
  <img width="789" alt="RLHF Model Architecture" src="https://github.com/iwinterknight/RLHF-Summary-Detoxifier/assets/37212007/efa1b1c9-7123-41a3-839b-b10c86a87c25">
</p>

The instruction fine-tuned version of the Encoder-Decoder style T5 architecture, FLAN T5 XL comprises 3B parameters pre-trained on various language tasks like translation, summarization, question-answering etc.
The encoder will be used to represent conversation data and the decoder will be provided a supervision signal from the ground truth summaries provided in the DialogSum dataset. The fine-tuning will be done in a low rank adaptation setting to meet the compute constraints.
Reward Model : Offensive Speech Classifier, Bias detection model, Sentiment Classifier with cumulative loss fusion through a joint training model.
To prevent Reward Hacking while aligning summaries to human preferences, a Kullback-Leibler divergence loss will be used in addition to the reinforcement learning alignment objective. This is to ensure retention of factual content from the dialogue, while redacting/transforming unsuitable content for summary alignment.
