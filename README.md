# RLHF-Summary-Detoxifier

This repo contains code to build a low rank adaptive T5-XL text summarization model with Reinforcement Learning from Human Feedback(RLHF) alignment using Proximal Policy Optimization(PPO) and Direct Preference Optimization(DPO).

# Motivation
Summarizing dialogue is a ubiquitous medium for knowledge extraction for a multitude of downstream business goals such as improving user experience, personalizing content recommendations, collecting user feedback, executing tasks(such as flight booking, restaurant reservations etc.). Unfortunately, user-facing systems can be fraught with language artifacts acquired from pre-training curricula of language models. This work is an attempt to generate dialogue summaries that align with the 3 core values in generated content, helpfulness, honesty and harmlessness. This work also attempts to remove inherent toxic content from the dialogue when generating high quality summaries, without losing factual information.
