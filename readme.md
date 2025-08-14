# Prompt Generator & Prompt Recovery via Tuning

This repository explores prompt tuning and prompt recovery using LLMs. The main goal is to generate social media-style post titles from prompts, and then optimize a prompt to recover the original prompt given a group of generated outputs.

## Features

- **Prompt Generation**: Automatically creates prompts designed to elicit engaging, social-media-style titles.
- **Post Generation**: Uses prompts to generate multiple post titles using an LLM.
- **Prompt Recovery (Prompt Tuning)**: Optimizes a prompt to infer the original prompt from a set of generated outputs.
- **Evaluation**: Includes a custom metric called "asking an LLM" to score how closely a recovered prompt matches the original.

## Workflow

1. **prompt-generator.py**: Generates prompts for social media posts.
2. **post-generator.py**: Uses these prompts to generate multiple post titles.
3. **tuner.py**: Trains and evaluates a prompt recovery model using the generated data.