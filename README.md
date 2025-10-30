# Transformer-Based Text Generation using GPT-2

This project demonstrates how to generate human-like text using the GPT-2 model, with a focus on understanding the internal workings of transformer-based architectures.

---

## Table of Contents

- [What is a Transformer?](#what-is-a-transformer)
- [What is GPT-2?](#what-is-gpt-2)
- [About TransformerLens](#about-transformerlens)
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Sample Output](#sample-output)
- [How the Code Works](#how-the-code-works)

---

## What is a Transformer?

Transformers are a type of deep learning architecture introduced in the paper [“Attention is All You Need” (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). Unlike RNNs or LSTMs, transformers rely entirely on attention mechanisms to capture relationships between words, enabling:

- **Parallel training** (unlike sequential RNNs)
- **Long-range dependency modeling**
- **Scalability to very large datasets**

They are the foundation of most modern large language models (LLMs), including GPT, BERT, T5, and others.

---

## What is GPT-2?

GPT-2 (Generative Pretrained Transformer 2) is a transformer-based language model developed by OpenAI. It's trained on a massive corpus of internet text and is designed to:

- Predict the next token in a sequence
- Generate coherent and fluent text
- Perform zero-shot learning on many NLP tasks

This project uses **GPT2-small**, a compact version with 117M parameters.

---

## About TransformerLens

[TransformerLens] is an interpretability-first PyTorch wrapper for transformer models. It allows you to:

- Access model internals like attention heads, MLP layers, and activations
- Hook into different parts of the forward pass
- Analyze or modify the computation graph

In this project, we use TransformerLens for both model access and analysis.

---

## Project Overview

This notebook does the following:

- Loads the GPT2-small model using TransformerLens
- Starts with a user-defined prompt (e.g., `"earth revolve around"`)
- Generates 100 additional tokens using **probabilistic sampling** (to avoid repetitive output)
- Outputs a complete, coherent continuation of the prompt

---

## Setup and Installation

Make sure you have Python ≥ 3.8 installed. Then, install the dependencies:


## How to Run the Notebook
- Open transformers.ipynb in Jupyter Notebook or Google Colab.
- Run all cells sequentially.
- View the generated text after the loop finishes.
- You can modify the prompt or number of tokens to explore different generations.

## Sample Output
Prompt: earth revolve around
Generated Text:
earth revolve around the sun

## How the Code Works
- The model is loaded using:
                          HookedTransformer.from_pretrained("gpt2-small")
- A loop runs 100 times:
 -- The input string is tokenized
 -- The model generates logits (raw predictions)
 -- Softmax is applied to get probabilities
 -- A token is sampled (not argmax) to avoid repetition
 -- The sampled token is decoded and appended
   
- Important fixes include:
 -- Detaching logits from the graph
 -- Replacing argmax() with multinomial()
 -- Re-normalizing probabilities to avoid NaNs
