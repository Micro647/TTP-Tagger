# TTP-Tagger : An LLM-Driven Framework for Advancing ATT&CK Label Classification via Data Augmentation and Full Fine-Tuning

## Project Overview
This repository contains the dataset and result used to run experiments for the paper: "TTP-Tagger : An LLM-Driven Framework for Advancing ATT&CK Label Classification via Data Augmentation and Full Fine-Tuning
"
We propose TTP-Tagger, an LLM-driven framework that advances ATT&CK label classification by leveraging data augmentation and full fine-tuning techniques



## Project Structure

```
TTP-Tagger/
├── Dataset/                    # Dataset directory
│   ├── Aug-Dataset/               
│   │   ├── Test-aug/              # Augmented Test Set
│   │   ├── Train-aug      # Augmented Train Set
│   │   └── Valid-aug        # Augmented Valid Set
│   └── Original-dataset/             
│       ├── test-final.json/   # Original Test Set
│       ├── train-final.json/   # Original Train Set
│       └── validation-final.json             # Original valida Set
└── Methods and Results/              # Implementation & Evaluation
    ├── Adema/       # Adema Method
    ├── Closed-source /      # Closed-source models
    ├── Open-source /       # Open-source model results
    ├── RAG/      # RAG Method

```

## Hyperparameter Description
The relevant hyperparameter configurations are described in detail within the paper.

### Closed-Source LLM Names
Deepseek-v3.1,Qwen-plus,DS-DL-RL-70b,GLM-4.5

### Open-Source LLM Names
qwen2-7b,glm4-9b,gpt-oss-20b






