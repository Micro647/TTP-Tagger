# TTP-Tagger : An LLM-Driven Framework for Advancing ATT&CK Label Classification via Data Augmentation and Full Fine-Tuning

## Overview
ATT&CK-CHAR.json
Presents a character-type dataset designed based on the MITRE ATT&CK framework, along with the inference results from full-parameter fine-tuning on the Llama3-8B-Instruct model.

ATT&CK-CHAR.json
Presents a numeric ID-type dataset designed based on the MITRE ATT&CK framework, along with the inference results from full-parameter fine-tuning on the Llama3-8B-Instruct model.

## Experimental Details
The full-parameter fine-tuning experiments were conducted on two NVIDIA A100 GPUs.
The hyperparameters used were as follows:
 number of training epochs = 3, learning rate = 1e-5, per-device batch size = 1,
and gradient accumulation steps = 2.

## Experiment Framework
Our method is re-implemented and adapted based on LLaMA-Factory
Url:https://github.com/hiyouga/LLaMAFactory

##Infer
The inference code is shown in Infer.py



