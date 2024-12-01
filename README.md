# Comparatively Assessing Large Language Models for Query Expansion in Information Retrieval via Zero-Shot and Chain-of-Thought Prompting

This repository contains the data and a sample code implementation for the paper titled *"Comparatively Assessing Large Language Models for Query Expansion in Information Retrieval via Zero-Shot and Chain-of-Thought Prompting"*.

Paper at https://ceur-ws.org/Vol-3802/paper22.pdf

# Sample Code
  
The example code `samplecode_llama3.py` is designed to generate data for the `scifact` test set using three distinct prompting strategies, as outlined in the paper:  
```
1. Write a passage that answers the given query: {query}
2. Write a list of keywords for the following query: {query}
3. Answer the following query:\n{query}\nGive the rationale before answering
```
This script uses the `Meta-Llama-3-8B-Instruct` model.    
Please ensure you have a personal `access_token` from Hugging Face to execute the code.  
  
Additionally, the code includes an evaluation part to assess the performance in terms of MAP and RECALL@1000.  

# Citation

Please cite the following paper if you use the data or code in this repo.

```
@inproceedings{rizzo-etal-2024-iir,
  title={Comparatively Assessing Large Language Models for Query Expansion in Information Retrieval via Zero-Shot and Chain-of-Thought Prompting},
  author={Rizzo, Daniele and Raganato, Alessandro and Viviani, Marco},
  booktitle={Proceedings of the 14th Italian Information Retrieval Workshop (IIR 2024)},
  address = {Udine, Italy},
  year={2024}
}
```
