# Persian Named Entity Recognition (NER) Using BERT-based Models

## Overview

In this project, I have fine-tuned several well-known BERT-based models for the task of Named Entity Recognition (NER) in Persian using the [Persian-NER dataset](https://github.com/Text-Mining/Persian-NER). This project encompasses comprehensive preprocessing, extensive error analysis, and an exploration of various fine-tuning techniques aimed at improving the F1 score, especially for underrepresented labels. Importantly, while optimizing these models, the underlying architectures have been preserved without any modifications.

## Introduction to Named Entity Recognition (NER)

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves the identification and classification of entities in a text into predefined categories such as persons, organizations, locations, dates, and more. It is a form of token classification where each token (word) in a sentence is assigned a label. The goal of NER is to accurately extract entities and classify them correctly, which is crucial for a variety of applications including information retrieval, question answering, and content analysis.

## Dataset

The dataset used in this project is a labeled Persian dataset specifically designed for NER tasks. It contains five distinct labels:

- **PER**: Person names
- **EVE**: Events
- **ORG**: Organizations
- **DAT**: Dates
- **LOC**: Locations

This dataset poses unique challenges compared to other benchmark datasets like [Arman and Peyma](https://hooshvare.github.io/docs/datasets/ner), due to its complexity and the nuanced nature of the labels. These challenges make it an excellent testbed for evaluating and improving NER models in Persian.

## Models Evaluated and Fine-Tuned

The following BERT-based models were evaluated and fine-tuned during the course of this project:

- [**faBERT**](https://huggingface.co/sbunlp/fabert)
- [**tookaBERT-base**](https://huggingface.co/PartAI/TookaBERT-Base)
- [**tookaBERT-large**](https://huggingface.co/PartAI/TookaBERT-Large)
- **XLM-roberta**

Each model was fine-tuned with the aim of enhancing the performance on the Persian-NER dataset, with particular attention to improving F1 scores across all labels, including those that are underrepresented.

# Preprocessing

The raw dataset containes text files. Each text file is converted from ConLL to token_label format and the appropriate columns are formed and the text files are converted to csv files.

## Fine-Tuning Methodologies

Below is an overview of the contents of each folder, detailing the fine-tuning processes and methodologies employed:

- [basic-finetune](./basic-finetune): In this folder, you will find each model finetuned and tested on a consistent subset of the original dataset.  
- [models_stratified](./models_stratified): In this folder, you will find models finetuned on stratified dataset where low representetive labeles are oversampled.
- [incremental-24_48](./incremental-24_48): In this folder, you will find models fine-tuned on the previously mentioned datasets using an incremental approach.
- [intermediate-finetune](./intermediate-finetune): In this folder, intermediate fine-tuning (MLM) was conducted on faBERT using the dataset to enhance its understanding of the data.
- [arman-peyma](./arman-peyma): In this folder, faBERT model was finetuned on ARMAN and PEYMA datasets and then used for our dataset.
- [large-80000](./large-80000): In this folder, better performing models where further finetuned on a larger subset of the original dataset.

## Evaluation

[Here](evaluation.ipynb) you can find the evaluation results, including precision, recall, and F1 scores for each model and label. Additionally, the notebook contains plots illustrating the challenge of overfitting that affected some models.

## Finetuned Models

All of the fine-tuned models have been pushed to the Hugging Face Hub and are accessible [here](https://huggingface.co/pouria82).
