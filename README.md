# IS6002.CH191 Final Project: Text-to-SQL with Fine-tuned Qwen3-0.6B

This repository contains notebooks, datasets, and experimental results for the final project of the Advanced Database Systems course (IS6002.CH191). The project focuses on fine-tuning the small language model Qwen3-0.6B to solve Text-to-SQL tasks.

## Project Overview

Text-to-SQL is the task of converting natural language questions into executable SQL queries. This project demonstrates the effectiveness of supervised fine-tuning (SFT) on a compact 0.6B parameter model, evaluating its performance against baseline models including GPT-4o on both single-table and multi-table database scenarios.

## Repository Structure

### 📁 `dataset/`

Contains training and evaluation datasets used throughout the fine-tuning and evaluation process. See the [dataset documentation](dataset/README.md) for detailed information about:

-   Single-table dataset (5,000 training samples, 200 dev samples)
-   BIRD multi-table dataset with complex enterprise schemas
-   Both JSON and Parquet formats available

### 📁 `notebooks/`

Contains Jupyter notebooks for the complete experimental pipeline:

-   **`text2sql_finetune.ipynb`**: Fine-tunes the Qwen3-0.6B model on Text-to-SQL datasets and uploads the trained model to Hugging Face
-   **`text2sql_eval.ipynb`**: Generates predictions on test datasets using the fine-tuned model
-   **`text2sql_extract_accuracy.ipynb`**: Calculates accuracy metrics from LLM-as-a-judge evaluation results

### 📁 `results/`

Contains experimental results from model evaluations. File naming conventions:

-   **Original-**: Pre-trained model without fine-tuning
-   **SFT-**: Supervised fine-tuned model
-   **GPT-4o-**: Baseline results from GPT-4o for comparison

## Experimental Results

The following table summarizes the accuracy of different models on Single-Table and BIRD datasets:

### Single-Table Dataset (200 samples)

| Model               | Correct | Total   | Accuracy   |
| ------------------- | ------- | ------- | ---------- |
| GPT-4o              | 90      | 200     | 45.00%     |
| Original-Qwen3-0.6B | 28      | 200     | 14.00%     |
| **SFT-Qwen3-0.6B**  | **98**  | **200** | **49.00%** |

### BIRD Dataset (500 samples)

| Model               | Correct | Total | Accuracy |
| ------------------- | ------- | ----- | -------- |
| GPT-4o              | 311     | 500   | 62.20%   |
| Original-Qwen3-0.6B | 33      | 500   | 6.60%    |
| SFT-Qwen3-0.6B      | 10      | 500   | 2.00%    |

### Key Findings

-   **Single-Table Performance**: The fine-tuned Qwen3-0.6B model achieves **49.00% accuracy**, surpassing GPT-4o (45.00%) and showing a **+35 percentage point improvement** over the original model (14.00%).
-   **Multi-Table Performance**: On the more complex BIRD dataset, GPT-4o maintains the lead at 62.20%. The fine-tuned model shows degraded performance (2.00%) compared to the original (6.60%), indicating that additional training strategies are needed for complex multi-table scenarios.

## Usage

1. **Training**: Open `text2sql_finetune.ipynb` to fine-tune the model
2. **Evaluation**: Use `text2sql_eval.ipynb` to generate predictions
3. **Analysis**: Run `text2sql_extract_accuracy.ipynb` to compute accuracy metrics

## Course Information

-   **Course**: Advanced Database Systems
-   **Course Code**: IS6002.CH191
-   **Project Type**: Final Project
