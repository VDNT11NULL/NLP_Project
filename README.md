# NLP Project: Simultaneous Machine Translation and Lightweight Language Models

## Project Overview

This repository contains the implementation of two key sub-projects in Natural Language Processing (NLP):

1. **Improving Results with Lightweight Models on Hivemind**  
   Exploring lightweight, parameter-efficient models and full fine-tuned models to achieve similar or better results compared to computationally expensive large models for domain-specific question answering.

2. **Dynamic Policy for Simultaneous Machine Translation (SMT) in Hindi to English**  
   Implementation and comparison of Wait-k Policy and a novel POS-SMT Policy to improve the efficiency and accuracy of simultaneous machine translation over Indic languages.

---

## Repository Structure

```
├── full_fine_tune_train.py     # Script for training full fine-tuned models
├── PEFT_train.py               # Script for training parameter-efficient fine-tuned models (PEFT)
├── Inference_Evaluation.ipynb  # Notebook for inference and evaluation (switchable models)
├── NLP_pos_indictrans.ipynb    # Jupyter Notebook for POS tagging and translation using IndicTrans
├── NLP_Project_Report.pdf      # Detailed project report
├── POS-SiLLM.py                # Script for implementing the POS-SMT Policy
├── POS-SiLLM.sh                # Shell script for running POS-SMT experiments
├── Wait-k-SiLLM.py             # Script for Wait-k Policy implementation
└── Wait-k-SiLLM.sh             # Shell script for running Wait-k Policy experiments
```

---

## Key Contributions

### 1. Lightweight Models for Hivemind
- **Objective**: Fine-tune both full-scale and parameter-efficient models for domain-specific tasks across STEM, Humanities, Social Sciences, and Others.
- **Models Used**:
  - Meta-Llama 1B and 3B with:
    - **Full Fine-Tuning** (`full_fine_tune_train.py`)
    - **Parameter-Efficient Fine-Tuning (PEFT)** (`PEFT_train.py`)
- **Results**:  
  - Lightweight and full fine-tuned models achieve competitive accuracy with Hivemind's large models.  
  - Significant reductions in computational cost using PEFT models.

### 2. Dynamic Policy for SMT in Hindi Language
- **Objective**: Develop an efficient, dynamic approach for simultaneous machine translation (Hindi to English).  
- **Implemented Policies**:
  - **Wait-k Policy**: A static seeding approach based on the `k` hyperparameter.
  - **POS-SMT Policy**: A dynamic seeding approach using Part-of-Speech (POS) tags (e.g., Noun, Verb).  
- **Evaluation Metrics**:
  - BLEU, SacreBLEU, and BertScore.  
- **Results**:  
  - POS-SMT (Verb) performs on par with Wait-k (k=7) and outperforms for lower k-values.
  - Dynamic POS-SMT Policy offers practical advantages over static Wait-k.

---

## Setup and Usage

### Prerequisites
- Python 3.8 or higher
- Required libraries: `stanza`, `transformers`, `torch`, `numpy`, `scipy`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments
- **Full Fine-Tuning**:
  ```bash
  python full_fine_tune_train.py
  ```
- **PEFT Training**:
  ```bash
  python PEFT_train.py
  ```
- **Inference and Evaluation**:
  Use the `Inference_Evaluation.ipynb` notebook. Update the model paths and configurations as needed (supports Llama-3.1 1B and 3B).

- **Wait-k Policy**:
  ```bash
  bash Wait-k-SiLLM.sh
  ```
- **POS-SMT Policy**:
  ```bash
  bash POS-SiLLM.sh
  ```

---

## Results Summary

### Lightweight Models for Hivemind
| Model                  | STEM  | Humanities | Social Sciences | Others  |
|------------------------|-------|------------|-----------------|---------|
| Llama-3B (PEFT)        | 0.465 | 0.496      | 0.638           | 0.623   |
| Hivemind (STEM)        | 0.7   | 0.58       | 0.64            | 0.6     |

### Dynamic Policy for SMT
| Approach               | BLEU   | SacreBLEU | BertScore |
|------------------------|--------|-----------|-----------|
| Wait-4                | 0.066  | 14.86     | 0.94253   |
| POS-SMT (Verb + Noun) | 0.042  | 11.93     | 0.93804   |
| POS-SMT (Verb)        | 0.080  | 16.06     | 0.94614   |

---

## Future Directions
- Extend dynamic POS-SMT Policy to other Indic languages.
- Optimize the inference pipeline for reduced latency.
- Investigate hybrid approaches combining Wait-k and POS-SMT.

---

## Authors
- **Bhavik Shangari**  
- **Vedant Marodkar**  
- **Rajeev Goel**

For more details, refer to the [project report](NLP_Project_Report_.pdf).
