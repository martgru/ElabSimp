# 🔹 Code

The `code` folder is structured into three main directories: **notebooks**, **scripts**, and **utils**, each serving a specific purpose in the workflow.

## 📂 Directories and Contents

### 🔹 `notebooks/`
This directory contains Jupyter notebooks covering the key stages of the project:
- **Fine-tuning Notebooks**: guides for fine-tuning **BART** and **LLaMA** models.
- **Elaboration Generation Notebook**: generates elaborations using the **pretrained LLaMA instruction-tuned model**.
- **Evaluation Notebook**: assesses the quality of generated elaborations using various metrics.
- **Results Inspection Notebook**: presents and compares the final results across different settings and models.
- **Target Estimation Notebook**: identifies the elaboration target within the text.

### 🔹 `scripts/`
This directory contains Python scripts for automating the process:
- `finetune_model.py` – Automates model fine-tuning.
- `generate_elaborations.py` – Generates elaborations using fine-tuned models.
- `calculate_scores.py` – Computes evaluation scores for the generated elaborations.

### 🔹 `utils/`
This directory includes Python files with utility functions for:
- **Dataset processing**
- **Visualization tools**
- **Model fine-tuning**
- **Prompt handling**



  


