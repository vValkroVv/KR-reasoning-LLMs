# Improving the Quality of SQL Query Generation via Adaptive Ensembling of Multi-Model Reasoning Paths

This repository implements **adaptive ensembling**, a dynamic framework for optimizing robustness and accuracy in Text-to-SQL systems through multi-path reasoning ensembles. The code supports evaluation on BIRD and SPIDER datasets using Qwen3 and OmniSQL models (1.7B–32B), with training capabilities for distilling ensemble knowledge into smaller models.

---

## 🔍 Key Features
- **Adaptive Ensembling**  
  - Dynamic strategy selection between reasoning+answer aggregation and answer-only aggregation
  - Adaptive path scaling (K=2–5) via complexity gap metric
- **Model Support**  
  - Qwen3 series (1.7B–32B) and OmniSQL models
  - Evaluation on **BIRD** and **SPIDER** datasets
- **Training Pipeline**  
  - Fine-tuning scripts for distilling ensemble reasoning paths into smaller models
  - Reduces runtime latency while maintaining accuracy
- **Comprehensive Analysis**  
  - Over 30+ Jupyter notebooks for scoring, analysis, and visualization

---

## 📁 Repository Structure
```bash
├── clear_scoring/          # Evaluation notebooks for all models
│   ├── OmniSql-*.ipynb     # OmniSQL model evaluations
│   ├── Qwen3-*.ipynb       # Qwen3 model evaluations (1.7B–32B)
│   └── analytics.ipynb     # Final result aggregation
├── src/                    # Core codebase
│   ├── prompt_*.py         # Prompt engineering modules
│   ├── score_sql.py        # SQL evaluation metrics
│   └── pipeline.py         # End-to-end inference pipeline
├── training/               # Training/distillation scripts
│   ├── train.py            # Training framework
│   └── train_bird_1.7b.sh  # Example training script
├── requirements_deepspeed.txt  # For training dependencies
└── requirements_validation.txt # For evaluation dependencies
```

---

## 🛠️ Installation

### For Evaluation (BIRD/SPIDER)
```bash
pip install -r requirements_validation.txt
```

### For Training
```bash
pip install -r requirements_deepspeed.txt
```

---

## 🧪 Usage

### 1. Model Evaluation
All evaluation notebooks are in `clear_scoring/`. For example:
```bash
jupyter notebook clear_scoring/Qwen3-14b-cot-DEV/Results_cot_Qwen3-14b-m_schema.ipynb
```
Key notebooks:
- `Results_cot_*`: Final accuracy metrics
- `*-cot(2-5)*`: Multi-path reasoning experiments
- `analytics.ipynb`: Cross-model comparison dashboard

### 2. Training New Models
Run training scripts from the `training/` directory:
```bash
cd training/
sh train_bird_1.7b.sh
```
Modify `accelerate_config_1.7b.yaml` for different hardware setups.

---

## 📚 Paper Citation
```bibtex
@article{texttosql_llm_reasoning_aggregation,
  title={Improving the Quality of SQL Query Generation based on a Text Query by aggregating Information from the reasoning of Various LLM Models},
  author={Valerii Kropotin},
  year={2025}
}
```

---

## 📄 License
[MIT License]

## 👥 Contributors
Valerii Kropotin

---

### Highlights from Paper Results
- **Qwen3-1.7B**: +5.9% execution accuracy on complex BIRD queries via multi-path ensembling
- **Qwen3-14B**: Matches ensemble performance with 2–3× lower cost using answer-only strategy
- **Distilled Models**: Achieve 46.5% EX accuracy (vs. 47.1% runtime ensemble) with single-path inference

This implementation enables reproducibility of all experiments in the paper, including complexity gap analysis, path scaling studies, and knowledge distillation benchmarks.
