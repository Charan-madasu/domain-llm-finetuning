# Domain-Specific LLM Fine-Tuning for Code Generation
> Mistral-7B + QLoRA on CodeAlpaca

Fine-tuned Mistral-7B (7 billion parameters) to generate Python code,
using QLoRA to make it run on a single free Colab GPU.

---

## What This Project Does

Took a general-purpose 7B language model and specialized it for code
generation — running entirely on a free T4 GPU through 4-bit quantization.

- **4-bit quantization** brought Mistral-7B from 14GB down to 4GB
- **QLoRA** trained only 0.19% of parameters (13.6M out of 7.25B)
- **58MB adapter** captures all the fine-tuning
- **Live Gradio demo** lets anyone generate code interactively

---

## Results — Base vs Fine-Tuned

Same coding prompts given to both the original Mistral and the fine-tuned version:

| Metric | Base Mistral | Fine-Tuned |
|---|---|---|
| Produces a function | 100% | 100% |
| Includes docstrings | 0% | 33% |
| Stays in Python | Sometimes drifts to JS | Consistent Python |

**Key finding:** On a Fibonacci prompt, base Mistral drifted into JavaScript,
while the fine-tuned model stayed in Python — showing the fine-tuning
successfully shaped the model toward the training data's style.

---

## Training Details

| Setting | Value |
|---|---|
| Base Model | Mistral-7B-v0.1 |
| Quantization | 4-bit NF4 (QLoRA) |
| Trainable params | 13.6M (0.19%) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Dataset | CodeAlpaca (5,000 filtered examples) |
| Epochs | 1 |
| Val Loss | 0.578 → 0.542 |

---

## Project Structure

---

## How To Run

### 1. Build the dataset (local)
```bash
python data_pipeline/collect.py
python data_pipeline/filter.py
```

### 2. Fine-tune (Google Colab with T4 GPU)
Open `notebooks/mistral_qlora_codegen.ipynb` in Colab, upload the
`clean_dataset.json`, and run all cells.

---

## Known Limitations

- Trained for only 1 epoch — the model occasionally repeats itself.
  Adding more epochs and a repetition penalty during decoding improves this.
- The 58MB adapter is hosted separately (Google Drive) since model
  weights aren't suited for Git.

---

## Skills

`Python` `PyTorch` `HuggingFace` `PEFT` `LoRA` `QLoRA` `4-bit Quantization`
`bitsandbytes` `Mistral-7B` `LLM Fine-Tuning` `Gradio` `Model Evaluation`

---

## Author
**Charan Krishna Madasu**
[GitHub](https://github.com/Charan-madasu)