# 🇳🇵 Nepali News Summarizer using QLoRA

A fine-tuned Large Language Model for summarizing Nepali news articles using **QLoRA** and **Qwen2.5-1.5B-Instruct**.

## 📌 Project Overview

This project fine-tunes the Qwen2.5-1.5B-Instruct model using QLoRA to generate concise summaries of Nepali news articles.

The project demonstrates:

- Parameter-Efficient Fine-Tuning (PEFT)
- QLoRA (4-bit Quantization)
- Hugging Face Transformers
- LoRA Adapters
- Gradio Interface
- Hugging Face Hub Deployment

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- QLoRA
- Gradio
- Accelerate
- Datasets

---

## 📂 Project Structure

```
Nepali-News-Summarize/
│
├── src/
├── notebooks/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🤗 Hugging Face Model

LoRA Adapter:

https://huggingface.co/tstuti7/nepali-qlora-model

---

## 🚀 Run Locally

Clone the repository

```bash
git clonehttps://github.com/istuti55/Nepali-News-Summarize.git  
cd Nepali-News-Summarize
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

---

## 🧠 Base Model

Qwen/Qwen2.5-1.5B-Instruct

---

## 📈 Fine-tuning

- Method: QLoRA
- Adapter Rank: 16
- PEFT
- Supervised Fine-Tuning (SFT)

---

## 👤 Author

tstuti