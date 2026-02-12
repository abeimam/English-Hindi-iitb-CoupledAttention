title: English to Hindi Translation



# English to Hindi Translation – Coupled Attention Transformer

**Version:** 1.0  
**Release Date:** February 12, 2026  
**Authors:** Yusuf Imam, Shams Tabrez Khan   
**Contact:** yousufabeimam@gmail.com  

---

## 📌 Overview

This repository hosts a **custom Neural Machine Translation (NMT) model** for English→Hindi.  
We propose **Coupled Attention** – a novel attention mechanism that **completely removes the value projection** and introduces **dual query/key pairs** with **pre‑softmax SiLU gating**.  

Our model achieves **27.4 BLEU** on the **IIT Bombay English–Hindi parallel corpus**, surpassing the previous state‑of‑the‑art by **+3.95 BLEU**, while using **fewer parameters** and running **faster** than a standard Transformer.

🔬 **This is the first work to systematically remove the value projection from multi‑head attention while retaining the dot‑product formulation.**

---



## 🧠 Model Architecture (Summary)

| Component               | Description                                      |
|------------------------|--------------------------------------------------|
| **Attention**          | Coupled Attention (no value projection, dual Q/K, pre‑softmax SiLU) |
| **Positional Encoding**| Rotary embeddings (RoPE) – self‑attention only   |
| **Feed‑Forward**       | Gated Linear Units (GLU)                        |
| **Language Control**   | Additive language embeddings (per token)        |
| **Weight Tying**       | Decoder output ↔ token embeddings              |
| **Encoder Layers**     | 8                                               |
| **Decoder Layers**     | 6                                               |
| **Hidden Size**        | 1024                                            |
| **FFN Size**           | 4096                                            |
| **Attention Heads**    | 12                                              |
| **Total Parameters**   | **256M**                                        |

*Standard Transformer with identical depth/dimensions has ~280M parameters – our model is **~9% smaller**.*

---

## 🔤 Tokenization

- **Algorithm:** SentencePiece Unigram  
- **Vocabulary size:** 42,000 (joint English–Hindi)  
- **Normalization:** NFKC  
- **Special tokens:** `<pad>`, `<unk>`, `<s>`, `</s>`, `<en>`, `<hi>`, `<sep>`  
- **Regularization:** Unigram sampling (α = 0.1) during training  

---

## 🏋️ Training Setup

### Dataset
- **IIT Bombay English–Hindi parallel corpus v2.0**  
  (Kunchukuttan et al., 2018) – 1.49M parallel sentences  
- **Split:** 95% train, 2.5% validation, 2.5% test  
- **Max sequence length:** 256 subword tokens  

### Hyperparameters

| Parameter               | Value                     |
|-------------------------|---------------------------|
| Optimizer               | AdamW                     |
| β₁, β₂                 | 0.9, 0.98                |
| ε                       | 1e-9                     |
| Weight decay            | 0.01                     |
| Peak learning rate      | 1e-4                     |
| LR schedule             | Cosine decay             |
| Warmup steps            | 4,000                    |
| Total steps             | 200,000                  |
| Effective batch size    | 64                       |
| Gradient clipping       | 1.0                      |
| Label smoothing         | 0.1                      |
| Mixed precision         | FP16                     |
| Gradient checkpointing  | Yes                      |
| Evaluation interval     | Every 2,000 steps        |
| Checkpoint selection    | Best validation BLEU     |

### Compute
- **Hardware:** 2× NVIDIA Tesla T4 (Kaggle) / 1× V100 (32 GB)  
- **Training time:** ~8 days (T4) / ~5 days (V100)  

---

## 📊 Evaluation Results

### Main Results (IIT Bombay En→Hi test set)

| Model | Params | BLEU | chrF++ |
|-------|--------|------|--------|
| SMT (Moses) – Kunchukuttan et al. (2018) | – | 11.75 | – |
| RNN Search – Kunchukuttan et al. (2018)  | – | 12.23 | – |
| 2‑layer BiLSTM+attn – Baruah et al. (2021) | – | 23.45 | – |
| Transformer (base) – our reimplementation | 280M | 24.1 | 50.3 |
| Transformer + RoPE + GLU – our reimpl. | 280M | 25.8 | 52.1 |
| **Coupled Attention (ours)** | **256M** | **27.4** | **54.2** |

✅ **New state‑of‑the‑art** – outperforms previous best by **+3.95 BLEU**.

---

### Ablation Studies

| Variant | BLEU | Δ    |
|---------|------|------|
| **Full Coupled Attention** | **27.4** | –    |
| w/o SiLU (raw dot)        | 25.6 | –1.8 |
| Single Q/K pair (only Qw,Kw) | 24.9 | –2.5 |
| + value projection (standard MHA) | 26.1 | –1.3 |
| w/o language embeddings   | 26.7 | –0.7 |
| w/ cross‑attention RoPE   | 27.1 | –0.3 (ns) |

*All differences except cross‑attention RoPE are statistically significant (p < 0.05, bootstrap resampling).*

---

### Inference Speed

| Model | Tokens/sec (V100) | Speedup |
|-------|-------------------|---------|
| Transformer + RoPE + GLU | 412 | 1.0× |
| **Coupled Attention (ours)** | **461** | **1.12×** |

**12% faster** – due to elimination of value projection matrices.

---

## 🧪 Example Translations

| English | Reference Hindi | Coupled Attention |
|--------|-----------------|-------------------|
| I love you. | मैं तुमसे प्यार करता हूँ। | मैं आपको प्यार करता हूं। |
| I love you too. | मैं भी तुमसे प्यार करता हूँ। | मैं आपको भी प्यार करता हूँ। |
| I am Yusuf Imam. | मैं यूसुफ इमाम हूँ। | मैं यूसफ इमाम हूं। |
| Where is the nearest hospital? | निकटतम अस्पताल कहाँ है? | सबसे नजदीकी अस्पताल कहां है? |
| Thank you for your help. | आपकी मदद के लिए धन्यवाद। | आपकी सहायता के लिए धन्यवाद। |

✅ **Observed strengths:**  
- Correct honorifics (आपको vs तुमसे).  
- Accurate particle placement (भी).  
- Proper named‑entity transliteration (यूसफ इमाम).  
- Fluent, grammatical Hindi output.

---

