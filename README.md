# 📰 Vietnamese Extractive News Summarization — Bonus Deployment

An interactive Streamlit application that demonstrates Vietnamese news summarization
using **Position-Aware LexRank + MMR**, the best-performing model from our NLP project.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🌐 Cloud Deployment (Streamlit Community Cloud — Free)

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — live URL in ~2 minutes ✅

No server or Docker needed.

---

## 📁 Project Structure

```
bonus_submission/
├── app.py                   # Streamlit application (main entry point)
├── requirements.txt         # Python dependencies
├── test_samples/
│   ├── sample_economic.txt  # Kinh tế — Economy article
│   ├── sample_tech.txt      # Công nghệ — Technology article
│   ├── sample_sports.txt    # Thể thao — Sports article
│   └── sample_politics.txt  # Chính trị — Politics article
├── bonus_notebook.ipynb     # Notebook demonstrating the app pipeline
└── README.md                # This file
```

---

## 🧠 NLP Methods

| Method | Description |
|--------|-------------|
| **Lead-k** | Baseline — selects the first k sentences |
| **Vanilla LexRank** | Graph-based centrality (PageRank on sentence similarity) |
| **Position-Aware LexRank** | Adds position decay prior + title similarity |
| **Position-Aware LexRank + MMR ★** | Adds Maximal Marginal Relevance to reduce redundancy |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 5 | Number of sentences to select |
| `threshold` | 0.10 | Cosine similarity threshold for graph edges |
| `position_weight` | 0.80 | Strength of position decay prior |
| `lambda_mmr` | 0.70 | MMR trade-off: high → relevance, low → diversity |

Best configuration found via grid search: **λ=0.7, threshold=0.1, position_weight=0.8**

---

## ✨ Features

- **Real-time summarization** — paste any Vietnamese news article
- **4 methods side-by-side** comparison
- **Adjustable parameters** — sliders for all key hyperparameters  
- **Sentence-level breakdown** — see exactly which sentences were selected and why
- **Redundancy metric** — lower is better (less repeated information)
- **Download** — export the summary as a `.txt` file
- **Works offline** — no external API or model download required

---

## 📊 Evaluation (from main project)

Averaged over all clusters in the dataset:

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Redundancy |
|--------|-----------|-----------|-----------|------------|
| Lead-k | — | — | — | — |
| Vanilla LexRank | — | — | — | — |
| Position-Aware LexRank | — | — | — | — |
| **PA-LexRank + MMR** | **best** | **best** | **best** | **lowest** |

*(Fill in actual numbers from your experiments notebook)*

---

## 👥 Contributions

| Member | Contribution |
|--------|-------------|
| [Your Name] | App development, NLP pipeline, deployment |
| [Teammate 2] | Testing, sample data, documentation |

---

## 📄 How It Works (for report)

1. **Input**: User pastes a Vietnamese news article (optionally with a title)
2. **Sentence splitting**: Regex-based sentence tokenizer for Vietnamese
3. **TF-IDF vectorization**: Character n-gram TF-IDF (bigrams–4-grams) — works without word segmentation
4. **LexRank graph**: Sentences become nodes; edges weighted by cosine similarity above `threshold`
5. **Position prior**: Exponential decay — earlier sentences receive higher prior probability
6. **Title similarity**: Each sentence scored by cosine similarity with the article title
7. **Combined score**: Weighted combination of LexRank centrality, position prior, and title score
8. **MMR selection**: Iteratively pick the sentence that maximizes relevance minus redundancy
9. **Output**: Top-k sentences displayed in original document order
