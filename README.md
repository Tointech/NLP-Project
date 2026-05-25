# 📰 Vietnamese Extractive News Summarization

An interactive Streamlit application for Vietnamese news summarization supporting
**6 extractive methods**, from a simple Lead-k baseline to the best-performing
**Position-Aware LexRank + MMR** — developed as part of COMP5040 NLP, VinUniversity Spring 2026.

🌐 **Live demo:** https://nlp-final-project.streamlit.app

📁 **Full project repo:** https://github.com/cnvcuong/VinUni_Spring26_NLP_COMP5040_FinalProject_Group1

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
├── app.py                   # Streamlit application (main entry point)
├── requirements.txt         # Python dependencies
├── test_samples/
│   ├── sample_economic.txt  # Kinh tế — Economy article
│   ├── sample_tech.txt      # Công nghệ — Technology article
│   ├── sample_sports.txt    # Thể thao — Sports article
│   └── sample_politics.txt  # Chính trị — Politics article
├── notebook.ipynb           # Notebook demonstrating the app pipeline
└── README.md                # This file
```

---

## 🧠 NLP Methods

The app supports all 6 methods studied in the research project:

| Method | Representation | Centrality | Selection |
|--------|---------------|------------|-----------|
| **Lead-k** | — | — | First k sentences |
| **Vanilla LexRank** | TF-IDF | LexRank (uniform prior) | Top-k |
| **Position-Aware LexRank** | TF-IDF | LexRank (pos + title prior) | Top-k |
| **BERT Centroid** | BERT | Centroid similarity | Top-k |
| **PACSUM** | BERT | Directed graph (β) | Top-k |
| **Position-Aware LexRank + MMR ★** | TF-IDF | LexRank (pos + title prior) | MMR |

★ Proposed method — best performance across all metrics.

BERT methods use `paraphrase-multilingual-MiniLM-L12-v2` for sentence encoding.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 5 | Number of sentences to select |
| `threshold` | 0.10 | Cosine similarity threshold for graph edges (τ) |
| `position_weight` | 0.80 | Strength of position decay prior (α) |
| `lambda_mmr` | 0.70 | MMR trade-off: high → relevance, low → diversity (λ) |
| `beta_pacsum` | 0.00 | PACSUM positional bias: 0 = max bias, 1 = symmetric (β) |

**Best configuration from grid search (80 configs):** λ=0.6, threshold=0.05, position_weight=0.7
→ ROUGE-L F1 = **0.3140**

---

## ✨ Features

- **6 methods** — from simple Lead-k to BERT-based PACSUM
- **Real-time summarization** — paste any Vietnamese news article
- **Adjustable parameters** — sliders for all key hyperparameters
- **Recommended config** displayed in sidebar for the proposed method
- **Sentence-level breakdown** — relevance score shown for each selected sentence
- **Redundancy metric** — lower is better (less repeated information)
- **Source highlighting** — each sentence linked back to its source article
- **Download** — export the summary as a `.txt` file
- **Works offline** — no external API required (BERT model auto-downloaded on first run)

---

## 📊 Evaluation Results

Averaged over 300 Vietnamese news clusters (ViMs dataset, 1,945 articles):

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | Redundancy | SrcCover |
|--------|-----------|-----------|-----------|-------------|------------|----------|
| Lead-k | 0.4513 | 0.2731 | 0.2897 | 0.7363 | 0.0945 | 0.1756 |
| Vanilla LexRank | 0.4656 | 0.2840 | 0.2862 | 0.7366 | 0.3307 | 0.6022 |
| Position-Aware LexRank | 0.4690 | 0.2977 | 0.2922 | 0.7452 | 0.3369 | 0.6416 |
| BERT Centroid | 0.4474 | 0.2609 | 0.2656 | 0.7314 | 0.6933 | 0.5972 |
| PACSUM | 0.4524 | 0.2637 | 0.2705 | 0.7299 | 0.5176 | 0.2578 |
| **PA-LexRank + MMR ★** | **0.5135** | **0.3354** | **0.3044** | **0.7569** | 0.2147 | 0.6166 |

- **Redundancy**: avg. pairwise cosine similarity among selected sentences (↓ better)
- **SrcCover**: fraction of source articles contributing at least one selected sentence (↑ better)
- BERTScore computed with `bert-base-multilingual-cased`

---

## 👥 Team

| Member | Student ID | Contribution |
|--------|------------|-------------|
| Tran Trung Duc | V202401788 | Proposed pipeline, TF-IDF experiments, first report draft |
| Le Anh Thu | V202503040 | Streamlit application, deployment, experiment design |
| Luu Duc Toan | V202502963 | BERT Centroid, PACSUM, BERTScore evaluation |
| Nguyen Van Cuong | V202502961 | Result consolidation, final report, GitHub management |

---

## 📄 How It Works

1. **Input**: User pastes a Vietnamese news article (optionally with a title)
2. **Sentence splitting**: Regex-based sentence tokenizer for Vietnamese
3. **Representation**: TF-IDF sparse vectors (LexRank family) or BERT dense embeddings (BERT Centroid, PACSUM)
4. **LexRank graph**: Sentences become nodes; edges weighted by cosine similarity above `threshold`
5. **Position prior**: Exponential decay — earlier sentences receive higher prior probability
6. **Title similarity**: Each sentence scored by cosine similarity with the article title
7. **Combined prior**: `p(i) = α · q_pos(i) + (1−α) · q_title(i)`
8. **MMR selection**: `MMR(c) = λ · Rel(c) − (1−λ) · max_sim(c, s), s ∈ S`
9. **Output**: Top-k sentences displayed in original document order with relevance scores
