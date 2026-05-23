"""
Vietnamese Extractive News Summarization — Interactive Demo
Position-Aware LexRank + MMR
"""

import streamlit as st
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────
# Core NLP logic (self-contained, no external model file)
# ─────────────────────────────────────────────────────────

def sentence_split(text: str) -> list[str]:
    """Split Vietnamese text into sentences."""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def build_similarity_matrix(sentences: list[str], threshold: float = 0.1) -> np.ndarray:
    """Compute cosine similarity matrix using TF-IDF."""
    if len(sentences) < 2:
        return np.eye(len(sentences))
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',  # Works for Vietnamese without word segmentation
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True
    )
    try:
        tfidf = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf, tfidf)
    except Exception:
        sim_matrix = np.eye(len(sentences))

    # Apply threshold: zero out weak edges
    sim_matrix[sim_matrix < threshold] = 0.0
    np.fill_diagonal(sim_matrix, 0.0)
    return sim_matrix


def lexrank_scores(sim_matrix: np.ndarray, max_iter: int = 100, damping: float = 0.85) -> np.ndarray:
    """Power iteration to compute LexRank centrality scores."""
    n = len(sim_matrix)
    if n == 0:
        return np.array([])
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_matrix = sim_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * norm_matrix.T @ scores
        if np.linalg.norm(new_scores - scores) < 1e-6:
            break
        scores = new_scores
    return scores


def position_prior(num_sentences: int, position_weight: float = 0.8) -> np.ndarray:
    """Exponential decay position prior — earlier sentences score higher."""
    positions = np.arange(num_sentences)
    prior = np.exp(-positions * position_weight / num_sentences)
    prior /= prior.sum()
    return prior


def title_similarity_scores(sentences: list[str], title: str) -> np.ndarray:
    """Score each sentence by cosine similarity with the title."""
    if not title.strip():
        return np.ones(len(sentences)) / len(sentences)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    try:
        corpus = [title] + sentences
        tfidf = vectorizer.fit_transform(corpus)
        title_vec = tfidf[0]
        sent_vecs = tfidf[1:]
        scores = cosine_similarity(title_vec, sent_vecs).flatten()
        if scores.sum() == 0:
            return np.ones(len(sentences)) / len(sentences)
        scores /= scores.sum()
    except Exception:
        scores = np.ones(len(sentences)) / len(sentences)
    return scores


def position_aware_lexrank(
    sentences: list[str],
    title: str = "",
    threshold: float = 0.1,
    position_weight: float = 0.8,
    title_alpha: float = 0.3,
) -> np.ndarray:
    """Combine LexRank centrality with position and title priors."""
    n = len(sentences)
    sim_matrix = build_similarity_matrix(sentences, threshold)
    lr_scores = lexrank_scores(sim_matrix)

    pos_prior = position_prior(n, position_weight)
    title_scores = title_similarity_scores(sentences, title)

    combined = lr_scores * (1 - title_alpha) + title_scores * title_alpha
    combined = combined * (1.0 - 0.3) + pos_prior * 0.3
    combined /= combined.sum() if combined.sum() > 0 else 1.0
    return combined, sim_matrix


def mmr_select(
    sentences: list[str],
    scores: np.ndarray,
    sim_matrix: np.ndarray,
    k: int,
    lambda_mmr: float = 0.7,
) -> list[int]:
    """Select k sentences using Maximal Marginal Relevance."""
    n = len(sentences)
    remaining = list(range(n))
    selected = []

    # Re-build a proper similarity matrix for MMR
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    try:
        tfidf = vectorizer.fit_transform(sentences)
        full_sim = cosine_similarity(tfidf, tfidf)
    except Exception:
        full_sim = sim_matrix

    for _ in range(k):
        if not remaining:
            break
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            best, best_score = None, -1e9
            for i in remaining:
                rel = scores[i]
                red = max(full_sim[i][j] for j in selected)
                mmr_score = lambda_mmr * rel - (1 - lambda_mmr) * red
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return sorted(selected)


def lead_k(sentences: list[str], k: int) -> list[int]:
    return list(range(min(k, len(sentences))))


def vanilla_lexrank(sentences: list[str], k: int, threshold: float = 0.1) -> list[int]:
    sim_matrix = build_similarity_matrix(sentences, threshold)
    lr_scores = lexrank_scores(sim_matrix)
    return sorted(np.argsort(lr_scores)[-k:].tolist())


def position_lexrank(
    sentences: list[str], k: int, title: str = "",
    threshold: float = 0.1, position_weight: float = 0.8
) -> list[int]:
    scores, _ = position_aware_lexrank(sentences, title, threshold, position_weight)
    return sorted(np.argsort(scores)[-k:].tolist())


def position_lexrank_mmr(
    sentences: list[str], k: int, title: str = "",
    threshold: float = 0.1, position_weight: float = 0.8, lambda_mmr: float = 0.7
) -> list[int]:
    scores, sim_matrix = position_aware_lexrank(sentences, title, threshold, position_weight)
    return mmr_select(sentences, scores, sim_matrix, k, lambda_mmr)


def compute_redundancy(sentences: list[str], indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0
    selected_sents = [sentences[i] for i in indices]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    try:
        tfidf = vectorizer.fit_transform(selected_sents)
        sim = cosine_similarity(tfidf, tfidf)
        n = len(selected_sents)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if not pairs:
            return 0.0
        return float(np.mean([sim[i][j] for i, j in pairs]))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tóm Tắt Tin Tức Tiếng Việt",
    page_icon="📰",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2rem; margin: 0; }
    .main-header p  { opacity: 0.85; margin: 0.5rem 0 0; }

    .summary-box {
        background: #f8f9ff;
        border-left: 4px solid #4f46e5;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .sentence-highlight {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.93rem;
    }
    .sentence-selected {
        background: #eff6ff;
        border: 1.5px solid #3b82f6;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.93rem;
        font-weight: 500;
    }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #4f46e5; }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; margin-top: 0.2rem; }
    .method-badge {
        display: inline-block;
        background: #4f46e5;
        color: white;
        border-radius: 6px;
        padding: 0.2rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📰 Tóm Tắt Tin Tức Tiếng Việt</h1>
    <p>Position-Aware LexRank + MMR · Extractive Summarization</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Parameters ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Tham số mô hình")

    method = st.selectbox(
        "Phương pháp",
        options=["lead_k", "vanilla_lexrank", "position_lexrank", "position_lexrank_mmr"],
        format_func=lambda m: {
            "lead_k": "Lead-k (Baseline)",
            "vanilla_lexrank": "Vanilla LexRank",
            "position_lexrank": "Position-Aware LexRank",
            "position_lexrank_mmr": "Position-Aware LexRank + MMR ★",
        }[m],
        index=3,
    )

    k = st.slider("Số câu tóm tắt (k)", min_value=1, max_value=10, value=5)
    threshold = st.slider("Ngưỡng cosine (threshold)", 0.01, 0.30, 0.10, 0.01)
    position_weight = st.slider("Trọng số vị trí (position_weight)", 0.1, 1.0, 0.80, 0.05)
    lambda_mmr = st.slider("Lambda MMR (λ)", 0.1, 1.0, 0.70, 0.05,
                           help="Cao → ưu tiên relevance. Thấp → giảm trùng lặp.")

    st.divider()
    st.markdown("**Phương pháp được khuyến nghị:** Position-Aware LexRank + MMR với λ=0.7, threshold=0.1, position_weight=0.8")

# ── Main Input ────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("📝 Nhập văn bản")

    title_input = st.text_input(
        "Tiêu đề bài báo (tuỳ chọn)",
        placeholder="VD: Kinh tế Việt Nam tăng trưởng mạnh trong quý 2...",
    )

    SAMPLE_TEXT = """Kinh tế Việt Nam tiếp tục ghi nhận tăng trưởng ấn tượng trong quý đầu năm 2025, với GDP tăng 7,2% so với cùng kỳ năm ngoái. Đây là mức tăng trưởng cao nhất trong vòng ba năm qua, phản ánh sự phục hồi mạnh mẽ của các ngành xuất khẩu và tiêu dùng nội địa. Bộ Kế hoạch và Đầu tư nhận định đây là tín hiệu tích cực cho thấy các chính sách kích thích kinh tế đã phát huy hiệu quả.

Xuất khẩu đạt 96 tỷ USD trong quý I, tăng 14% so với cùng kỳ năm 2024. Các mặt hàng chủ lực như điện tử, dệt may và giày dép đều ghi nhận mức tăng trưởng đáng kể. Riêng ngành điện tử đóng góp gần 40% tổng kim ngạch xuất khẩu, với Samsung và Intel dẫn đầu về giá trị. Thị trường Mỹ, EU và Trung Quốc tiếp tục là các đối tác thương mại lớn nhất của Việt Nam.

Về đầu tư nước ngoài, tổng vốn FDI đăng ký đạt 8,5 tỷ USD, tăng 20% so với cùng kỳ. Hàn Quốc, Nhật Bản và Singapore là ba quốc gia đầu tư nhiều nhất vào Việt Nam. Chính phủ đã phê duyệt nhiều dự án hạ tầng quy mô lớn, trong đó có tuyến đường sắt cao tốc Bắc – Nam và các khu công nghiệp công nghệ cao tại Hà Nội và TP.HCM.

Tuy nhiên, lạm phát vẫn là mối lo ngại khi chỉ số CPI tăng 3,8% so với cùng kỳ. Ngân hàng Nhà nước đã quyết định giữ nguyên lãi suất điều hành để cân bằng giữa mục tiêu kiểm soát lạm phát và hỗ trợ tăng trưởng. Các chuyên gia kinh tế khuyến nghị cần theo dõi chặt chẽ diễn biến giá cả hàng hóa và năng lượng trên thị trường quốc tế.

Thị trường lao động tiếp tục phục hồi với tỷ lệ thất nghiệp giảm xuống còn 2,1%, mức thấp nhất từ trước đến nay. Khu vực dịch vụ và công nghiệp chế biến chế tạo tạo ra nhiều việc làm nhất. Tiền lương bình quân tăng 6% so với cùng kỳ, góp phần thúc đẩy tiêu dùng trong nước. Chính phủ đặt mục tiêu tăng trưởng GDP cả năm 2025 đạt 7,0–7,5%."""

    article_input = st.text_area(
        "Nội dung bài báo",
        value=SAMPLE_TEXT,
        height=320,
        placeholder="Dán nội dung bài báo tiếng Việt vào đây...",
    )

    run_btn = st.button("🚀 Tóm tắt ngay")

# ── Output ────────────────────────────────────────────────
with col_out:
    st.subheader("📄 Kết quả tóm tắt")

    if run_btn or article_input:
        sentences = sentence_split(article_input)
        n_sent = len(sentences)

        if n_sent < 2:
            st.warning("⚠️ Văn bản quá ngắn. Vui lòng nhập ít nhất 2 câu.")
        else:
            k_actual = min(k, n_sent)

            # ── Run selected method ────────────────────────
            if method == "lead_k":
                selected = lead_k(sentences, k_actual)
            elif method == "vanilla_lexrank":
                selected = vanilla_lexrank(sentences, k_actual, threshold)
            elif method == "position_lexrank":
                selected = position_lexrank(sentences, k_actual, title_input, threshold, position_weight)
            else:
                selected = position_lexrank_mmr(sentences, k_actual, title_input, threshold, position_weight, lambda_mmr)

            redundancy = compute_redundancy(sentences, selected)
            coverage = len(selected) / n_sent if n_sent > 0 else 0.0
            compression = 1 - len(selected) / n_sent

            # ── Metrics ───────────────────────────────────
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f"""<div class="metric-card"><div class="value">{len(selected)}/{n_sent}</div><div class="label">Câu được chọn</div></div>""", unsafe_allow_html=True)
            mc2.markdown(f"""<div class="metric-card"><div class="value">{redundancy:.3f}</div><div class="label">Redundancy ↓</div></div>""", unsafe_allow_html=True)
            mc3.markdown(f"""<div class="metric-card"><div class="value">{compression:.0%}</div><div class="label">Tỷ lệ nén</div></div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Summary text ──────────────────────────────
            summary_text = " ".join(sentences[i] for i in selected)
            st.markdown(f'<span class="method-badge">{method}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)

            # ── Download ──────────────────────────────────
            st.download_button(
                "⬇️ Tải xuống tóm tắt",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain",
            )

            # ── Sentence-level breakdown ──────────────────
            with st.expander("🔍 Xem chi tiết từng câu"):
                for idx, sent in enumerate(sentences):
                    if idx in selected:
                        rank = selected.index(idx) + 1
                        st.markdown(
                            f'<div class="sentence-selected">✅ <b>[Câu {idx+1} → Rank #{rank}]</b> {sent}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="sentence-highlight">◻ [Câu {idx+1}] {sent}</div>',
                            unsafe_allow_html=True,
                        )

# ── Compare all methods ────────────────────────────────────
st.divider()
st.subheader("📊 So sánh tất cả phương pháp")

if st.button("🔄 So sánh 4 phương pháp cùng lúc"):
    sentences = sentence_split(article_input)
    if len(sentences) < 2:
        st.warning("⚠️ Văn bản quá ngắn.")
    else:
        k_actual = min(k, len(sentences))
        methods_map = {
            "Lead-k": lead_k(sentences, k_actual),
            "Vanilla LexRank": vanilla_lexrank(sentences, k_actual, threshold),
            "Position-Aware LexRank": position_lexrank(sentences, k_actual, title_input, threshold, position_weight),
            "Position-Aware LexRank + MMR ★": position_lexrank_mmr(sentences, k_actual, title_input, threshold, position_weight, lambda_mmr),
        }

        cols = st.columns(2)
        for i, (name, sel) in enumerate(methods_map.items()):
            red = compute_redundancy(sentences, sel)
            text = " ".join(sentences[j] for j in sel)
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                st.markdown(f"*Redundancy: {red:.4f} | Câu được chọn: {sel}*")
                st.info(text)

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#9ca3af;font-size:0.8rem;'>"
    "Vietnamese Extractive News Summarization · Position-Aware LexRank + MMR"
    "</center>",
    unsafe_allow_html=True,
)
