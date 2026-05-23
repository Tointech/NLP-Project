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
        scores = cosine_similarity(tfidf[0], tfidf[1:]).flatten()
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
) -> tuple[np.ndarray, np.ndarray]:
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
    page_title="Vietnamese News Summarizer",
    page_icon="📰",
    layout="wide",
)

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
        line-height: 1.7;
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
    <h1>📰 Vietnamese News Summarizer</h1>
    <p>Position-Aware LexRank + MMR · Extractive Summarization</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Parameters")

    method = st.selectbox(
        "Method",
        options=["lead_k", "vanilla_lexrank", "position_lexrank", "position_lexrank_mmr"],
        format_func=lambda m: {
            "lead_k": "Lead-k (Baseline)",
            "vanilla_lexrank": "Vanilla LexRank",
            "position_lexrank": "Position-Aware LexRank",
            "position_lexrank_mmr": "Position-Aware LexRank + MMR ★",
        }[m],
        index=3,
    )

    k = st.slider("Number of sentences (k)", min_value=1, max_value=10, value=5)
    threshold = st.slider("Cosine threshold", 0.01, 0.30, 0.10, 0.01)
    position_weight = st.slider("Position weight", 0.1, 1.0, 0.80, 0.05)
    lambda_mmr = st.slider(
        "MMR Lambda (λ)", 0.1, 1.0, 0.70, 0.05,
        help="Higher → favour relevance. Lower → favour diversity."
    )

    st.divider()
    st.markdown("**Recommended:** Position-Aware LexRank + MMR with λ=0.7, threshold=0.1, position_weight=0.8")

    st.divider()
    st.markdown("### 📂 Load sample article")
    sample_choice = st.selectbox(
        "Choose a sample",
        ["— paste your own —", "Economy", "Technology", "Sports", "Environment"],
    )

# ── Sample texts ───────────────────────────────────────────
SAMPLES = {
    "Economy": {
        "title": "Kinh tế Việt Nam tăng trưởng 7,2% trong quý I năm 2025",
        "body": """Kinh tế Việt Nam tiếp tục ghi nhận tăng trưởng ấn tượng trong quý đầu năm 2025, với GDP tăng 7,2% so với cùng kỳ năm ngoái. Đây là mức tăng trưởng cao nhất trong vòng ba năm qua, phản ánh sự phục hồi mạnh mẽ của các ngành xuất khẩu và tiêu dùng nội địa. Bộ Kế hoạch và Đầu tư nhận định đây là tín hiệu tích cực cho thấy các chính sách kích thích kinh tế đã phát huy hiệu quả.

Xuất khẩu đạt 96 tỷ USD trong quý I, tăng 14% so với cùng kỳ năm 2024. Các mặt hàng chủ lực như điện tử, dệt may và giày dép đều ghi nhận mức tăng trưởng đáng kể. Riêng ngành điện tử đóng góp gần 40% tổng kim ngạch xuất khẩu, với Samsung và Intel dẫn đầu về giá trị. Thị trường Mỹ, EU và Trung Quốc tiếp tục là các đối tác thương mại lớn nhất của Việt Nam.

Về đầu tư nước ngoài, tổng vốn FDI đăng ký đạt 8,5 tỷ USD, tăng 20% so với cùng kỳ. Hàn Quốc, Nhật Bản và Singapore là ba quốc gia đầu tư nhiều nhất vào Việt Nam. Chính phủ đã phê duyệt nhiều dự án hạ tầng quy mô lớn, trong đó có tuyến đường sắt cao tốc Bắc – Nam và các khu công nghiệp công nghệ cao tại Hà Nội và TP.HCM.

Tuy nhiên, lạm phát vẫn là mối lo ngại khi chỉ số CPI tăng 3,8% so với cùng kỳ. Ngân hàng Nhà nước đã quyết định giữ nguyên lãi suất điều hành để cân bằng giữa mục tiêu kiểm soát lạm phát và hỗ trợ tăng trưởng. Các chuyên gia kinh tế khuyến nghị cần theo dõi chặt chẽ diễn biến giá cả hàng hóa và năng lượng trên thị trường quốc tế.

Thị trường lao động tiếp tục phục hồi với tỷ lệ thất nghiệp giảm xuống còn 2,1%, mức thấp nhất từ trước đến nay. Khu vực dịch vụ và công nghiệp chế biến chế tạo tạo ra nhiều việc làm nhất. Tiền lương bình quân tăng 6% so với cùng kỳ, góp phần thúc đẩy tiêu dùng trong nước. Chính phủ đặt mục tiêu tăng trưởng GDP cả năm 2025 đạt 7,0–7,5%.""",
    },
    "Technology": {
        "title": "Việt Nam đẩy mạnh phát triển trí tuệ nhân tạo và chuyển đổi số",
        "body": """Chính phủ Việt Nam vừa phê duyệt Chiến lược quốc gia về trí tuệ nhân tạo đến năm 2030, với tổng ngân sách đầu tư lên tới 2 tỷ USD. Chiến lược tập trung vào bốn lĩnh vực ưu tiên gồm y tế thông minh, nông nghiệp chính xác, thành phố thông minh và giáo dục số. Đây được xem là bước đi chiến lược để Việt Nam không bị tụt hậu trong cuộc cách mạng công nghiệp 4.0.

Bộ Thông tin và Truyền thông cho biết có hơn 500 doanh nghiệp công nghệ trong nước đang nghiên cứu và ứng dụng AI. Các công ty như VinAI, VinBigData và FPT AI đang dẫn đầu trong việc phát triển các giải pháp AI tiếng Việt. VinAI gần đây đã ra mắt mô hình ngôn ngữ lớn hỗ trợ tiếng Việt với hiệu suất vượt trội so với các mô hình quốc tế trên các tập dữ liệu tiếng Việt.

Hạ tầng số cũng được đầu tư mạnh mẽ với tốc độ phủ sóng 5G đạt 70% dân số vào cuối năm 2024. Việt Nam hiện có hơn 80 triệu người dùng internet, chiếm 82% dân số. Tốc độ internet di động của Việt Nam xếp thứ 38 thế giới theo báo cáo của Ookla. Chính phủ đặt mục tiêu phủ sóng 5G toàn quốc vào năm 2026.

Lĩnh vực thương mại điện tử bùng nổ với doanh thu đạt 25 tỷ USD trong năm 2024, tăng 35% so với năm trước. Shopee, Lazada và TikTok Shop cạnh tranh quyết liệt trên thị trường, trong khi các sàn thương mại điện tử nội địa như Tiki và Sendo nỗ lực giữ thị phần. Thanh toán không dùng tiền mặt đạt 60% tổng giao dịch bán lẻ, tăng đáng kể so với 40% của năm 2022.

Giáo dục công nghệ thông tin cũng được chú trọng, với 50 trường đại học mở thêm chương trình đào tạo về AI và khoa học dữ liệu. Số lượng sinh viên ngành CNTT tăng 40% trong vòng hai năm qua. Mục tiêu là có 100.000 chuyên gia AI vào năm 2030.""",
    },
    "Sports": {
        "title": "Đội tuyển bóng đá Việt Nam vào bán kết AFF Cup 2024",
        "body": """Đội tuyển bóng đá quốc gia Việt Nam đã giành chiến thắng thuyết phục 3-0 trước đội tuyển Myanmar trong trận tứ kết AFF Cup 2024 tổ chức tại sân Mỹ Đình, Hà Nội. Đây là lần thứ sáu liên tiếp Việt Nam lọt vào bán kết giải đấu khu vực Đông Nam Á này. Hơn 40.000 khán giả đã đến sân cổ vũ nhiệt tình cho đội nhà trong bầu không khí sôi động chưa từng có.

Tiền đạo Nguyễn Văn Toàn ghi hai bàn thắng ở phút 23 và phút 67, trở thành cầu thủ ghi nhiều bàn nhất tại giải đấu năm nay với tổng cộng năm bàn. Bàn thắng thứ ba đến từ chân của Hoàng Đức ở phút 85 sau pha bật cao đánh đầu hoàn hảo từ đường chuyền của Văn Hậu. Thủ môn Đặng Văn Lâm có màn trình diễn xuất sắc với ba pha cứu thua quan trọng trong hiệp một.

Huấn luyện viên trưởng Kim Sang-sik sau trận cho biết ông hài lòng với màn trình diễn tổng thể của đội tuyển, đặc biệt là khả năng phối hợp nhóm và tinh thần chiến đấu của các cầu thủ. Ông cũng nhấn mạnh rằng đội cần cải thiện hơn nữa khả năng phòng thủ trong các trận đấu tới.

Trong trận bán kết, Việt Nam sẽ đối đầu với Thái Lan — đội vô địch AFF Cup sáu lần. Trận lượt đi diễn ra tại Bangkok vào ngày 22 tháng 12, trận lượt về tại Hà Nội vào ngày 26 tháng 12. Lịch sử đối đầu gần đây nghiêng về phía Thái Lan, nhưng Việt Nam đã từng đánh bại họ tại AFF Cup 2018 và 2022. Giá vé trận lượt về tại sân Mỹ Đình đã được bán hết trong vòng hai giờ sau khi mở bán.""",
    },
    "Environment": {
        "title": "Việt Nam cam kết đạt phát thải ròng bằng 0 vào năm 2050",
        "body": """Tại Hội nghị COP29 diễn ra tại Azerbaijan, Việt Nam tái khẳng định cam kết đạt mức phát thải ròng bằng 0 vào năm 2050, đồng thời công bố kế hoạch chi tiết về chuyển dịch năng lượng. Đây là một trong những cam kết tham vọng nhất trong khu vực Đông Nam Á. Thủ tướng Phạm Minh Chính đã dẫn đầu đoàn đại biểu Việt Nam tham dự hội nghị và trực tiếp ký kết nhiều thỏa thuận hợp tác khí hậu quốc tế.

Bộ Công Thương cho biết Việt Nam sẽ giảm tỷ trọng điện than xuống dưới 20% vào năm 2030, từ mức 47% hiện nay. Năng lượng tái tạo, bao gồm điện gió và điện mặt trời, sẽ chiếm 45% tổng công suất phát điện vào năm 2030. Đến năm 2050, con số này sẽ tăng lên 70-80%. Các dự án điện gió ngoài khơi tại vùng biển Việt Nam đang được đẩy nhanh tiến độ với tổng công suất dự kiến 6.000 MW.

Tuy nhiên, quá trình chuyển dịch năng lượng gặp nhiều thách thức về tài chính. Việt Nam cần khoảng 400 tỷ USD trong 25 năm tới để thực hiện các mục tiêu khí hậu. Chương trình Đối tác chuyển dịch năng lượng công bằng cam kết cung cấp 15,5 tỷ USD từ các nước phát triển và tổ chức tài chính quốc tế.

Việt Nam cũng đang đối mặt với tác động ngày càng nghiêm trọng của biến đổi khí hậu. Đồng bằng sông Cửu Long đang bị xâm nhập mặn nghiêm trọng, ảnh hưởng đến sản xuất nông nghiệp của hàng triệu người dân. Bão và lũ lụt ngày càng xuất hiện thường xuyên và dữ dội hơn, gây thiệt hại hàng nghìn tỷ đồng mỗi năm. Nhiệt độ trung bình tại Việt Nam đã tăng 0,89°C trong 50 năm qua.

Chính phủ đã ban hành Nghị quyết về phát triển bền vững, yêu cầu tất cả các bộ ngành và địa phương tích hợp mục tiêu khí hậu vào kế hoạch phát triển kinh tế xã hội. Các doanh nghiệp sản xuất lớn phải tuân thủ lộ trình giảm phát thải nghiêm ngặt hoặc sẽ bị đánh thuế carbon bắt đầu từ năm 2026.""",
    },
}

# ── Determine input values from sidebar sample selector ───
if sample_choice != "— paste your own —":
    default_title = SAMPLES[sample_choice]["title"]
    default_body  = SAMPLES[sample_choice]["body"]
else:
    default_title = ""
    default_body  = ""

# ── Main layout ────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("📝 Article Input")

    title_input = st.text_input(
        "Article title (optional — improves sentence scoring)",
        value=default_title,
        placeholder="e.g. Vietnam economy grows strongly in Q2...",
    )

    article_input = st.text_area(
        "Article body",
        value=default_body,
        height=340,
        placeholder="Paste a Vietnamese news article here...",
    )

    run_btn = st.button("🚀 Summarise")

# ── Output ─────────────────────────────────────────────────
with col_out:
    st.subheader("📄 Summary Output")

    if run_btn or article_input:
        sentences = sentence_split(article_input)
        n_sent = len(sentences)

        if n_sent < 2:
            st.warning("⚠️ Text is too short. Please enter at least 2 sentences.")
        else:
            k_actual = min(k, n_sent)

            if method == "lead_k":
                selected = lead_k(sentences, k_actual)
            elif method == "vanilla_lexrank":
                selected = vanilla_lexrank(sentences, k_actual, threshold)
            elif method == "position_lexrank":
                selected = position_lexrank(sentences, k_actual, title_input, threshold, position_weight)
            else:
                selected = position_lexrank_mmr(sentences, k_actual, title_input, threshold, position_weight, lambda_mmr)

            redundancy  = compute_redundancy(sentences, selected)
            compression = 1 - len(selected) / n_sent

            # Metrics
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f'<div class="metric-card"><div class="value">{len(selected)}/{n_sent}</div><div class="label">Sentences selected</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="metric-card"><div class="value">{redundancy:.3f}</div><div class="label">Redundancy ↓</div></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div class="metric-card"><div class="value">{compression:.0%}</div><div class="label">Compression ratio</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            summary_text = " ".join(sentences[i] for i in selected)
            method_labels = {
                "lead_k": "Lead-k",
                "vanilla_lexrank": "Vanilla LexRank",
                "position_lexrank": "Position-Aware LexRank",
                "position_lexrank_mmr": "Position-Aware LexRank + MMR",
            }
            st.markdown(f'<span class="method-badge">{method_labels[method]}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download summary (.txt)",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain",
            )

            with st.expander("🔍 Sentence-level breakdown"):
                for idx, sent in enumerate(sentences):
                    if idx in selected:
                        rank = selected.index(idx) + 1
                        st.markdown(
                            f'<div class="sentence-selected">✅ <b>[Sentence {idx+1} → Rank #{rank}]</b> {sent}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="sentence-highlight">◻ [Sentence {idx+1}] {sent}</div>',
                            unsafe_allow_html=True,
                        )

# ── Compare all methods ─────────────────────────────────────
st.divider()
st.subheader("📊 Compare All Methods Side by Side")

if st.button("🔄 Run all 4 methods"):
    sentences = sentence_split(article_input)
    if len(sentences) < 2:
        st.warning("⚠️ Text is too short.")
    else:
        k_actual = min(k, len(sentences))
        methods_map = {
            "Lead-k (Baseline)": lead_k(sentences, k_actual),
            "Vanilla LexRank": vanilla_lexrank(sentences, k_actual, threshold),
            "Position-Aware LexRank": position_lexrank(sentences, k_actual, title_input, threshold, position_weight),
            "Position-Aware LexRank + MMR ★": position_lexrank_mmr(sentences, k_actual, title_input, threshold, position_weight, lambda_mmr),
        }
        cols = st.columns(2)
        for i, (name, sel) in enumerate(methods_map.items()):
            red  = compute_redundancy(sentences, sel)
            text = " ".join(sentences[j] for j in sel)
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                st.markdown(f"*Redundancy: {red:.4f} · Selected sentences: {sel}*")
                st.info(text)

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#9ca3af;font-size:0.8rem;'>"
    "Vietnamese Extractive News Summarization · Position-Aware LexRank + MMR"
    "</center>",
    unsafe_allow_html=True,
)
