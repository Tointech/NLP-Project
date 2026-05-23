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
    st.markdown("**Recommended:** Position-Aware LexRank + MMR with λ=0.7, threshold=0.1, position_weight=0.8")

# ── Sample article library ─────────────────────────────────
SAMPLES = {
    "📈 Economy": [
        {
            "label": "GDP Growth Q1 2025",
            "title": "Kinh tế Việt Nam tăng trưởng 7,2% trong quý I năm 2025",
            "body": """Kinh tế Việt Nam tiếp tục ghi nhận tăng trưởng ấn tượng trong quý đầu năm 2025, với GDP tăng 7,2% so với cùng kỳ năm ngoái. Đây là mức tăng trưởng cao nhất trong vòng ba năm qua, phản ánh sự phục hồi mạnh mẽ của các ngành xuất khẩu và tiêu dùng nội địa. Bộ Kế hoạch và Đầu tư nhận định đây là tín hiệu tích cực cho thấy các chính sách kích thích kinh tế đã phát huy hiệu quả.

Xuất khẩu đạt 96 tỷ USD trong quý I, tăng 14% so với cùng kỳ năm 2024. Các mặt hàng chủ lực như điện tử, dệt may và giày dép đều ghi nhận mức tăng trưởng đáng kể. Riêng ngành điện tử đóng góp gần 40% tổng kim ngạch xuất khẩu, với Samsung và Intel dẫn đầu về giá trị. Thị trường Mỹ, EU và Trung Quốc tiếp tục là các đối tác thương mại lớn nhất của Việt Nam.

Về đầu tư nước ngoài, tổng vốn FDI đăng ký đạt 8,5 tỷ USD, tăng 20% so với cùng kỳ. Hàn Quốc, Nhật Bản và Singapore là ba quốc gia đầu tư nhiều nhất vào Việt Nam. Chính phủ đã phê duyệt nhiều dự án hạ tầng quy mô lớn, trong đó có tuyến đường sắt cao tốc Bắc – Nam và các khu công nghiệp công nghệ cao tại Hà Nội và TP.HCM.

Tuy nhiên, lạm phát vẫn là mối lo ngại khi chỉ số CPI tăng 3,8% so với cùng kỳ. Ngân hàng Nhà nước đã quyết định giữ nguyên lãi suất điều hành để cân bằng giữa mục tiêu kiểm soát lạm phát và hỗ trợ tăng trưởng. Các chuyên gia kinh tế khuyến nghị cần theo dõi chặt chẽ diễn biến giá cả hàng hóa và năng lượng trên thị trường quốc tế.

Thị trường lao động tiếp tục phục hồi với tỷ lệ thất nghiệp giảm xuống còn 2,1%, mức thấp nhất từ trước đến nay. Khu vực dịch vụ và công nghiệp chế biến chế tạo tạo ra nhiều việc làm nhất. Tiền lương bình quân tăng 6% so với cùng kỳ, góp phần thúc đẩy tiêu dùng trong nước. Chính phủ đặt mục tiêu tăng trưởng GDP cả năm 2025 đạt 7,0–7,5%.""",
        },
        {
            "label": "FDI Attraction Policy",
            "title": "Việt Nam thu hút FDI kỷ lục với chính sách ưu đãi mới",
            "body": """Việt Nam đã thu hút được 36,6 tỷ USD vốn đầu tư trực tiếp nước ngoài trong năm 2024, mức cao nhất trong lịch sử. Chính phủ đã ban hành nhiều chính sách ưu đãi thuế và đất đai để thu hút các tập đoàn đa quốc gia lớn. Các khu công nghiệp thế hệ mới được xây dựng đồng bộ hạ tầng, đáp ứng tiêu chuẩn quốc tế về môi trường và công nghệ.

Intel, Samsung, LG và LEGO đã công bố mở rộng đầu tư tại Việt Nam với tổng vốn hơn 5 tỷ USD. Ngành bán dẫn và điện tử được xác định là lĩnh vực ưu tiên thu hút đầu tư chiến lược. Chính phủ đặt mục tiêu đến năm 2030, Việt Nam sẽ nằm trong top 3 quốc gia sản xuất điện tử lớn nhất khu vực ASEAN.

Tuy nhiên, thiếu hụt lao động chất lượng cao vẫn là thách thức lớn. Hệ thống đào tạo nghề và đại học đang được cải cách để đáp ứng nhu cầu của doanh nghiệp FDI. Bộ Lao động, Thương binh và Xã hội đã phê duyệt chương trình đào tạo 50.000 kỹ sư bán dẫn đến năm 2030.

Hạ tầng logistics cũng được đầu tư mạnh để hỗ trợ xuất khẩu. Cảng Lạch Huyện giai đoạn 2 và sân bay Long Thành đang được đẩy nhanh tiến độ. Chi phí logistics của Việt Nam hiện chiếm khoảng 16% GDP, chính phủ đặt mục tiêu giảm xuống 12% vào năm 2030.

Môi trường kinh doanh tiếp tục được cải thiện với thứ hạng của Việt Nam tăng 12 bậc trong bảng xếp hạng năng lực cạnh tranh toàn cầu. Thủ tục hành chính được đơn giản hóa, thời gian cấp phép đầu tư giảm từ 30 ngày xuống còn 15 ngày.""",
        },
    ],
    "💻 Technology": [
        {
            "label": "AI National Strategy",
            "title": "Việt Nam đẩy mạnh phát triển trí tuệ nhân tạo và chuyển đổi số",
            "body": """Chính phủ Việt Nam vừa phê duyệt Chiến lược quốc gia về trí tuệ nhân tạo đến năm 2030, với tổng ngân sách đầu tư lên tới 2 tỷ USD. Chiến lược tập trung vào bốn lĩnh vực ưu tiên gồm y tế thông minh, nông nghiệp chính xác, thành phố thông minh và giáo dục số. Đây được xem là bước đi chiến lược để Việt Nam không bị tụt hậu trong cuộc cách mạng công nghiệp 4.0.

Bộ Thông tin và Truyền thông cho biết có hơn 500 doanh nghiệp công nghệ trong nước đang nghiên cứu và ứng dụng AI. Các công ty như VinAI, VinBigData và FPT AI đang dẫn đầu trong việc phát triển các giải pháp AI tiếng Việt. VinAI gần đây đã ra mắt mô hình ngôn ngữ lớn hỗ trợ tiếng Việt với hiệu suất vượt trội so với các mô hình quốc tế trên các tập dữ liệu tiếng Việt.

Hạ tầng số cũng được đầu tư mạnh mẽ với tốc độ phủ sóng 5G đạt 70% dân số vào cuối năm 2024. Việt Nam hiện có hơn 80 triệu người dùng internet, chiếm 82% dân số. Tốc độ internet di động của Việt Nam xếp thứ 38 thế giới theo báo cáo của Ookla. Chính phủ đặt mục tiêu phủ sóng 5G toàn quốc vào năm 2026.

Lĩnh vực thương mại điện tử bùng nổ với doanh thu đạt 25 tỷ USD trong năm 2024, tăng 35% so với năm trước. Shopee, Lazada và TikTok Shop cạnh tranh quyết liệt trên thị trường, trong khi các sàn thương mại điện tử nội địa như Tiki và Sendo nỗ lực giữ thị phần. Thanh toán không dùng tiền mặt đạt 60% tổng giao dịch bán lẻ, tăng đáng kể so với 40% của năm 2022.

Giáo dục công nghệ thông tin cũng được chú trọng, với 50 trường đại học mở thêm chương trình đào tạo về AI và khoa học dữ liệu. Số lượng sinh viên ngành CNTT tăng 40% trong vòng hai năm qua. Mục tiêu là có 100.000 chuyên gia AI vào năm 2030.""",
        },
        {
            "label": "Semiconductor Industry",
            "title": "Việt Nam gia nhập chuỗi cung ứng bán dẫn toàn cầu",
            "body": """Việt Nam đang nổi lên như một trung tâm sản xuất bán dẫn quan trọng trong chuỗi cung ứng toàn cầu. Tập đoàn Intel đã đầu tư hơn 1,5 tỷ USD vào nhà máy lắp ráp và kiểm định chip tại TP.HCM, trở thành cơ sở sản xuất chip lớn nhất của Intel tại Đông Nam Á. Samsung cũng vừa công bố kế hoạch đầu tư thêm 2 tỷ USD để mở rộng sản xuất chip nhớ tại Thái Nguyên.

Chính phủ đã thành lập Trung tâm Đổi mới Sáng tạo Quốc gia tại Hà Nội và TP.HCM nhằm hỗ trợ startup trong lĩnh vực bán dẫn và công nghệ cao. Quỹ Đổi mới Sáng tạo Quốc gia với vốn 2.000 tỷ đồng sẽ đầu tư vào các dự án nghiên cứu thiết kế chip. Đây là bước đi quan trọng để Việt Nam không chỉ làm gia công mà còn tham gia vào khâu thiết kế có giá trị gia tăng cao.

Tuy nhiên, ngành bán dẫn Việt Nam vẫn còn ở giai đoạn đầu phát triển. Hầu hết hoạt động hiện tại tập trung vào lắp ráp, đóng gói và kiểm định — các công đoạn có giá trị gia tăng thấp hơn so với thiết kế và chế tạo wafer. Thiếu kỹ sư thiết kế chip là rào cản lớn nhất, khi mỗi năm Việt Nam chỉ đào tạo được khoảng 500 kỹ sư chuyên ngành bán dẫn.

Để giải quyết vấn đề nhân lực, Bộ Giáo dục và Đào tạo đã phê duyệt chương trình đào tạo kỹ sư thiết kế vi mạch tại 10 trường đại học trọng điểm. Các tập đoàn như Synopsys, Cadence và NVIDIA cam kết hỗ trợ chương trình đào tạo thông qua học bổng và trang thiết bị phòng lab. Việt Nam đặt mục tiêu có 50.000 kỹ sư bán dẫn vào năm 2030.""",
        },
    ],
    "⚽ Sports": [
        {
            "label": "AFF Cup 2024 Semi-final",
            "title": "Đội tuyển bóng đá Việt Nam vào bán kết AFF Cup 2024",
            "body": """Đội tuyển bóng đá quốc gia Việt Nam đã giành chiến thắng thuyết phục 3-0 trước đội tuyển Myanmar trong trận tứ kết AFF Cup 2024 tổ chức tại sân Mỹ Đình, Hà Nội. Đây là lần thứ sáu liên tiếp Việt Nam lọt vào bán kết giải đấu khu vực Đông Nam Á này. Hơn 40.000 khán giả đã đến sân cổ vũ nhiệt tình cho đội nhà trong bầu không khí sôi động chưa từng có.

Tiền đạo Nguyễn Văn Toàn ghi hai bàn thắng ở phút 23 và phút 67, trở thành cầu thủ ghi nhiều bàn nhất tại giải đấu năm nay với tổng cộng năm bàn. Bàn thắng thứ ba đến từ chân của Hoàng Đức ở phút 85 sau pha bật cao đánh đầu hoàn hảo từ đường chuyền của Văn Hậu. Thủ môn Đặng Văn Lâm có màn trình diễn xuất sắc với ba pha cứu thua quan trọng trong hiệp một.

Huấn luyện viên trưởng Kim Sang-sik sau trận cho biết ông hài lòng với màn trình diễn tổng thể của đội tuyển, đặc biệt là khả năng phối hợp nhóm và tinh thần chiến đấu của các cầu thủ. Ông cũng nhấn mạnh rằng đội cần cải thiện hơn nữa khả năng phòng thủ trong các trận đấu tới.

Trong trận bán kết, Việt Nam sẽ đối đầu với Thái Lan — đội vô địch AFF Cup sáu lần. Trận lượt đi diễn ra tại Bangkok vào ngày 22 tháng 12, trận lượt về tại Hà Nội vào ngày 26 tháng 12. Lịch sử đối đầu gần đây nghiêng về phía Thái Lan, nhưng Việt Nam đã từng đánh bại họ tại AFF Cup 2018 và 2022. Giá vé trận lượt về tại sân Mỹ Đình đã được bán hết trong vòng hai giờ sau khi mở bán.""",
        },
        {
            "label": "SEA Games Preparation",
            "title": "Thể thao Việt Nam chuẩn bị cho SEA Games 2025",
            "body": """Uỷ ban Olympic Việt Nam đã công bố kế hoạch chuẩn bị toàn diện cho SEA Games 34 diễn ra tại Thái Lan vào tháng 12 năm 2025. Đoàn thể thao Việt Nam dự kiến cử hơn 700 vận động viên tham dự 40 môn thi đấu, với mục tiêu giành ít nhất 65 huy chương vàng. Đây sẽ là thử thách lớn khi Thái Lan đăng cai và có lợi thế sân nhà.

Bộ Văn hóa, Thể thao và Du lịch đã phê duyệt ngân sách 500 tỷ đồng cho công tác chuẩn bị, tăng 30% so với kỳ SEA Games trước. Các trung tâm huấn luyện thể thao quốc gia tại Hà Nội và TP.HCM được đầu tư nâng cấp trang thiết bị hiện đại. Chương trình tập huấn nước ngoài cũng được mở rộng, với hàng chục đội tuyển quốc gia được cử đi tập huấn tại Hàn Quốc, Nhật Bản và châu Âu.

Bóng đá, bơi lội, điền kinh và các môn võ thuật được xác định là thế mạnh của thể thao Việt Nam. Kỳ thủ cờ vua Lê Quang Liêm và tay vợt cầu lông Nguyễn Tiến Minh sẽ là những gương mặt tiêu biểu. Riêng môn bơi lội, Nguyễn Huy Hoàng đang được kỳ vọng lớn sau thành tích phá kỷ lục SEA Games ở kỳ trước.

Công tác phát triển thể thao trẻ cũng được chú trọng nhằm chuẩn bị cho tương lai. Học viện bóng đá Hoàng Anh Gia Lai-Arsenal đã sản sinh ra nhiều tài năng trẻ cho đội tuyển quốc gia. Chính phủ đặt mục tiêu xây dựng 500 trung tâm thể thao cộng đồng trên cả nước vào năm 2030 để phát triển thể thao đại chúng.""",
        },
    ],
    "🌿 Environment": [
        {
            "label": "Net Zero 2050 Commitment",
            "title": "Việt Nam cam kết đạt phát thải ròng bằng 0 vào năm 2050",
            "body": """Tại Hội nghị COP29 diễn ra tại Azerbaijan, Việt Nam tái khẳng định cam kết đạt mức phát thải ròng bằng 0 vào năm 2050, đồng thời công bố kế hoạch chi tiết về chuyển dịch năng lượng. Đây là một trong những cam kết tham vọng nhất trong khu vực Đông Nam Á. Thủ tướng Phạm Minh Chính đã dẫn đầu đoàn đại biểu Việt Nam tham dự hội nghị và trực tiếp ký kết nhiều thỏa thuận hợp tác khí hậu quốc tế.

Bộ Công Thương cho biết Việt Nam sẽ giảm tỷ trọng điện than xuống dưới 20% vào năm 2030, từ mức 47% hiện nay. Năng lượng tái tạo, bao gồm điện gió và điện mặt trời, sẽ chiếm 45% tổng công suất phát điện vào năm 2030. Đến năm 2050, con số này sẽ tăng lên 70-80%. Các dự án điện gió ngoài khơi tại vùng biển Việt Nam đang được đẩy nhanh tiến độ với tổng công suất dự kiến 6.000 MW.

Tuy nhiên, quá trình chuyển dịch năng lượng gặp nhiều thách thức về tài chính. Việt Nam cần khoảng 400 tỷ USD trong 25 năm tới để thực hiện các mục tiêu khí hậu. Chương trình Đối tác chuyển dịch năng lượng công bằng cam kết cung cấp 15,5 tỷ USD từ các nước phát triển và tổ chức tài chính quốc tế.

Việt Nam cũng đang đối mặt với tác động ngày càng nghiêm trọng của biến đổi khí hậu. Đồng bằng sông Cửu Long đang bị xâm nhập mặn nghiêm trọng, ảnh hưởng đến sản xuất nông nghiệp của hàng triệu người dân. Bão và lũ lụt ngày càng xuất hiện thường xuyên và dữ dội hơn, gây thiệt hại hàng nghìn tỷ đồng mỗi năm. Nhiệt độ trung bình tại Việt Nam đã tăng 0,89°C trong 50 năm qua.

Chính phủ đã ban hành Nghị quyết về phát triển bền vững, yêu cầu tất cả các bộ ngành và địa phương tích hợp mục tiêu khí hậu vào kế hoạch phát triển kinh tế xã hội. Các doanh nghiệp sản xuất lớn phải tuân thủ lộ trình giảm phát thải nghiêm ngặt hoặc sẽ bị đánh thuế carbon bắt đầu từ năm 2026.""",
        },
        {
            "label": "Mekong Delta Crisis",
            "title": "Đồng bằng sông Cửu Long đối mặt với khủng hoảng nước và xâm nhập mặn",
            "body": """Đồng bằng sông Cửu Long, vựa lúa và trái cây lớn nhất Việt Nam, đang đối mặt với cuộc khủng hoảng môi trường chưa từng có. Xâm nhập mặn năm 2025 được dự báo nghiêm trọng hơn năm 2016 — năm từng gây thiệt hại hơn 5.500 tỷ đồng cho nông nghiệp. Hơn 82.000 hộ dân tại 8 tỉnh ven biển đang thiếu nước ngọt sinh hoạt trầm trọng trong mùa khô.

Nguyên nhân chính bao gồm biến đổi khí hậu làm mực nước biển dâng, các đập thủy điện thượng nguồn sông Mê Kông làm giảm lượng phù sa và nước ngọt về hạ lưu, cùng với việc khai thác nước ngầm quá mức khiến mặt đất sụt lún. Tốc độ sụt lún tại một số khu vực đô thị ven biển lên đến 2-3 cm/năm, nhanh hơn nhiều so với mức nước biển dâng. Nếu không có biện pháp can thiệp, đến năm 2100, một phần lớn đồng bằng có thể bị ngập.

Chính phủ đã triển khai nhiều giải pháp khẩn cấp như xây dựng hệ thống cống ngăn mặn, nạo vét kênh rạch và cấp nước ngọt cho dân. Dự án cống Cái Lớn - Cái Bé trị giá 3.300 tỷ đồng đã hoàn thành và đang phát huy hiệu quả trong việc kiểm soát mặn cho vùng tứ giác Long Xuyên. Tuy nhiên, các chuyên gia cảnh báo đây chỉ là giải pháp tình thế, cần có chiến lược dài hạn hơn.

Về lâu dài, chính phủ đang thúc đẩy chuyển đổi cơ cấu nông nghiệp theo hướng thích ứng với xâm nhập mặn. Mô hình tôm-lúa, nuôi trồng thủy sản nước lợ và trồng các loại cây chịu mặn đang được nhân rộng. Hàng trăm nghìn hộ nông dân đã được hỗ trợ chuyển đổi sang các mô hình sản xuất bền vững hơn. Bộ Nông nghiệp đặt mục tiêu đến năm 2030 có 30% diện tích nông nghiệp vùng đồng bằng chuyển sang mô hình thích ứng khí hậu.""",
        },
    ],
    "🏥 Health": [
        {
            "label": "Healthcare System Reform",
            "title": "Việt Nam cải cách hệ thống y tế hướng tới bao phủ sức khỏe toàn dân",
            "body": """Bộ Y tế Việt Nam vừa công bố Chiến lược cải cách hệ thống y tế giai đoạn 2025-2030 với mục tiêu đạt bao phủ sức khỏe toàn dân. Tổng ngân sách dành cho y tế sẽ tăng lên 10% GDP vào năm 2030, từ mức 6,5% hiện nay. Đây là bước đột phá quan trọng trong việc nâng cao chất lượng chăm sóc sức khỏe cho 100 triệu người dân Việt Nam.

Hệ thống y tế cơ sở được xác định là trọng tâm cải cách. Hơn 11.000 trạm y tế xã, phường trên cả nước sẽ được nâng cấp thiết bị và nhân lực. Mỗi trạm y tế sẽ có ít nhất một bác sĩ thay vì chỉ có y sĩ như hiện nay. Chương trình đào tạo bác sĩ gia đình được đẩy mạnh với mục tiêu có 35.000 bác sĩ gia đình vào năm 2030.

Ứng dụng công nghệ thông tin trong y tế đang được triển khai rộng rãi. Hồ sơ sức khỏe điện tử đã được triển khai cho hơn 60 triệu người dân. Nền tảng telehealth cho phép người dân khám bệnh từ xa đã có hơn 5 triệu lượt sử dụng trong năm 2024. Hệ thống AI hỗ trợ chẩn đoán ung thư và đọc phim X-quang đang được thí điểm tại 50 bệnh viện lớn.

Bảo hiểm y tế toàn dân đã đạt tỷ lệ bao phủ 93%, mục tiêu đến 2025 đạt 95%. Gói quyền lợi bảo hiểm y tế được mở rộng để bao gồm thêm nhiều loại thuốc và dịch vụ kỹ thuật cao. Chi phí y tế bình quân đầu người tăng từ 130 USD lên 180 USD/năm trong giai đoạn 2020-2024, phản ánh sự cải thiện về chất lượng dịch vụ.

Việt Nam cũng đang phát triển ngành công nghiệp dược phẩm trong nước. Tỷ lệ thuốc sản xuất trong nước đạt 50% nhu cầu, mục tiêu tăng lên 70% vào năm 2030. Các doanh nghiệp dược lớn như DHG, Imexpharm đang đầu tư vào nghiên cứu phát triển thuốc generic và dược liệu. Việt Nam đặt mục tiêu xuất khẩu dược phẩm đạt 1 tỷ USD vào năm 2030.""",
        },
    ],
    "🎓 Education": [
        {
            "label": "University Autonomy Reform",
            "title": "Các trường đại học Việt Nam tự chủ: Cơ hội và thách thức",
            "body": """Sau 5 năm thực hiện cơ chế tự chủ đại học theo Nghị định 99/2019/NĐ-CP, hệ thống giáo dục đại học Việt Nam đã có nhiều chuyển biến tích cực. Hiện có 23 trường đại học được thí điểm tự chủ toàn diện, trong đó có Đại học Quốc gia Hà Nội, Đại học Quốc gia TP.HCM và nhiều trường công lập lớn. Học phí trung bình tăng 3-5 lần nhưng chất lượng đào tạo và cơ sở vật chất cũng được cải thiện đáng kể.

Các trường tự chủ đã có thêm nguồn lực để thu hút giảng viên giỏi, đầu tư phòng thí nghiệm và xây dựng chương trình đào tạo theo chuẩn quốc tế. Một số trường như Đại học Bách khoa Hà Nội và Đại học Kinh tế TP.HCM đã lọt vào top 500 đại học hàng đầu châu Á. Hợp tác quốc tế trong nghiên cứu và giảng dạy cũng được mở rộng đáng kể.

Tuy nhiên, tự chủ đại học cũng đặt ra nhiều thách thức. Học phí tăng cao tạo áp lực cho sinh viên từ gia đình khó khăn, dù các trường đã tăng số lượng học bổng. Cơ chế quản trị nội bộ tại nhiều trường vẫn chưa thực sự minh bạch và hiệu quả. Một số trường lợi dụng tự chủ để mở ngành tràn lan mà không đảm bảo chất lượng đầu ra.

Bộ Giáo dục và Đào tạo đang xây dựng khung pháp lý mới để tăng cường giám sát chất lượng trong môi trường tự chủ. Hệ thống kiểm định chất lượng độc lập sẽ được tăng cường với sự tham gia của các tổ chức quốc tế. Các tiêu chí đánh giá xếp hạng đại học Việt Nam cũng đang được cập nhật để phù hợp với chuẩn mực quốc tế như QS và THE.

Xu hướng liên kết đào tạo với doanh nghiệp ngày càng mạnh mẽ. Nhiều trường đại học đã ký hợp đồng đào tạo theo đặt hàng với các tập đoàn lớn như FPT, Vingroup và Samsung. Mô hình này giúp sinh viên có việc làm ngay sau tốt nghiệp và doanh nghiệp có nguồn nhân lực chất lượng phù hợp với nhu cầu thực tế.""",
        },
    ],
}


# ── Resource selector ──────────────────────────────────────
def render_resource_selector() -> tuple[str, str]:
    """
    Display a tabbed article picker.
    Returns (selected_title, selected_body).
    Persists the user's choice in st.session_state.
    """
    st.subheader("📚 Select Article Source")

    tab_labels = list(SAMPLES.keys()) + ["📁 Upload file", "✏️ Paste your own"]
    tabs = st.tabs(tab_labels)

    chosen_title, chosen_body = "", ""

    # ── Built-in sample tabs ───────────────────────────────
    for tab, (category, articles) in zip(tabs, SAMPLES.items()):
        with tab:
            cols = st.columns(len(articles))
            for col, article in zip(cols, articles):
                with col:
                    # Card-style container
                    with st.container(border=True):
                        st.markdown(f"**{article['label']}**")
                        preview = article["body"][:120].replace("\n", " ") + "…"
                        st.caption(preview)
                        if st.button("Load this article", key=f"load_{article['label']}"):
                            st.session_state["sel_title"] = article["title"]
                            st.session_state["sel_body"]  = article["body"]
                            st.session_state["sel_source"] = article["label"]

    # ── Upload tab ─────────────────────────────────────────
    with tabs[-2]:
        st.markdown("Upload a plain-text (`.txt`) Vietnamese news article.")
        uploaded = st.file_uploader("Choose a .txt file", type=["txt"], label_visibility="collapsed")
        if uploaded:
            raw = uploaded.read().decode("utf-8", errors="replace")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            # Try to auto-detect a title on the first line
            guessed_title = lines[0] if lines else ""
            guessed_body  = "\n\n".join(lines[1:]) if len(lines) > 1 else raw
            if st.button("Load uploaded file", key="load_upload"):
                st.session_state["sel_title"]  = guessed_title
                st.session_state["sel_body"]   = guessed_body
                st.session_state["sel_source"] = uploaded.name

    # ── Paste tab ──────────────────────────────────────────
    with tabs[-1]:
        st.markdown("Type or paste any Vietnamese article directly.")
        manual_title = st.text_input("Title (optional)", key="manual_title_input",
                                     placeholder="e.g. Kinh tế Việt Nam tăng trưởng...")
        manual_body  = st.text_area("Article body", key="manual_body_input",
                                    height=200,
                                    placeholder="Paste article text here...")
        if st.button("Use this text", key="load_manual"):
            st.session_state["sel_title"]  = manual_title
            st.session_state["sel_body"]   = manual_body
            st.session_state["sel_source"] = "custom input"

    # ── Show what is currently loaded ──────────────────────
    if st.session_state.get("sel_body"):
        src   = st.session_state.get("sel_source", "unknown")
        title = st.session_state.get("sel_title", "")
        words = len(st.session_state["sel_body"].split())
        st.success(f"✅ Loaded: **{src}** · {words} words" + (f" · *{title[:60]}*" if title else ""))
        chosen_title = title
        chosen_body  = st.session_state["sel_body"]

    return chosen_title, chosen_body


# ── Initialise session state ───────────────────────────────
if "sel_title" not in st.session_state:
    # Pre-load the first sample so the app isn't blank on first visit
    first = SAMPLES["📈 Economy"][0]
    st.session_state["sel_title"]  = first["title"]
    st.session_state["sel_body"]   = first["body"]
    st.session_state["sel_source"] = first["label"]

selected_title, selected_body = render_resource_selector()

st.divider()

# ── Main layout ────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("📝 Article Input")

    title_input = st.text_input(
        "Article title (optional — improves sentence scoring)",
        value=selected_title,
        placeholder="e.g. Vietnam economy grows strongly in Q2...",
    )

    article_input = st.text_area(
        "Article body",
        value=selected_body,
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
