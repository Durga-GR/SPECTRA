import streamlit as st
import base64
import tempfile
from text_detector import analyze_text
from image_detector import analyze_image
from video_detector import analyze_video

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Spectra – AI Multi-Modal Authenticity Analyzer",
    layout="wide"
)

# =========================================================
# BACKGROUND + STYLING
# =========================================================
def add_bg():
    with open("space.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    /* Background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Glass Card */
    .glass {{
        background: rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 20px;
        backdrop-filter: blur(15px);
        box-shadow: 0 0 40px rgba(0,255,255,0.4);
    }}

    /* Title */
    .title {{
        font-size: 50px;
        font-weight: 800;
        color: white;
        text-align: center;
        text-shadow: 0 0 20px #00ffff;
    }}

    .subtitle {{
        font-size: 20px;
        color: #f3f4f6;
        text-align: center;
        margin-bottom: 40px;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg,#7c3aed,#06b6d4);
        color: white;
        border-radius: 30px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 0 20px #06b6d4;
    }}

    /* Text Area */
    textarea {{
        color: black !important;
    }}

    /* Result Box */
    .result-box {{
        margin-top: 20px;
        padding: 20px;
        border-radius: 15px;
        background: rgba(0,0,0,0.6);
        color: white;
        font-size: 20px;
        text-align: center;
    }}

    .detail-box {{
        margin-top: 10px;
        padding: 15px;
        border-radius: 12px;
        background: rgba(255,255,255,0.07);
        color: #d1fae5;
        font-size: 14px;
        font-family: monospace;
    }}

    </style>
    """, unsafe_allow_html=True)

add_bg()

# =========================================================
# HEADER SECTION
# =========================================================
st.markdown('<div class="title">SPECTRA – AI MULTI-MODAL AUTHENTICITY ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">DETECTING REALITY IN THE AGE OF ARTIFICIAL INTELLIGENCE.</div>', unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# HELPER: render result
# =========================================================
def render_result(result):
    label = result["label"]
    score = result["score"]
    pct = int(score * 100)

    # Color the label based on AI vs human
    color = "#00ffcc" if "Human" in label else "#ff4d6d"

    st.markdown(f"""
    <div class="result-box">
        <span style="color:{color}; font-size:24px; font-weight:bold;">{label}</span><br><br>
        Confidence: <strong>{pct}%</strong>
    </div>
    """, unsafe_allow_html=True)

    st.progress(pct)

    # Show details if available
    if "details" in result and result["details"]:
        details_html = "".join(
            f"<div>• {k.replace('_', ' ').title()}: <strong>{v}</strong></div>"
            for k, v in result["details"].items()
        )
        st.markdown(f'<div class="detail-box">🔍 Analysis Breakdown:<br>{details_html}</div>', unsafe_allow_html=True)


# =========================================================
# TABS SECTION
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["📝 Text Detection", "🖼 Image Detection", "🎥 Video Detection"]
)

# =========================================================
# TEXT DETECTION
# =========================================================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    text_input = st.text_area("Paste your text here...", height=200)

    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing Text..."):
                result = analyze_text(text_input)
            render_result(result)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# IMAGE DETECTION
# =========================================================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing Image..."):
                result = analyze_image(uploaded_image)
            render_result(result)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# VIDEO DETECTION
# =========================================================
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        temp_file.flush()

        if st.button("Analyze Video"):
            with st.spinner("Analyzing Video..."):
                result = analyze_video(temp_file.name)
            render_result(result)

    st.markdown('</div>', unsafe_allow_html=True)
