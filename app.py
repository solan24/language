"""
English to Nepali Neural Machine Translation - Streamlit App
अंग्रेजी देखि नेपाली मेसिन अनुवाद एप
Powered by fine-tuned NLLB-200 model.
"""

import time
import logging
from datetime import datetime

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nllb-translator")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODEL_PATH = "en_ne_nllb_finetuned"
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"
MAX_LENGTH = 128

DEEP_BLUE = "#1a237e"
SAFFRON = "#ff6f00"
CRIMSON = "#c62828"

EXAMPLES = [
    ("The student reads a book.", "विद्यार्थीले किताब पढ्छ"),
    ("I am going to Kathmandu tomorrow.", "म भोलि काठमाडौं जाँदैछु"),
    ("How are you doing today?", "आज तपाईंलाई कस्तो छ?"),
    ("Namaste, welcome to Nepal.", "नमस्ते, नेपालमा स्वागत छ।"),
    ("Momo is delicious.", "मोमो स्वादिष्ट छ।"),
    ("I want to go to Pokhara.", "म पोखरा जान चाहन्छु।"),
    ("It is raining heavily today.", "आज धेरै पानी परिरहेको छ।"),
    ("I have been studying Nepali for three months.", "मैले तीन महिनादेखि नेपाली पढिरहेको छु।"),
]

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="English → Nepali Translator | अनुवादक",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {src, tgt, ms, time}
if "total_translations" not in st.session_state:
    st.session_state.total_translations = 0
if "total_time_ms" not in st.session_state:
    st.session_state.total_time_ms = 0.0
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "swapped" not in st.session_state:
    st.session_state.swapped = False


# --------------------------------------------------------------------------
# Theming (CSS)
# --------------------------------------------------------------------------
def inject_css(dark: bool):
    bg = "#0e1117" if dark else "#ffffff"
    text_color = "#f5f5f5" if dark else "#1a1a1a"
    card_bg = "#1c2030" if dark else "#f7f7fb"

    st.markdown(
        f"""
        <style>
        .main {{
            background-color: {bg};
            color: {text_color};
        }}
        .translator-header {{
            background: linear-gradient(90deg, {DEEP_BLUE}, {CRIMSON});
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
        }}
        .translator-header h1 {{
            margin: 0;
            font-size: 1.8rem;
        }}
        .translator-header p {{
            margin: 0.3rem 0 0 0;
            opacity: 0.9;
        }}
        .stButton>button {{
            background-color: {SAFFRON};
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
        }}
        .stButton>button:hover {{
            background-color: {CRIMSON};
            color: white;
        }}
        .translation-card {{
            background-color: {card_bg};
            border-left: 5px solid {SAFFRON};
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin-top: 0.5rem;
        }}
        .history-item {{
            background-color: {card_bg};
            border-radius: 6px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.4rem;
            font-size: 0.85rem;
        }}
        .footer {{
            text-align: center;
            opacity: 0.6;
            margin-top: 3rem;
            font-size: 0.85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(st.session_state.dark_mode)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load tokenizer & model once, cached across reruns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from '{MODEL_PATH}' on device: {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        model.to(device)
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
        model.eval()
        logger.info("Model loaded successfully.")
        return tokenizer, model, device, None
    except Exception as e:
        logger.exception("Model loading failed.")
        return None, None, None, str(e)


def translate(text, tokenizer, model, device, max_length=128, temperature=0.7, num_beams=4):
    """Run translation and return the Nepali text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_new_tokens=max_length,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=True if temperature > 0.1 else False,
            early_stopping=True,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


tokenizer, model, device, load_error = load_model()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings | सेटिङ्स")

    dark_toggle = st.toggle("🌙 Dark Mode | डार्क मोड", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.divider()

    st.markdown("### 🧠 Model Information | मोडेल जानकारी")
    st.markdown(
        f"""
        - **Architecture:** NLLB-200
        - **Parameters:** ~600M
        - **Fine-tuned on:** 200,000 sentence pairs
        - **Source lang:** English (`{SRC_LANG}`)
        - **Target lang:** Nepali (`{TGT_LANG}`)
        - **Device:** `{device if device else 'unavailable'}`
        """
    )
    if load_error:
        st.error(f"⚠️ Model failed to load: {load_error}")

    st.divider()

    st.markdown("### 📊 Translation Statistics | तथ्याङ्क")
    avg_time = (
        st.session_state.total_time_ms / st.session_state.total_translations
        if st.session_state.total_translations > 0
        else 0
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Total", st.session_state.total_translations)
    col_b.metric("Avg. Time", f"{avg_time:.0f} ms")

    st.divider()

    st.markdown("### 🕘 History | इतिहास (last 20)")
    if st.button("🗑️ Clear History | इतिहास मेटाउनुहोस्"):
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        for item in reversed(st.session_state.history[-20:]):
            st.markdown(
                f"""<div class="history-item">
                <b>{item['time']}</b><br>
                <span>🇬🇧 {item['src']}</span><br>
                <span>🇳🇵 {item['tgt']}</span><br>
                <i>{item['ms']:.0f} ms</i>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No translations yet.")

    st.divider()

    with st.expander("🛠️ Advanced Settings | उन्नत सेटिङ्स"):
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
        max_length = st.slider("Max Length (tokens)", 50, 256, 128, 1)
        num_beams = st.slider("Num Beams", 1, 5, 4, 1)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="translator-header">
        <h1>🇳🇵 English → Nepali Translator | अंग्रेजी–नेपाली अनुवादक</h1>
        <p>Powered by a fine-tuned NLLB-200 Neural Machine Translation model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Example sentences
# --------------------------------------------------------------------------
with st.expander("💡 Example Sentences | उदाहरण वाक्यहरू"):
    ex_cols = st.columns(2)
    for i, (en, ne) in enumerate(EXAMPLES):
        col = ex_cols[i % 2]
        if col.button(f"🇬🇧 {en}\n🇳🇵 {ne}", key=f"example_{i}", use_container_width=True):
            st.session_state.input_text = en
            st.rerun()

# --------------------------------------------------------------------------
# Language swap + input area
# --------------------------------------------------------------------------
lang_col1, swap_col, lang_col2 = st.columns([5, 1, 5])

if st.session_state.swapped:
    src_label, tgt_label = "🇳🇵 Nepali (नेपाली)", "🇬🇧 English"
else:
    src_label, tgt_label = "🇬🇧 English", "🇳🇵 Nepali (नेपाली)"

with lang_col1:
    st.markdown(f"**{src_label}**")

with swap_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄", help="Swap languages | भाषा साट्नुहोस्"):
        st.session_state.swapped = not st.session_state.swapped
        st.rerun()

with lang_col2:
    st.markdown(f"**{tgt_label}**")

if st.session_state.swapped:
    st.info(
        "ℹ️ This model is fine-tuned for English → Nepali translation. "
        "Reverse translation quality may vary. | यो मोडेल अंग्रेजी देखि नेपालीका लागि तालिम गरिएको हो।"
    )

input_text = st.text_area(
    "Enter text to translate | अनुवाद गर्न पाठ लेख्नुहोस्",
    value=st.session_state.input_text,
    height=150,
    key="text_input_area",
    placeholder="Type or paste English text here...",
)

word_count = len(input_text.split())
char_count = len(input_text)
token_warning = char_count > 0 and len(tokenizer(input_text)["input_ids"]) > MAX_LENGTH if tokenizer else False

count_col1, count_col2, count_col3 = st.columns(3)
count_col1.caption(f"📝 Words: {word_count}")
count_col2.caption(f"🔤 Characters: {char_count}")
if token_warning:
    count_col3.warning(f"⚠️ Exceeds {MAX_LENGTH} tokens — will be truncated.")

translate_col, _ = st.columns([1, 4])
do_translate = translate_col.button("🌐 Translate | अनुवाद गर्नुहोस्", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Translation execution
# --------------------------------------------------------------------------
if do_translate:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to translate. | कृपया अनुवाद गर्न पाठ लेख्नुहोस्।")
    elif load_error or tokenizer is None or model is None:
        st.error(
            "❌ The translation model is not available. Please check that the model folder "
            f"'{MODEL_PATH}' exists and is a valid NLLB checkpoint."
        )
    else:
        with st.spinner("Translating... | अनुवाद गर्दै..."):
            try:
                start = time.perf_counter()
                result = translate(
                    input_text,
                    tokenizer,
                    model,
                    device,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                st.session_state.total_translations += 1
                st.session_state.total_time_ms += elapsed_ms
                st.session_state.history.append(
                    {
                        "src": input_text,
                        "tgt": result,
                        "ms": elapsed_ms,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                )

                st.markdown("### ✅ Translation | अनुवाद")
                st.markdown(
                    f"""<div class="translation-card">{result}</div>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"⏱️ Translated in {elapsed_ms:.0f} ms")

                # Copy-to-clipboard via a small HTML/JS snippet
                st.components.v1.html(
                    f"""
                    <button onclick="navigator.clipboard.writeText(`{result}`)"
                        style="background-color:{SAFFRON};color:white;border:none;
                        padding:8px 14px;border-radius:6px;cursor:pointer;font-weight:600;">
                        📋 Copy to Clipboard
                    </button>
                    """,
                    height=50,
                )
            except Exception as e:
                logger.exception("Translation failed.")
                st.error(f"❌ Translation failed: {e}")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using NLLB-200 & Streamlit | नेपाली भाषाका लागि निर्मित<br>
        🇳🇵 English–Nepali Neural Machine Translation Project
    </div>
    """,
    unsafe_allow_html=True,
)
