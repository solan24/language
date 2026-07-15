# 🇳🇵 English → Nepali Neural Machine Translator

A complete, production-ready English-to-Nepali translation system powered by a fine-tuned **NLLB-200** model, shipped with four interchangeable frontends (Streamlit, Gradio, FastAPI, Flask).

## 📋 Overview

This project wraps a fine-tuned NLLB-200 (~600M parameter) checkpoint — trained on 200,000 English–Nepali sentence pairs — in multiple ready-to-run interfaces:

| Interface | Purpose | Run Command |
|---|---|---|
| **Streamlit** (`app.py`) | Primary, full-featured UI | `streamlit run app.py` |
| **Gradio** (`gradio_app.py`) | Lightweight alternative UI | `python gradio_app.py` |
| **FastAPI** (`backend/app.py`) | REST API for integration | `uvicorn backend.app:app --reload` |
| **Flask** (`flask_app.py`) | REST API + HTML frontend | `python flask_app.py` |

### ✨ Features

- Deep blue / saffron / crimson Nepali-inspired theme
- Character & word counters, 128-token auto-truncation warning
- Adjustable temperature, max length, and beam count
- Translation history (last 20) with timestamps, stored per session
- Translation statistics (count, average latency)
- Dark/light mode toggle
- Language swap button
- Copy-to-clipboard
- 8 built-in example sentences across everyday domains
- Graceful error handling if the model fails to load
- Structured logging throughout

## 🗂️ Project Structure

```
EnglishNepali-Translator/
├── app.py                     # Streamlit app (primary)
├── gradio_app.py               # Gradio app
├── backend/
│   ├── __init__.py
│   └── app.py                  # FastAPI REST API
├── flask_app.py                # Flask backend
├── templates/
│   └── index.html              # Flask frontend
├── en_ne_nllb_finetuned/        # Your fine-tuned model (bring your own)
├── requirements.txt
├── README.md
├── .env
└── test_app.py                  # Standalone test script
```

## 🔧 Installation

1. **Clone / place the project folder** and `cd` into it:
   ```bash
   cd EnglishNepali-Translator
   ```

2. **Ensure your fine-tuned model** is present at `en_ne_nllb_finetuned/` (a standard Hugging Face `save_pretrained` checkpoint — `config.json`, `pytorch_model.bin`/`model.safetensors`, tokenizer files, etc.).

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running Each Version

### Streamlit (primary UI)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

### Gradio
```bash
python gradio_app.py
```
Opens at `http://localhost:7860`.

### FastAPI
```bash
uvicorn backend.app:app --reload
```
Interactive docs at `http://localhost:8000/docs`.

### Flask
```bash
python flask_app.py
```
Opens at `http://localhost:5000`.

## 📡 API Documentation (FastAPI / Flask)

### `POST /translate`

**Request body:**
```json
{
  "text": "I am going to Kathmandu tomorrow.",
  "max_length": 128,
  "temperature": 0.7,
  "num_beams": 4
}
```

**Response:**
```json
{
  "original_text": "I am going to Kathmandu tomorrow.",
  "translation": "म भोलि काठमाडौं जाँदैछु",
  "time_taken_ms": 182.4,
  "input_token_count": 9,
  "output_token_count": 11
}
```

### `GET /health`
Returns model load status and active device.

### `GET /info` (FastAPI only)
Returns model architecture, parameter count, and language codes.

### Example curl request
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Namaste, welcome to Nepal."}'
```

## 🧠 Model Information

- **Base architecture:** NLLB-200 (No Language Left Behind)
- **Parameters:** ~600M
- **Fine-tuning data:** 200,000 English–Nepali sentence pairs
- **Source language code:** `eng_Latn`
- **Target language code:** `npi_Deva`
- **Max sequence length:** 128 tokens (configurable up to 256/512 depending on interface)

## 🛠️ Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `OSError: en_ne_nllb_finetuned does not appear to be...` | Model folder missing/misnamed | Confirm the folder exists at the project root and contains a valid HF checkpoint |
| Very slow translation | Running on CPU | Use a CUDA-enabled GPU, or reduce `num_beams`/`max_length` |
| `CUDA out of memory` | GPU VRAM exceeded | Lower `max_length`/`num_beams`, or force CPU by setting `CUDA_VISIBLE_DEVICES=""` |
| Garbled or empty output | Wrong `forced_bos_token_id` | Verify tokenizer was loaded with `src_lang`/`tgt_lang` set correctly |
| `flask_cors` import error | Dependency not installed | `pip install flask-cors` |
| Streamlit shows blank sidebar stats | No translations run yet | This is expected until at least one translation completes |

## ☁️ Deployment Options

### Streamlit Community Cloud
1. Push this repo to GitHub (model files may need Git LFS if large).
2. Go to [share.streamlit.io](https://share.streamlit.io), link your repo, set `app.py` as the entry point.
3. Add `requirements.txt` — Streamlit Cloud installs it automatically.

### Hugging Face Spaces
1. Create a new Space (SDK: **Streamlit** or **Gradio**).
2. Upload `app.py` (or `gradio_app.py`), `requirements.txt`, and your model folder — or reference a model repo on the Hub instead of a local folder.
3. Spaces will build and launch automatically.

### Docker (FastAPI example)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
Build and run:
```bash
docker build -t en-ne-translator .
docker run -p 8000:8000 en-ne-translator
```

## 🧪 Testing

Run the standalone test script to sanity-check the model outside any web framework:
```bash
python test_app.py
```

---

Made with ❤️ for the Nepali language community | नेपाली भाषाका लागि निर्मित 🇳🇵
