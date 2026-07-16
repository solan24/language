# 🇳🇵 English → Nepali Neural Machine Translator

A complete, production-ready English-to-Nepali translation system powered by a **Parameter-Efficient Fine-Tuned (PEFT) NLLB-200** model, shipped with a decoupled Flask REST API and a lightweight, responsive HTML/JS frontend.

## 📋 Overview

This project utilizes Facebook's NLLB-200 (~600M parameter) foundational model, enhanced with custom **LoRA (Low-Rank Adaptation)** weights trained on English–Nepali sentence pairs. The inference engine is wrapped in a stateless REST API, ensuring strict separation between the machine learning backend and the client-side user interface.

### ✨ Features

- **Decoupled Architecture:** Strict Model-View-Controller (MVC) separation using a Python/Flask API and a Vanilla JS client.
- **Asynchronous Processing:** Non-blocking UI with loading states and dynamic DOM updates via the `fetch` API.
- **Responsive UI:** Deep blue / saffron / crimson Nepali-inspired theme with CSS Grid for mobile-to-desktop scaling.
- **Optimized Inference:** Utilizes `.safetensors` for secure, rapid weight loading and dynamic `cuda`/`cpu` hardware routing.
- **Stateless API:** Safe for concurrent network requests in a production environment.
- **Graceful Error Handling:** Structured logging and user-friendly error catching if the model fails to load or the network drops.

## 🗂️ Project Structure

```text
EnglishNepali-Translator/
│
├── en_ne_nllb_finetuned/       # Custom trained LoRA weights & Tokenizer
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── training_args.bin
│   └── README.md
│
├── templates/
│   └── index.html              # The Vanilla JS / UI frontend
│
├── .env                        # Environment variables
├── .gitignore                  # Git ignore rules
├── flask_app.py                # The main Flask REST API server
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🔧 Installation

1. **Clone or download the project folder** and navigate into it:
   ```bash
   cd EnglishNepali-Translator
   ```

2. **Ensure your trained LoRA weights** and tokenizer files are present in the `en_ne_nllb_finetuned/` directory.

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

Start the Flask REST API and web server:
```bash
python flask_app.py
```
*The application UI will be available at `http://localhost:5000`.*

## 📡 API Documentation

### `POST /translate`

**Request body:**
```json
{
  "text": "I am going to Kathmandu tomorrow."
}
```

**Response:**
```json
{
  "translation": "म भोलि काठमाडौं जाँदैछु"
}
```

### Example cURL request
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Namaste, welcome to Nepal."}'
```

## 🧠 Model Information

- **Base architecture:** NLLB-200 (No Language Left Behind)
- **Parameters:** ~600M (Base) + Custom LoRA delta weights
- **Fine-tuning Strategy:** PEFT (Parameter-Efficient Fine-Tuning) to prevent catastrophic forgetting.
- **Source language code:** `eng_Latn`
- **Target language code:** `npi_Deva`
- **Weight Format:** `.safetensors`

## 🛠️ Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `OSError: en_ne_nllb_finetuned does not appear to be...` | Adapter folder missing/misnamed | Confirm the folder exists at the project root and contains `adapter_model.safetensors`. |
| Very slow translation | Running on CPU | Ensure PyTorch is compiled with CUDA and an NVIDIA GPU is available. |
| UI freezes on "Translating..." | Backend disconnected | Check the terminal running `flask_app.py` for Python traceback errors. |
| Garbled or empty output | Wrong language IDs | Verify the tokenizer is passing `src_lang="eng_Latn"` and target ID `npi_Deva`. |

## ☁️ Deployment Options

### Docker
To deploy this stateless API in a containerized environment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "flask_app.py"]
```
Build and run:
```bash
docker build -t en-ne-translator .
docker run -p 5000:5000 en-ne-translator
```

---
**Author:** Solan Gurung  
**Program:** B.Tech Artificial Intelligence and Machine Learning
