"""
English to Nepali Neural Machine Translation - Flask Backend
अंग्रेजी देखि नेपाली मेसिन अनुवाद - Flask एप
"""

import time
import logging

import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nllb-flask")

MODEL_PATH = "en_ne_nllb_finetuned"
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"
DEFAULT_MAX_LENGTH = 128

app = Flask(__name__)
CORS(app)

_tokenizer = None
_model = None
_device = None
_load_error = None


def load_model():
    global _tokenizer, _model, _device, _load_error
    if _model is not None or _load_error is not None:
        return
    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {_device}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        _model.to(_device)
        _model.config.forced_bos_token_id = _tokenizer.convert_tokens_to_ids(TGT_LANG)
        _model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Model loading failed.")
        _load_error = str(e)


# Load model at import time so the server is ready immediately.
load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    max_length = int(data.get("max_length", DEFAULT_MAX_LENGTH))
    temperature = float(data.get("temperature", 0.7))
    num_beams = int(data.get("num_beams", 4))

    if not text:
        return jsonify({"error": "Please provide non-empty 'text' to translate."}), 400

    if _load_error or _model is None:
        return jsonify({"error": f"Model unavailable: {_load_error}"}), 503

    try:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            generated = _model.generate(
                **inputs,
                forced_bos_token_id=_tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_new_tokens=max_length,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=True if temperature > 0.1 else False,
                early_stopping=True,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        translation = _tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return jsonify(
            {
                "original_text": text,
                "translation": translation,
                "time_taken_ms": round(elapsed_ms, 2),
            }
        )
    except Exception as e:
        logger.exception("Translation failed.")
        return jsonify({"error": f"Translation failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok" if _model is not None else "degraded", "device": _device})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
