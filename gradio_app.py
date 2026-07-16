"""
English to Nepali Neural Machine Translation - Gradio App
अंग्रेजी देखि नेपाली मेसिन अनुवाद - Gradio इन्टरफेस
"""

import time
import logging

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel  # <-- NEW: Required for loading adapter weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nllb-gradio")

# NEW: Define the original foundation model used during training
BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"
MODEL_PATH = "en_ne_nllb_finetuned"
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"
MAX_LENGTH = 128

EXAMPLES = [
    "The student reads a book.",
    "I am going to Kathmandu tomorrow.",
    "How are you doing today?",
    "Namaste, welcome to Nepal.",
    "Momo is delicious.",
    "I want to go to Pokhara.",
    "It is raining heavily today.",
    "I have been studying Nepali for three months.",
]

_tokenizer = None
_model = None
_device = None
_load_error = None


def load_model():
    """Load and cache model/tokenizer (singleton)."""
    global _tokenizer, _model, _device, _load_error
    if _model is not None or _load_error is not None:
        return _tokenizer, _model, _device, _load_error
    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {_device}")
        
        # 1. Load Tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        
        # 2. Load the Base Model first
        logger.info(f"Loading base model: {BASE_MODEL_NAME}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
        
        # 3. Apply your fine-tuned PEFT adapter weights on top of the base model
        logger.info(f"Applying PEFT adapter from: {MODEL_PATH}")
        _model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        _model.to(_device)
        _model.config.forced_bos_token_id = _tokenizer.convert_tokens_to_ids(TGT_LANG)
        _model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Model loading failed.")
        _load_error = str(e)
    return _tokenizer, _model, _device, _load_error


def translate(text, tokenizer, model, device, max_length=128, temperature=0.7, num_beams=4):
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


def translate_ui(text, temperature, max_length, num_beams):
    if not text or not text.strip():
        return "⚠️ Please enter text to translate.", ""

    tokenizer, model, device, error = load_model()
    if error or model is None:
        return f"❌ Model failed to load: {error}", ""

    try:
        start = time.perf_counter()
        result = translate(
            text, tokenizer, model, device,
            max_length=int(max_length), temperature=temperature, num_beams=int(num_beams),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, f"⏱️ {elapsed_ms:.0f} ms"
    except Exception as e:
        logger.exception("Translation failed.")
        return f"❌ Translation failed: {e}", ""


with gr.Blocks(title="English → Nepali Translator 🇳🇵", theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown(
        """
        # 🇳🇵 English → Nepali Neural Machine Translator
        ### अंग्रेजी देखि नेपाली मेसिन अनुवाद
        Powered by a fine-tuned NLLB-200 model.
        """
    )

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="English Text | अंग्रेजी पाठ",
                placeholder="Type English text here...",
                lines=6,
            )
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature")
                max_length = gr.Slider(50, 256, value=128, step=1, label="Max Length")
                num_beams = gr.Slider(1, 5, value=4, step=1, label="Num Beams")
            translate_btn = gr.Button("🌐 Translate | अनुवाद गर्नुहोस्", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(label="Nepali Translation | नेपाली अनुवाद", lines=6, interactive=False)
            time_box = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=input_box, label="💡 Example Sentences | उदाहरणहरू")

    translate_btn.click(
        fn=translate_ui,
        inputs=[input_box, temperature, max_length, num_beams],
        outputs=[output_box, time_box],
    )
    input_box.submit(
        fn=translate_ui,
        inputs=[input_box, temperature, max_length, num_beams],
        outputs=[output_box, time_box],
    )

    gr.Markdown("---\nMade with ❤️ using NLLB-200 & Gradio | नेपाली भाषाका लागि निर्मित")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="orange"), share=True)