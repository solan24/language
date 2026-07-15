"""
Basic test script for the English → Nepali translation model.
Run with: python test_app.py
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "en_ne_nllb_finetuned"
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"
MAX_LENGTH = 128

TEST_SENTENCES = [
    "The student reads a book.",
    "I am going to Kathmandu tomorrow.",
    "How are you doing today?",
    "Namaste, welcome to Nepal.",
    "Momo is delicious.",
]


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model.eval()
    return tokenizer, model, device


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


def main():
    try:
        tokenizer, model, device = load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\nRunning translation tests...\n")
    for sentence in TEST_SENTENCES:
        start = time.perf_counter()
        try:
            result = translate(sentence, tokenizer, model, device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"EN: {sentence}")
            print(f"NE: {result}")
            print(f"⏱️  {elapsed_ms:.0f} ms\n")
        except Exception as e:
            print(f"❌ Translation failed for '{sentence}': {e}\n")


if __name__ == "__main__":
    main()
