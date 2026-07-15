"""
English to Nepali Neural Machine Translation - FastAPI Backend
अंग्रेजी देखि नेपाली मेसिन अनुवाद - REST API
"""

import time
import logging

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nllb-api")

MODEL_PATH = "en_ne_nllb_finetuned"
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"
DEFAULT_MAX_LENGTH = 128

app = FastAPI(
    title="English → Nepali Translation API",
    description="REST API for NLLB-200 fine-tuned English-to-Nepali translation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Singleton model manager
# --------------------------------------------------------------------------
class ModelManager:
    _instance = None
    tokenizer = None
    model = None
    device = None
    load_error = None
    loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self.loaded or self.load_error:
            return
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model on device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, src_lang=SRC_LANG, tgt_lang=TGT_LANG
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            self.model.to(self.device)
            self.model.config.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(TGT_LANG)
            self.model.eval()
            self.loaded = True
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.exception("Model loading failed.")
            self.load_error = str(e)


manager = ModelManager()


@app.on_event("startup")
def startup_event():
    manager.load()


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="English text to translate")
    max_length: int = Field(DEFAULT_MAX_LENGTH, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.1, le=1.0)
    num_beams: int = Field(4, ge=1, le=8)


class TranslationResponse(BaseModel):
    original_text: str
    translation: str
    time_taken_ms: float
    input_token_count: int
    output_token_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None


class InfoResponse(BaseModel):
    model_name: str
    architecture: str
    parameters: str
    source_language: str
    target_language: str
    device: str | None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if manager.loaded else "degraded",
        model_loaded=manager.loaded,
        device=manager.device,
    )


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        model_name="en_ne_nllb_finetuned",
        architecture="NLLB-200 (fine-tuned)",
        parameters="~600M",
        source_language=SRC_LANG,
        target_language=TGT_LANG,
        device=manager.device,
    )


@app.post("/translate", response_model=TranslationResponse)
def translate_endpoint(request: TranslationRequest):
    if manager.load_error:
        raise HTTPException(
            status_code=503,
            detail=f"Translation model unavailable: {manager.load_error}",
        )
    if not manager.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Please retry shortly.")

    try:
        tokenizer = manager.tokenizer
        model = manager.model
        device = manager.device

        inputs = tokenizer(
            request.text, return_tensors="pt", truncation=True, max_length=request.max_length
        )
        input_token_count = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                num_beams=request.num_beams,
                do_sample=True if request.temperature > 0.1 else False,
                early_stopping=True,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        output_token_count = generated.shape[1]

        return TranslationResponse(
            original_text=request.text,
            translation=translation,
            time_taken_ms=round(elapsed_ms, 2),
            input_token_count=int(input_token_count),
            output_token_count=int(output_token_count),
        )
    except Exception as e:
        logger.exception("Translation failed.")
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")


@app.get("/")
def root():
    return {
        "message": "English → Nepali Translation API 🇳🇵",
        "docs": "/docs",
        "endpoints": ["/translate", "/health", "/info"],
    }
