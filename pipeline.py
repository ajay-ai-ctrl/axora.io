"""
SmartKhet — NLP & Voice Processing Pipeline
=============================================
Components:
  1. Speech-to-Text    : OpenAI Whisper (fine-tuned on Indic agri corpus)
  2. Language Detection: langdetect + custom ISO-639 mapping
  3. Intent Classifier : IndicBERT fine-tuned on 12 agricultural intents
  4. Entity Extractor  : spaCy + custom agri NER model
  5. Query Router      : Maps (intent, entities) → backend service call
  6. Response Renderer : Formats response back to farmer's language

Supported Languages: hi, mr, pa, bn, te, ta, gu, kn, or, ml + dialects
Author: Axora / SmartKhet ML Team
"""

import os
import json
import logging
import numpy as np
import torch
import mlflow
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import whisper
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline as hf_pipeline,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from torch.nn.functional import softmax

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

INDICBERT_MODEL = "ai4bharat/indic-bert"
WHISPER_MODEL_SIZE = "small"  # 244M params, good Hindi WER

# 12 agricultural intents
INTENTS = [
    "crop_recommendation",      # "मुझे क्या बोना चाहिए?"
    "disease_detection",        # "मेरी फसल में कीड़े लग गए"
    "fertilizer_query",         # "कितनी खाद डालूँ?"
    "irrigation_query",         # "पानी कब देना है?"
    "market_price_query",       # "आज गेहूँ का भाव क्या है?"
    "sell_advisory",            # "अभी बेचूँ या रुकूँ?"
    "weather_query",            # "कल बारिश होगी?"
    "pest_query",               # "टिड्डी से कैसे बचाऊँ?"
    "government_scheme_query",  # "PM किसान का पैसा कब आएगा?"
    "soil_query",               # "मिट्टी की जाँच कैसे करें?"
    "harvesting_query",         # "फसल कब काटें?"
    "general_agri_query",       # Catch-all
]

# Named entity types for agri domain
NER_LABELS = [
    "O",
    "B-CROP", "I-CROP",           # फसल नाम
    "B-DISEASE", "I-DISEASE",     # बीमारी
    "B-CHEMICAL", "I-CHEMICAL",   # दवाई / खाद
    "B-LOCATION", "I-LOCATION",   # जिला / गाँव
    "B-QUANTITY", "I-QUANTITY",   # मात्रा
    "B-TIME", "I-TIME",           # समय / मौसम
]

LANG_TO_ISO = {
    "hi": "Hindi", "mr": "Marathi", "pa": "Punjabi",
    "bn": "Bengali", "te": "Telugu", "ta": "Tamil",
    "gu": "Gujarati", "kn": "Kannada", "or": "Odia",
    "ml": "Malayalam", "en": "English",
}


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_name: str
    confidence: float
    segments: list = field(default_factory=list)

@dataclass
class IntentResult:
    intent: str
    confidence: float
    all_scores: dict = field(default_factory=dict)

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

@dataclass
class ParsedQuery:
    raw_text: str
    language: str
    intent: IntentResult
    entities: list[Entity]
    crop: Optional[str] = None
    disease: Optional[str] = None
    location: Optional[str] = None
    quantity: Optional[str] = None
    time_ref: Optional[str] = None


# ── Whisper STT ────────────────────────────────────────────────────────────────

class IndicSTT:
    """
    OpenAI Whisper fine-tuned for Indian agricultural speech.
    Handles code-switching (Hindi + English mixing), rural accents,
    background noise (farm environment).
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE,
                 fine_tuned_path: Optional[str] = None):
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            log.info(f"Loading fine-tuned Whisper from {fine_tuned_path}")
            self.model = whisper.load_model(fine_tuned_path)
        else:
            log.info(f"Loading base Whisper-{model_size}")
            self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str,
                   language_hint: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio file. If language_hint given (e.g., "hi"),
        skip auto-detection for faster inference.
        """
        options = {
            "task": "transcribe",
            "fp16": torch.cuda.is_available(),
            "verbose": False,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "word_timestamps": True,
        }
        if language_hint:
            options["language"] = language_hint

        result = self.model.transcribe(audio_path, **options)

        # Compute average log probability as confidence proxy
        avg_logprob = np.mean([s.get("avg_logprob", -1.0)
                               for s in result.get("segments", [])])
        confidence = float(np.exp(avg_logprob)) if avg_logprob > -5 else 0.1

        detected_lang = result.get("language", "hi")

        return TranscriptionResult(
            text=result["text"].strip(),
            language=detected_lang,
            language_name=LANG_TO_ISO.get(detected_lang, "Unknown"),
            confidence=confidence,
            segments=result.get("segments", []),
        )

    def transcribe_bytes(self, audio_bytes: bytes,
                         language_hint: Optional[str] = None) -> TranscriptionResult:
        """Transcribe from raw audio bytes (wav format)."""
        import io
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe(tmp_path, language_hint)
        finally:
            os.unlink(tmp_path)


# ── Intent Classifier Fine-tuning ─────────────────────────────────────────────

def fine_tune_intent_classifier(
    train_data: list[dict],  # [{"text": ..., "intent": ...}, ...]
    output_dir: str = "models/nlp/intent_classifier/",
    experiment_name: str = "smartkhet-intent-classifier",
):
    """
    Fine-tune IndicBERT for agricultural intent classification.
    Input data should be multilingual (mix of Hindi + regional languages).
    """
    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    intent2id = {intent: i for i, intent in enumerate(INTENTS)}
    id2intent = {v: k for k, v in intent2id.items()}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL)

    # Build HuggingFace Dataset
    df_data = [{"text": d["text"], "label": intent2id[d["intent"]]}
               for d in train_data if d["intent"] in intent2id]

    split_idx = int(len(df_data) * 0.85)
    raw_train = Dataset.from_list(df_data[:split_idx])
    raw_val = Dataset.from_list(df_data[split_idx:])

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128,
                         padding=False)

    train_ds = raw_train.map(tokenize, batched=True)
    val_ds = raw_val.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        INDICBERT_MODEL,
        num_labels=len(INTENTS),
        id2label=id2intent,
        label2id=intent2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import f1_score, accuracy_score
        return {
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "accuracy": accuracy_score(labels, preds),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        fp16=torch.cuda.is_available(),
        report_to="mlflow",
        logging_steps=50,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="indicbert_intent_ft"):
        trainer.train()
        eval_results = trainer.evaluate()
        log.info(f"Intent Classifier Eval: {eval_results}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        json.dump(id2intent, open(os.path.join(output_dir, "id2intent.json"), "w"),
                  ensure_ascii=False)

    log.info(f"✅ Intent classifier saved: {output_dir}")
    return model, tokenizer


# ── Inference Pipeline ─────────────────────────────────────────────────────────

class AgroNLPPipeline:
    """
    End-to-end NLP pipeline:
    Audio → STT → Intent + Entity → Structured ParsedQuery
    Also accepts raw text input (for WhatsApp text messages).
    """

    def __init__(
        self,
        intent_model_dir: str = "models/nlp/intent_classifier/",
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        fine_tuned_whisper: Optional[str] = None,
    ):
        log.info("Initializing AgroNLPPipeline...")

        # STT
        self.stt = IndicSTT(
            model_size=whisper_model_size,
            fine_tuned_path=fine_tuned_whisper,
        )

        # Intent classifier
        self.tokenizer = AutoTokenizer.from_pretrained(intent_model_dir)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            intent_model_dir
        )
        self.intent_model.eval()
        id2intent_path = os.path.join(intent_model_dir, "id2intent.json")
        if os.path.exists(id2intent_path):
            with open(id2intent_path) as f:
                self.id2intent = {int(k): v for k, v in json.load(f).items()}
        else:
            self.id2intent = {i: intent for i, intent in enumerate(INTENTS)}

        # Entity extraction via spaCy (load if model exists, else use regex fallback)
        self._load_ner()
        log.info("AgroNLPPipeline ready ✅")

    def _load_ner(self):
        """Load spaCy agri NER model or fall back to rule-based extraction."""
        try:
            import spacy
            self.nlp = spacy.load("models/nlp/agri_ner/")
            self.use_spacy_ner = True
            log.info("spaCy agri NER model loaded")
        except Exception:
            log.warning("spaCy NER model not found — using rule-based entity extraction")
            self.use_spacy_ner = False

    def process_audio(self, audio_path_or_bytes,
                      language_hint: Optional[str] = None) -> ParsedQuery:
        """Full pipeline from audio input."""
        if isinstance(audio_path_or_bytes, bytes):
            transcription = self.stt.transcribe_bytes(audio_path_or_bytes, language_hint)
        else:
            transcription = self.stt.transcribe(audio_path_or_bytes, language_hint)

        return self.process_text(transcription.text, language=transcription.language)

    def process_text(self, text: str, language: str = "hi") -> ParsedQuery:
        """Process raw text input (from WhatsApp or SMS)."""
        # Classify intent
        intent_result = self._classify_intent(text)

        # Extract entities
        entities = self._extract_entities(text)

        # Pull key entities to top-level fields
        query = ParsedQuery(
            raw_text=text,
            language=language,
            intent=intent_result,
            entities=entities,
        )

        for ent in entities:
            if ent.label == "CROP" and query.crop is None:
                query.crop = ent.text
            elif ent.label == "DISEASE" and query.disease is None:
                query.disease = ent.text
            elif ent.label == "LOCATION" and query.location is None:
                query.location = ent.text
            elif ent.label == "QUANTITY" and query.quantity is None:
                query.quantity = ent.text
            elif ent.label == "TIME" and query.time_ref is None:
                query.time_ref = ent.text

        return query

    def _classify_intent(self, text: str) -> IntentResult:
        """Run IndicBERT intent classification."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self.intent_model(**inputs).logits
            probs = softmax(logits, dim=-1)[0]

        top_idx = int(probs.argmax().item())
        all_scores = {self.id2intent[i]: float(probs[i])
                      for i in range(len(self.id2intent))}

        return IntentResult(
            intent=self.id2intent[top_idx],
            confidence=float(probs[top_idx]),
            all_scores=all_scores,
        )

    def _extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities using spaCy or rule-based fallback."""
        if self.use_spacy_ner:
            return self._spacy_extract(text)
        return self._rule_based_extract(text)

    def _spacy_extract(self, text: str) -> list[Entity]:
        doc = self.nlp(text)
        return [
            Entity(text=ent.text, label=ent.label_,
                   start=ent.start_char, end=ent.end_char)
            for ent in doc.ents
        ]

    def _rule_based_extract(self, text: str) -> list[Entity]:
        """Keyword-based entity extraction as fallback."""
        import re
        entities = []
        text_lower = text.lower()

        crop_keywords = {
            "rice": ["चावल", "धान", "rice"],
            "wheat": ["गेहूँ", "गेहू", "wheat"],
            "maize": ["मक्का", "maize", "corn"],
            "cotton": ["कपास", "cotton"],
            "sugarcane": ["गन्ना", "sugarcane"],
            "soybean": ["सोयाबीन", "soybean"],
        }
        for crop, keywords in crop_keywords.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    idx = text_lower.index(kw.lower())
                    entities.append(Entity(text=kw, label="CROP",
                                           start=idx, end=idx + len(kw)))
                    break

        # Quantity pattern: e.g., "50 किलो", "2 एकड़"
        qty_pattern = r"\d+\s*(?:किलो|kg|एकड़|acre|hectare|quintal|qtl|बोरी)"
        for m in re.finditer(qty_pattern, text, re.IGNORECASE):
            entities.append(Entity(text=m.group(), label="QUANTITY",
                                   start=m.start(), end=m.end()))

        return entities

    def to_route_payload(self, query: ParsedQuery) -> dict:
        """
        Convert parsed query into a routing payload for the API gateway.
        Maps intent → backend service endpoint.
        """
        INTENT_TO_SERVICE = {
            "crop_recommendation":   "/api/v1/advisory/crop",
            "disease_detection":     "/api/v1/disease/analyze-text",
            "fertilizer_query":      "/api/v1/advisory/fertilizer",
            "irrigation_query":      "/api/v1/advisory/irrigation",
            "market_price_query":    "/api/v1/market/price",
            "sell_advisory":         "/api/v1/market/sell-signal",
            "weather_query":         "/api/v1/weather/advisory",
            "pest_query":            "/api/v1/advisory/pest",
            "government_scheme_query": "/api/v1/schemes/query",
            "soil_query":            "/api/v1/advisory/soil",
            "harvesting_query":      "/api/v1/advisory/harvest",
            "general_agri_query":    "/api/v1/advisory/general",
        }

        return {
            "target_service": INTENT_TO_SERVICE[query.intent.intent],
            "intent": query.intent.intent,
            "confidence": query.intent.confidence,
            "language": query.language,
            "crop": query.crop,
            "disease": query.disease,
            "location": query.location,
            "raw_text": query.raw_text,
            "entities": [
                {"text": e.text, "label": e.label} for e in query.entities
            ],
        }


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune SmartKhet NLP models")
    parser.add_argument("--mode", choices=["train_intent", "test_pipeline"],
                        required=True)
    parser.add_argument("--data", type=str, help="Path to training JSON")
    parser.add_argument("--audio", type=str, help="Audio file for pipeline test")
    args = parser.parse_args()

    if args.mode == "train_intent":
        with open(args.data) as f:
            train_data = json.load(f)
        fine_tune_intent_classifier(train_data)

    elif args.mode == "test_pipeline":
        pipe = AgroNLPPipeline()
        if args.audio:
            result = pipe.process_audio(args.audio)
        else:
            # Test with sample Hindi text
            result = pipe.process_text("मेरे धान में पीली पत्तियाँ आ रही हैं, क्या करूँ?")
        print(json.dumps(pipe.to_route_payload(result), ensure_ascii=False, indent=2))
