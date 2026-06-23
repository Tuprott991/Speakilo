"""
Trạm 2: Bilingual Machine Translation — MarianMT (VI ↔ EN)
===========================================================
Focused exclusively on Vietnamese ↔ English translation using
Helsinki-NLP's MarianMT models. Supports fine-tuned checkpoints
for industrial/engineering terminology accuracy.

References:
  Helsinki-NLP/opus-mt-vi-en — University of Helsinki (Apache 2.0)
  Helsinki-NLP/opus-mt-en-vi — University of Helsinki (Apache 2.0)
  https://huggingface.co/Helsinki-NLP
"""

import time
import queue
import re
import os
import torch

# 🛠️ HOTFIX: PyTorch 2.6.0 removed float8_e8m0fnu which crashes Transformers
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", getattr(torch, "float8_e4m3fn", None))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


# ── Industrial terminology override dictionary ────────────────────────────────
# Loaded from data/industrial_terms.csv at runtime if available.
# These are direct substitutions BEFORE translation to improve accuracy.
_TERM_OVERRIDE_VI_EN: dict[str, str] = {}
_TERM_OVERRIDE_EN_VI: dict[str, str] = {}
_POST_TRANSLATE_FIX: dict[str, str] = {}   # EN→EN corrections for known envit5 errors


def load_terminology(csv_path: str):
    """Load industrial term pairs from CSV into override dictionaries."""
    import csv
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vi = row["vi_text"].strip().strip('"').lower()
                en = row["en_text"].strip().strip('"').lower()
                domain = row.get("domain", "").strip().strip('"').lower()
                if vi and en:
                    if domain == "post-translation-fix":
                        # These are EN→EN corrections (wrong model output → correct term)
                        _POST_TRANSLATE_FIX[vi] = en
                    else:
                        _TERM_OVERRIDE_VI_EN[vi] = en
                        _TERM_OVERRIDE_EN_VI[en] = vi
        n_terms = len(_TERM_OVERRIDE_VI_EN)
        n_fixes = len(_POST_TRANSLATE_FIX)
        print(f"[MT] ✅ Loaded {n_terms} technical term pairs + {n_fixes} post-translation fixes.")
    except FileNotFoundError:
        print(f"[MT] ⚠ Terminology file not found: {csv_path}")
    except Exception as e:
        print(f"[MT] ⚠ Could not load terminology: {e}")


def _apply_terminology(text: str, override: dict[str, str]) -> str:
    """Replace known technical terms. Uses word boundaries to avoid partial matches."""
    result = text
    for src, tgt in sorted(override.items(), key=lambda x: -len(x[0])):
        # Use word boundaries (\b) to prevent 'roof' matching inside 'rooftop'
        pattern = re.compile(r'\b' + re.escape(src) + r'\b', re.IGNORECASE)
        result = pattern.sub(tgt, result)
    return result


# Vietnamese discourse markers used as natural split points
_VI_SPLIT_MARKERS = [
    " nên ", " vậy chứ ", " vậy ", " mà ", " nhưng ", " còn ",
    " rồi ", " thì ", " để ", " sau đó ", " xong ", " luôn ",
    " chứ ", " nha ", " hả ", " hả? ", " à ", " nữa ",
]


def _chunk_text(text: str, max_words: int = 15) -> list[str]:
    """
    Split long text into manageable chunks for MT to reduce streaming latency.
    Prioritizes splitting at punctuation boundaries (. ? ! ;).
    Falls back to discourse markers if a sentence is too long.
    """
    import re
    # Split by punctuation, keeping the punctuation attached
    raw_sentences = re.split(r'(?<=[.?!;])\s+', text.strip())
    
    chunks = []
    for sentence in raw_sentences:
        if not sentence.strip():
            continue
            
        words = sentence.split()
        # If sentence is within limits, keep it whole (preserves maximum context)
        if len(words) <= max_words * 1.5:
            chunks.append(sentence)
        else:
            # Sentence is too long, split by discourse markers
            current_words = []
            for i, word in enumerate(words):
                current_words.append(word)
                current_text = " ".join(current_words)
                is_marker = any(current_text.lower().endswith(m.strip()) for m in _VI_SPLIT_MARKERS)
                
                if len(current_words) >= max_words and (is_marker or i == len(words) - 1):
                    chunks.append(current_text.strip())
                    current_words = []
                elif len(current_words) >= max_words * 2:
                    # Hard split
                    chunks.append(current_text.strip())
                    current_words = []
            
            if current_words:
                chunks.append(" ".join(current_words).strip())
                
    # Merge very short chunks (< 4 words) into the previous chunk to avoid broken context
    final_chunks = []
    for c in chunks:
        if not final_chunks:
            final_chunks.append(c)
        elif len(c.split()) < 4:
            final_chunks[-1] = final_chunks[-1] + " " + c
        else:
            final_chunks.append(c)
            
    return final_chunks


class Translator:
    """
    Bidirectional translator: Vietnamese ↔ English (VI2EN, EN2VI).

    Sử dụng VietAI/envit5-translation — 1 model duy nhất cho cả 2 chiều.
    Prefix T5: "vi: [text]" cho VI→EN, "en: [text]" cho EN→VI.
    Kích thước ~600MB, nhanh hơn VinAI mBART (3.4GB) ≈ 5 lần.
    """

    REMOTE_MODEL = "VietAI/envit5-translation"

    # T5 prefix cho từng chiều dịch
    _T5_PREFIX = {
        "vi2en": "vi: ",   # Tiếng Việt → Tiếng Anh
        "en2vi": "en: ",   # Tiếng Anh → Tiếng Việt
    }

    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = config["translation"]
        self.model_dir = cfg.get("model_dir", None)
        self.max_length = cfg.get("max_length", 512)
        self.model_name = cfg.get("model", self.REMOTE_MODEL)
        self.backend = cfg.get("backend", "auto")
        self.ct2_model_dir = cfg.get("ct2_model_dir", "models/envit5-ct2-int8/")
        self.compute_type = cfg.get("compute_type", "int8")
        self.inter_threads = int(cfg.get("inter_threads", 1))
        self.intra_threads = int(cfg.get("intra_threads", 4))
        self.beam_size = int(cfg.get("beam_size", 1))
        self._active_backend = "transformers"
        self._ct2_translator = None
        self._tokenizer = None
        self._model = None
        # Backward compat — các phương thức khác vẫn dùng dict
        self._models: dict = {}
        self._tokenizers: dict = {}

    def _get_model_source(self) -> str:
        """Return local checkpoint if available, else HuggingFace remote."""
        import os
        if self.model_dir:
            config_file = os.path.join(self.model_dir, "config.json")
            if os.path.exists(config_file):
                print(f"[MT] Using local checkpoint: {self.model_dir}")
                return self.model_dir
        print(f"[MT] Using pretrained: {self.model_name}")
        return self.model_name

    def load(self):
        """Load VietAI/envit5 — 1 model cho cả VI↔EN."""
        src = self._get_model_source()

        if self._try_load_ctranslate2():
            print("[MT] ✅ CTranslate2 envit5 loaded (VI↔EN single model).")
            term_path = os.path.join(
                os.path.dirname(__file__), "../../data/industrial_terms.csv"
            )
            load_terminology(os.path.abspath(term_path))
            return

        print(f"[MT] Loading {self.model_name} on {self.device}...")

        self._tokenizer = self._load_tokenizer(src)
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(src)
        config.tie_word_embeddings = False
        self._model = AutoModelForSeq2SeqLM.from_pretrained(src, config=config).to(self.device)

        # Backward compat: share same instance cho cả 2 chiều
        for direction in ["vi2en", "en2vi"]:
            self._tokenizers[direction] = self._tokenizer
            self._models[direction] = self._model

        print("[MT] ✅ VietAI/envit5 loaded (VI↔EN single model).")

        # Load industrial terminology overrides
        term_path = os.path.join(
            os.path.dirname(__file__), "../../data/industrial_terms.csv"
        )
        load_terminology(os.path.abspath(term_path))

    def _try_load_ctranslate2(self) -> bool:
        """Load an exported CTranslate2 envit5 model when configured and available."""
        if self.backend not in {"auto", "ctranslate2"}:
            return False

        ct2_dir = os.path.abspath(self.ct2_model_dir)
        model_bin = os.path.join(ct2_dir, "model.bin")
        if not os.path.exists(model_bin):
            if self.backend == "ctranslate2":
                raise FileNotFoundError(
                    f"CTranslate2 model not found at {ct2_dir}. "
                    "Run scripts/export_envit5_ct2.py first."
                )
            print(f"[MT] CTranslate2 export not found at {ct2_dir}; using Transformers.")
            return False

        try:
            import ctranslate2

            self._tokenizer = self._load_tokenizer(ct2_dir)
            self._ct2_translator = ctranslate2.Translator(
                ct2_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type=self.compute_type,
                inter_threads=self.inter_threads,
                intra_threads=self.intra_threads,
            )
            self._active_backend = "ctranslate2"
            for direction in ["vi2en", "en2vi"]:
                self._tokenizers[direction] = self._tokenizer
            print(
                f"[MT] Loading CTranslate2 from {ct2_dir} "
                f"(compute_type={self.compute_type}, beam={self.beam_size})"
            )
            return True
        except Exception:
            if self.backend == "ctranslate2":
                raise
            print("[MT] CTranslate2 load failed; using Transformers fallback.")
            return False

    def _load_tokenizer(self, source: str):
        try:
            return AutoTokenizer.from_pretrained(source)
        except Exception as exc:
            print(f"[MT] AutoTokenizer failed ({exc}); trying PreTrainedTokenizerFast.")
            return PreTrainedTokenizerFast.from_pretrained(source)

    def translate(self, text: str, direction: str = "vi2en") -> str:
        """
        Translate text between Vietnamese and English.
        Automatically chunks long unpunctuated sentences to avoid truncation.

        Args:
            text: input text
            direction: "vi2en" (Vietnamese→English) or "en2vi" (English→Vietnamese)

        Returns:
            Translated text string.
        """
        if not text.strip():
            return ""
        valid = direction in self._tokenizers if self._active_backend == "ctranslate2" else direction in self._models
        if not valid:
            raise ValueError(f"Unsupported direction: {direction}. Use 'vi2en' or 'en2vi'.")

        src_lang = "vi" if direction == "vi2en" else "en"

        # ── Bước 0: Khôi phục dấu câu (Punctuation Restoration) ──────────────
        # Chuyển "cậu đã làm gì hả nó hoạt động vậy"
        #   → "Cậu đã làm gì hả? Nó hoạt động vậy."
        # Giúp MT dịch từng câu hoàn chỉnh thay vì chuỗi liên tục
        from utils.punctuator import restore_punctuation
        punced_text = restore_punctuation(text, lang=src_lang)

        # ── Bước 1: Normalize (lowercase, unit expansion) ─────────────────────
        from utils.text_normalizer import normalize
        normalized_text = normalize(punced_text, lang=src_lang)

        # Do NOT inject dictionary before translation to preserve model's context reasoning.
        text_with_terms = normalized_text

        # Chunk text based on sentences to preserve context for translation
        chunks = _chunk_text(text_with_terms, max_words=15)
        if len(chunks) > 1:
            print(f"[MT] 🔀 Dịch theo {len(chunks)} câu/đoạn để bảo toàn ngữ cảnh")

        t0_total = time.perf_counter()
        translated_chunks = []

        for chunk in chunks:
            # Áp dụng T5 prefix cho từng chunk
            prefix = self._T5_PREFIX.get(direction, "vi: ")
            prefixed_chunk = prefix + chunk

            if self._active_backend == "ctranslate2":
                chunk_result = self._translate_chunk_ct2(prefixed_chunk, direction)
            else:
                chunk_result = self._translate_chunk_transformers(prefixed_chunk, direction)
            # Remove "en: " or "vi: " prefix that VietAI/envit5 sometimes generates
            chunk_result = re.sub(r"^(en|vi):\s*", "", chunk_result, flags=re.IGNORECASE).strip()
            translated_chunks.append(chunk_result)

        result = " ".join(translated_chunks)
        
        # ── Bước 3: Post-translation terminology correction ────────────────────
        # Apply terminology AFTER translation to enforce dictionary terms.
        if direction == "vi2en":
            result = _apply_terminology(result, _TERM_OVERRIDE_VI_EN)
        else:
            result = _apply_terminology(result, _TERM_OVERRIDE_EN_VI)
            
        # Apply EN→EN or VI→VI post-translation corrections to fix known model errors.
        result = _apply_terminology(result, _POST_TRANSLATE_FIX)

        elapsed_ms = (time.perf_counter() - t0_total) * 1000
        arrow = "VI→EN" if direction == "vi2en" else "EN→VI"
        print(f"[MT] ⏱ {elapsed_ms:.0f}ms | {arrow} | \"{text_with_terms}\" → \"{result}\"")

        return result

    def _translate_chunk_transformers(self, prefixed_chunk: str, direction: str) -> str:
        tokenizer = self._tokenizers[direction]
        model = self._models[direction]

        inputs = tokenizer(
            prefixed_chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=max(self.beam_size, 1),
                early_stopping=True,
                repetition_penalty=1.1,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_chunk_ct2(self, prefixed_chunk: str, direction: str) -> str:
        tokenizer = self._tokenizers[direction]
        token_ids = tokenizer.encode(
            prefixed_chunk,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        source_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        result = self._ct2_translator.translate_batch(
            [source_tokens],
            beam_size=max(self.beam_size, 1),
            max_decoding_length=self.max_length,
        )[0]
        output_ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    def run(self, text_in_queue: queue.Queue, text_out_queue: queue.Queue):
        """
        Worker loop: reads ASR results, translates, pushes to TTS queue.
        Each item: {"text": str, "lang": str, "direction": str}
        """
        print("[MT Worker] ✅ Started (VI↔EN)")
        while True:
            try:
                item = text_in_queue.get(timeout=1)
                direction = item.get("direction", "vi2en")
                text = item["text"]
                translated = self.translate(text, direction=direction)

                if translated:
                    text_out_queue.put({
                        "text": translated,
                        "direction": direction,
                        "original": text,
                    })
                text_in_queue.task_done()
            except queue.Empty:
                continue


if __name__ == "__main__":
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    t = Translator(cfg)
    t.load()

    tests = [
        ("vi2en", "Máy xúc số 3 đang bị lỗi thủy lực, cần kỹ sư kiểm tra ngay."),
        ("vi2en", "Van an toàn trên đường ống số 5 bị rò rỉ."),
        ("en2vi", "The hydraulic jack on excavator number 3 has failed."),
        ("en2vi", "Please check the safety valve on pipeline 5 immediately."),
    ]
    print("\n── Translation Tests ──")
    for direction, text in tests:
        result = t.translate(text, direction)
        print(f"  [{direction}] {text}")
        print(f"         → {result}\n")
