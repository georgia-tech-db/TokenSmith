"""Checks whether retrieved passages entail each claim in an answer.

Word overlap and embedding similarity both score "A does not preserve X" as a
near-perfect match for a passage saying "A preserves X". A natural language
inference (NLI) cross-encoder reads the two texts together and says whether the
passage entails, contradicts, or says nothing about the claim, which is the
question abstention actually has to answer.
"""

import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy ships with the runtime
    np = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional until the runtime is rebuilt
    ort = None  # type: ignore

try:
    from tokenizers import Tokenizer  # type: ignore
except Exception:  # pragma: no cover - optional until the runtime is rebuilt
    Tokenizer = None  # type: ignore


NLI_MODEL_DIR_ENV = "TOKENSMITH_NLI_MODEL_DIR"
DEFAULT_NLI_MODEL_NAME = "deberta-v3-base-tasksource-nli"

MAX_SEQUENCE_TOKENS = 512
BATCH_SIZE = 16

# NLI models are trained on premises of a sentence or two. Fed a whole textbook
# passage they grow noisy, finding "contradictions" in text that has nothing to
# do with the claim. So each passage is cut into spans of up to this many
# consecutive sentences, and each claim is read only against the spans that
# talk about it.
MAX_SPAN_SENTENCES = 4
SPANS_PER_CLAIM = 4

# Consecutive answer sentences checked together as one claim. Each check costs
# a model call per span, so grouping keeps long answers from taking seconds.
SENTENCES_PER_CLAIM = 1

# A claim with fewer words than this is a connective or a courtesy ("Here is
# why:"), not something a passage can support.
MIN_CLAIM_WORDS = 4

# An answer this short is usually a bare fragment ("gcc, C", "November 11th,
# 2025") that only states something together with the question it answers.
FRAGMENT_ANSWER_WORDS = 12

# Sentences about the context itself ("The text does not mention a random
# scheduler") report an absence. No passage can entail that, and it is not a
# claim about the subject, so it is left out rather than counted against the
# answer.
_context_reference = re.compile(
    r"\b(?:the|this|provided|given)\s+(?:context|text|excerpts?|passages?|sources?|material|materials|documents?)\b",
    re.IGNORECASE,
)
_absence = re.compile(
    r"\b(?:no|not|never|nothing|doesn't|does not|don't|do not|isn't|is not|without)\b|\bn't\b",
    re.IGNORECASE,
)


class NliModel:
    def __init__(self, model_dir: Path) -> None:
        if ort is None or Tokenizer is None or np is None:
            raise RuntimeError("onnxruntime and tokenizers are required for answer verification.")

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        labels = {str(label).lower(): int(index) for index, label in config["id2label"].items()}
        self.entailment_index = labels["entailment"]
        self.contradiction_index = labels["contradiction"]
        self.neutral_index = labels["neutral"]

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        # Windows keep premises short, so this only guards an unusually long claim.
        self.pair_tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.pair_tokenizer.no_padding()
        self.pair_tokenizer.enable_truncation(MAX_SEQUENCE_TOKENS, strategy="only_first")
        self.pad_id = int(config.get("pad_token_id") or 0)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def probabilities(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for offset in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[offset:offset + BATCH_SIZE]
            ids = [list(encoding.ids) for encoding in self.pair_tokenizer.encode_batch(list(batch))]
            width = max(len(row) for row in ids)
            input_ids = np.full((len(ids), width), self.pad_id, dtype=np.int64)
            attention = np.zeros((len(ids), width), dtype=np.int64)
            for row, values in enumerate(ids):
                input_ids[row, : len(values)] = values
                attention[row, : len(values)] = 1

            feed = {"input_ids": input_ids, "attention_mask": attention}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            logits = self.session.run(None, feed)[0]
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)
            for row in probs:
                results.append(
                    {
                        "entailment": float(row[self.entailment_index]),
                        "contradiction": float(row[self.contradiction_index]),
                        "neutral": float(row[self.neutral_index]),
                    }
                )
        return results


_model: Optional[NliModel] = None
_model_dir: Optional[Path] = None
_model_lock = threading.Lock()


def default_model_dir() -> Path:
    configured = os.environ.get(NLI_MODEL_DIR_ENV)
    if configured:
        return Path(configured)
    # The model ships beside the private Python runtime: app_runtime/models/<name>.
    runtime_root = Path(os.path.realpath(sys.executable)).parent.parent
    return runtime_root.parent / "models" / DEFAULT_NLI_MODEL_NAME


def load_model(model_dir: Optional[str] = None) -> NliModel:
    global _model, _model_dir
    path = Path(model_dir) if model_dir else default_model_dir()
    with _model_lock:
        if _model is None or _model_dir != path:
            _model = NliModel(path)
            _model_dir = path
        return _model


_markdown_prefix = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s*)")
_sentence_boundary = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9`*])")
_word = re.compile(r"[\w'’]+", re.UNICODE)


def is_context_absence(sentence: str) -> bool:
    return bool(_context_reference.search(sentence) and _absence.search(sentence))


def split_claims(answer: str, question: str = "") -> List[str]:
    """Splits an answer into sentence-level claims, dropping code and filler."""
    sentences = [sentence for sentence in split_sentences(answer) if not is_context_absence(sentence)]
    answer_words = sum(len(_word.findall(sentence)) for sentence in sentences)
    if question.strip() and sentences and answer_words <= FRAGMENT_ANSWER_WORDS:
        return [f"{question.strip()} {' '.join(sentences)}"]

    kept = [
        sentence
        for sentence in sentences
        if not sentence.endswith(("?", ":")) and len(_word.findall(sentence)) >= MIN_CLAIM_WORDS
    ]
    return [" ".join(kept[start:start + SENTENCES_PER_CLAIM]) for start in range(0, len(kept), SENTENCES_PER_CLAIM)]


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    sentences: List[str] = []
    for line in text.splitlines():
        line = _markdown_prefix.sub("", line).strip().replace("**", "").replace("__", "")
        sentences.extend(part.strip() for part in _sentence_boundary.split(line) if part.strip())
    return sentences


def word_key(word: str) -> str:
    """Folds common inflections so "pages"/"page" and "evicted"/"evict" match."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def span_words(text: str) -> set:
    return {word_key(word) for word in _word.findall(text.lower()) if len(word) > 2}


def passage_spans(passages: Sequence[str]) -> List[Tuple[int, str]]:
    spans: List[Tuple[int, str]] = []
    for passage_index, passage in enumerate(passages):
        sentences = split_sentences(str(passage or ""))
        for start in range(len(sentences)):
            for size in range(1, MAX_SPAN_SENTENCES + 1):
                if start + size > len(sentences):
                    break
                spans.append((passage_index, " ".join(sentences[start:start + size])))
    return spans


_named_term = re.compile(
    r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b"  # acronyms and CamelCase: SJF, OLTP, PostgreSQL
    r"|\b\d[\d.,]*\b"  # numbers: 1985, 64, 13.2
    r"|\b\w+_\w+\b"  # code identifiers: try_to_insert
    r"|(?<=[a-z,;] )[A-Z][a-z]{2,}\b"  # a capitalized word mid-sentence: Java, Rust
)


def named_terms(text: str) -> List[str]:
    """Names, numbers, and identifiers a claim asserts about specifically."""
    return sorted({match.group(0).rstrip(".,") for match in _named_term.finditer(text)})


def term_in(term: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", text, re.IGNORECASE) is not None


def candidate_spans(
    claim: str,
    spans: Sequence[Tuple[int, str]],
    span_word_sets: Sequence[set],
    idf: Dict[str, float],
    terms: Sequence[str],
) -> List[int]:
    """Picks the spans most about this claim, by IDF-weighted shared words.

    This only chooses what the NLI model reads; it never decides support. Its
    job is to keep unrelated sentences, which make NLI models hallucinate
    contradictions, out of the premise.
    """
    words = span_words(claim)
    if not words:
        return []

    def relevance(index: int) -> float:
        shared = words & span_word_sets[index]
        # A span that names what the claim names is read first. Without the
        # name in the premise, NLI models assume it: a sentence about a
        # sequential scan "entails" that OLTP runs scans.
        named = sum(1 for term in terms if term_in(term, spans[index][1]))
        # Prefer the shortest span that covers the claim: a longer span wins
        # only by covering more of it, not by being longer.
        return 100.0 * named + sum(idf.get(word, 0.0) for word in shared) - 0.01 * len(span_word_sets[index])

    ranked = sorted(range(len(spans)), key=relevance, reverse=True)
    return [index for index in ranked[:SPANS_PER_CLAIM] if span_word_sets[index] & words]


def verify_claims(
    answer: str,
    passages: Sequence[str],
    model_dir: Optional[str] = None,
    question: str = "",
) -> Dict[str, Any]:
    """Grades every claim in `answer` against the passages.

    For each claim this reports, per passage, the strongest entailment and
    contradiction among the spans read, plus the span that entails it best, so
    the caller can decide support per claim and credit the passage that
    carried it.
    """
    claims = split_claims(answer, question)
    if not claims or not passages:
        return {"claims": [], "available": True}

    model = load_model(model_dir)
    spans = passage_spans(passages)
    span_word_sets = [span_words(text) for _, text in spans]
    sentence_count = max(1, len(spans))
    document_frequency: Dict[str, int] = {}
    for words in span_word_sets:
        for word in words:
            document_frequency[word] = document_frequency.get(word, 0) + 1
    idf = {word: math.log(1 + sentence_count / count) for word, count in document_frequency.items()}

    # A fragment answer is checked as "question + answer", but only the answer
    # part asserts anything, so only its names must appear in the material.
    claim_terms = [named_terms(claim[len(question.strip()):] if question.strip() and claim.startswith(question.strip()) else claim) for claim in claims]
    passage_text = "\n".join(str(passage or "") for passage in passages)
    missing_terms = [[term for term in terms if not term_in(term, passage_text)] for terms in claim_terms]
    selected = [
        candidate_spans(claim, spans, span_word_sets, idf, terms)
        for claim, terms in zip(claims, claim_terms)
    ]
    pairs = [(spans[index][1], claim) for claim, indices in zip(claims, selected) for index in indices]
    scores = iter(model.probabilities(pairs) if pairs else [])

    results: List[Dict[str, Any]] = []
    for claim, indices, missing in zip(claims, selected, missing_terms):
        per_passage = [{"index": index, "entailment": 0.0, "contradiction": 0.0} for index in range(len(passages))]
        evidence: Optional[str] = None
        best_entailment = 0.0
        for span_index in indices:
            score = next(scores)
            passage_index, text = spans[span_index]
            best = per_passage[passage_index]
            if missing:
                # The material never names what this claim is about, so no
                # passage can entail it, whatever the model thinks.
                score = {**score, "entailment": 0.0}
            best["entailment"] = max(best["entailment"], score["entailment"])
            best["contradiction"] = max(best["contradiction"], score["contradiction"])
            if score["entailment"] > best_entailment:
                best_entailment = score["entailment"]
                evidence = text
        results.append({"text": claim, "evidence": evidence, "missingTerms": missing, "passages": per_passage})

    return {"claims": results, "available": True}


def verify_answer(payload: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(payload.get("answer") or "")
    passages = [str(passage or "") for passage in payload.get("passages") or []]
    try:
        return verify_claims(answer, passages, payload.get("modelDir"), str(payload.get("question") or ""))
    except Exception as error:
        return {"claims": [], "available": False, "error": str(error)}
