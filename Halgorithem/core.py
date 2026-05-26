import os
import re
import warnings
from pathlib import Path

import pysbd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .claim_extraction import extract_claims
from .confidence import classify_support, confidence_score
from .contradiction import equivalent_unit_numbers, find_contradiction, missing_location_evidence, numbers_conflict
from .evidence import best_evidence, build_evidence
from .math_utils import numbers_close, safe_eval
from .retrieval import rank_chunks
from .source_quality import score_source
from .temporal import temporal_warning
from .temporal import extract_years
from .text_processing import (
    clean_text,
    extract_entities,
    extract_numbers,
    get_synonyms,
    has_negation_mismatch,
    lemmatize_tokens,
    tokenize,
)
from .nlp import nlp


class LocalEmbedder:
    kind = "lexical"
    model_name = "HashingVectorizer"
    fallback_reason = None

    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=2 ** 14,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def encode(self, text, convert_to_tensor=False):
        return self.vectorizer.transform([text or ""])

    def similarity(self, left, right):
        return float(cosine_similarity(left, right)[0][0])


def _load_embedder():
    mode = os.getenv("HALGORITHEM_EMBEDDER", "semantic").lower()
    if mode in {"semantic", "sentence-transformers", "sentence_transformers", "st"}:
        try:
            from sentence_transformers import SentenceTransformer, util

            model_name = os.getenv("HALGORITHEM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            allow_download = os.getenv("HALGORITHEM_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
            model = SentenceTransformer(model_name, local_files_only=not allow_download)

            class SentenceTransformerEmbedder:
                kind = "semantic"
                fallback_reason = None
                model_name = None

                def __init__(self, loaded_model_name):
                    self.model_name = loaded_model_name

                def encode(self, text, convert_to_tensor=False):
                    return model.encode(text or "", convert_to_tensor=True)

                def similarity(self, left, right):
                    return float(util.cos_sim(left, right))

            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:
            warnings.warn(
                f"Could not load semantic embedder ({exc}); using local lexical hashing embedder.",
                RuntimeWarning,
            )
            fallback = LocalEmbedder()
            fallback.fallback_reason = str(exc)
            return fallback
    return LocalEmbedder()
INITIAL_RE = re.compile(r"\b[a-z]\.$", re.IGNORECASE)


class Halgorithm:
    def __init__(self, sentences_per_chunk=2, sentence_overlap=1, embedder=None):
        sentences_per_chunk = int(sentences_per_chunk)
        sentence_overlap = int(sentence_overlap)
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be at least 1.")
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be at least 0.")
        if sentence_overlap >= sentences_per_chunk:
            raise ValueError("sentence_overlap must be less than sentences_per_chunk.")
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.embedder = embedder or _load_embedder()
        self.parser = pysbd.Segmenter(language="en", clean=False)

    @property
    def diagnostics(self):
        return {
            "embedder": getattr(self.embedder, "kind", "unknown"),
            "embedding_model": getattr(self.embedder, "model_name", None),
            "embedding_fallback_reason": getattr(self.embedder, "fallback_reason", None),
        }

    # ── Text prep ─────────────────────────────────────────────────────────────

    def clean_text(self, text):
        return clean_text(text)

    def split_sentences(self, text):
        text = self.clean_text(text)
        sentences = self.parser.segment(text)
        merged = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if merged and INITIAL_RE.search(merged[-1]):
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                merged.append(sentence)
        return merged

    def tokenize(self, text):
        return tokenize(text)

    def lemmatize_tokens(self, text):
        return lemmatize_tokens(text)

    def extract_numbers(self, text):
        return extract_numbers(text)

    def extract_entities(self, text):
        return extract_entities(text)

    def has_negation_mismatch(self, claim, chunk_text):
        return has_negation_mismatch(claim, chunk_text)

    def get_synonyms(self, word):
        return get_synonyms(word)

    # ── File loading ──────────────────────────────────────────────────────────

    def load_file(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not read {file_path!s} as UTF-8 text.") from exc

    def load_files(self, file_paths):
        return [
            {"file_id": i, "file_path": str(fp), "text": self.load_file(fp)}
            for i, fp in enumerate(file_paths, 1)
        ]

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_text(self, text, doc_id=1, source_name=None):
        sentences = self.split_sentences(text)
        chunks, start, chunk_id = [], 0, 1
        quality = score_source(source_name, text)
        while start < len(sentences):
            end = start + self.sentences_per_chunk
            chunk = " ".join(sentences[start:end])
            chunks.append({
                "doc_id": doc_id,
                "source_name": source_name,
                "source_quality": quality,
                "chunk_id": chunk_id,
                "sentence_start": start + 1,
                "sentence_end": min(end, len(sentences)),
                "text": chunk,
                "tokens": self.tokenize(chunk),
                "entities": self.extract_entities(chunk),
                "numbers": self.extract_numbers(chunk),
                "embedding": self.embedder.encode(chunk, convert_to_tensor=True),
            })
            chunk_id += 1
            if end >= len(sentences):
                break
            start = end - self.sentence_overlap
        return chunks

    # ── Scoring ───────────────────────────────────────────────────────────────

    def support_score(self, claim, chunk):
        # semantic similarity via sentence-transformers — topic-agnostic
        claim_emb = self.embedder.encode(claim, convert_to_tensor=True)
        return self.embedder.similarity(claim_emb, chunk["embedding"])

    # ── Math claims ───────────────────────────────────────────────────────────

    def classify_claim_type(self, claim):
        if re.search(r"(?<!\w)=(?!\w)", claim.lower()):
            return "MATH"
        return "SOURCE"

    def verify_math_claim(self, claim):
        base = self.empty_result(claim, status="ERROR", reason="Malformed math claim", result_type="MATH")
        if "=" not in claim:
            base["reason"] = "No expression found"
            return base
        parts = claim.split("=", 1)
        if len(parts) != 2:
            base["reason"] = "Malformed expression"
            return base
        try:
            left, right = safe_eval(parts[0].strip()), safe_eval(parts[1].strip())
            if numbers_close(left, right):
                result = self.empty_result(claim, status="SUPPORTED", reason="", result_type="MATH")
                result["confidence"] = 1.0
                result["score"] = 1.0
                return result
            result = self.empty_result(claim, status="CONTRADICTION", reason="Math mismatch", result_type="MATH")
            result["expected"] = left
            result["got"] = right
            result["confidence"] = 1.0
            result["score"] = 1.0
            return result
        except Exception as e:
            base["reason"] = str(e)
            return base

    def empty_result(self, claim, status="HALLUCINATION", reason="", result_type="SOURCE"):
        return {
            "claim": claim,
            "status": status,
            "confidence": 0.0,
            "score": 0.0,
            "matched_doc_id": None,
            "matched_source": None,
            "matched_chunk_id": None,
            "matched_chunk": "",
            "chunk_text": "",
            "evidence": [],
            "unsupported_terms": [],
            "reason": reason,
            "warning": None,
            "type": result_type,
        }

    # ── Number conflict ───────────────────────────────────────────────────────

    def has_number_conflict(self, claim, chunk):
        issue = numbers_conflict(claim, chunk, self.extract_numbers)
        claim_numbers = set(self.extract_numbers(claim))
        truth_numbers = set(chunk["numbers"])
        return bool(issue), claim_numbers, truth_numbers

    # ── Meaningful claim filter ───────────────────────────────────────────────

    def is_meaningful_claim(self, claim):
        claim_l = claim.lower().strip()

        # filter metadata/citations
        BAD_PATTERNS = [
            "adapted from",
            "source:",
            "sources:",
            "http://",
            "https://",
            "www.",
        ]
        if any(p in claim_l for p in BAD_PATTERNS):
            return False

        tokens = self.tokenize(claim)

        last_word = claim.strip().rstrip(".").split()[-1].lower()
        if last_word in {"including", "such", "namely", "follows", "following", "as"}:
            return False

        doc = nlp(claim)

        has_anchor = any(doc.ents) or any(t.like_num for t in doc) or any(t.pos_ == "PROPN" for t in doc)

        # reject vague summary sentences
        subject = next((t for t in doc if t.dep_ == "nsubj"), None)
        if subject and subject.text.lower() in {"these", "this", "those", "such"}:
            return False

        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        SUMMARY_VERBS = {
            "reflect", "demonstrate", "highlight", "illustrate", "suggest",
            "indicate", "underscore", "emphasize", "represent", "signal",
            "mark", "mean", "position", "pivot",
        }
        if root and root.lemma_.lower() in SUMMARY_VERBS:
            return False

        # NEW: accept definition/explanation claims
        TECHNICAL_ANCHORS = {
            "class", "classes", "object", "objects", "instance", "instances",
            "attribute", "attributes", "method", "methods", "function",
            "functions", "type", "data", "namespace", "inheritance",
            "module", "argument", "variable", "state", "language", "programming"
        }

        if any(t in TECHNICAL_ANCHORS for t in tokens):
            return True

        FACTUAL_RELATION_VERBS = {
            "create", "invent", "develop", "originate", "build", "design", "use",
            "interpret", "allow", "become"
        }
        if any(t.lemma_.lower() in FACTUAL_RELATION_VERBS for t in doc) and len(tokens) >= 3:
            return True

        # original anchor logic, but now only fallback
        if has_anchor:
            return True

        if len(tokens) < 4:
            return False

        # accept normal factual sentences with subject + verb
        has_subject = any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc)
        has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc)

        if has_subject and has_verb and len(tokens) >= 4:
            return True

        if has_verb and len(tokens) >= 4:
            return True

        return False

    # ── Unsupported terms ─────────────────────────────────────────────────────

    def get_unsupported_terms(self, claim, all_truth_tokens):
        def is_year_token(token):
            try:
                value = float(token)
            except (TypeError, ValueError):
                return False
            return value.is_integer() and 1400 <= value <= 2100

        claim_tokens = set(self.tokenize(claim))
        all_truth_tokens = set(all_truth_tokens)
        unsupported = {
            t for t in claim_tokens
            if t not in all_truth_tokens
            and not (self.get_synonyms(t) & all_truth_tokens)
        }
        doc = nlp(claim)
        # only proper nouns and non-year numbers are real hallucination signals
        content = {
            t.lemma_.lower()
            for t in doc
            if t.pos_ in {"PROPN", "NUM"}
            and not t.is_stop
            and not is_year_token(t.lemma_.lower())
        }
        relation_terms = set()
        lowered = claim.lower()
        if re.search(r"\b(created|invented|developed|originated|came|built|maintained|located|priced|report|reports)\b", lowered):
            relation_terms = {
                t for t in unsupported
                if t.isalnum()
                and not t.isdigit()
                and t not in {
                    "created", "invented", "developed", "originated", "came",
                    "built", "maintained", "located", "priced", "report", "reports"
                }
            }
        identifier_terms = {
            token
            for token in re.findall(r"\b(?:product|model|version)\s+([a-z0-9]+)\b", lowered)
            if token in unsupported
        }
        return sorted(
            t for t in unsupported
            if t in content
            or t in relation_terms
            or t in identifier_terms
            or (t.replace(".", "", 1).isdigit() and not is_year_token(t))
        )

    # ── Core claim checker ────────────────────────────────────────────────────

    def check_claim_against_chunks(self, claim, chunks, all_truth_tokens, threshold=0.30):
        candidates = rank_chunks(
            claim=claim,
            chunks=chunks,
            score_fn=self.support_score,
            extract_numbers=self.extract_numbers,
            has_negation_mismatch=self.has_negation_mismatch,
            threshold=threshold,
            top_k=5,
        )
        unsupported_terms = self.get_unsupported_terms(claim, all_truth_tokens)
        evidence = build_evidence(candidates)
        best = best_evidence(candidates)

        if not best:
            result = self.empty_result(claim, status="HALLUCINATION", reason="No matching chunk found")
            result["unsupported_terms"] = unsupported_terms
            result.update(self.diagnostics)
            warning = temporal_warning(claim)
            if warning:
                result["warning"] = warning["warning"]
                result["as_of_year"] = warning["as_of_year"]
            return result

        best_chunk = candidates[0]["chunk"]
        best_score = candidates[0]["score"]
        equivalent_numbers = equivalent_unit_numbers(claim, best_chunk.get("text", ""))
        if equivalent_numbers:
            unsupported_terms = [t for t in unsupported_terms if t not in equivalent_numbers]
        contradiction = None
        contradiction_score = best_score
        best_years = extract_years(best_chunk.get("text", ""))
        claim_years = extract_years(claim)
        best_has_claim_year = bool(claim_years and best_years and claim_years & best_years)
        for candidate in candidates:
            if candidate["score"] < max(threshold, best_score * 0.75):
                continue
            issue = find_contradiction(
                claim=claim,
                chunk=candidate["chunk"],
                extract_numbers=self.extract_numbers,
                has_negation_mismatch=self.has_negation_mismatch,
                score=candidate["score"],
                threshold=threshold,
            )
            if issue:
                if issue.get("reason") == "Date mismatch" and best_has_claim_year:
                    continue
                contradiction = issue
                contradiction_score = candidate["score"]
                break
        if not contradiction and missing_location_evidence(claim, best_chunk.get("text", "")):
            unsupported_terms = sorted(set(unsupported_terms) | {"location"})
        status = classify_support(
            score=best_score,
            threshold=threshold,
            contradiction=contradiction,
            unsupported_terms=unsupported_terms,
            claim=claim,
        )
        confidence = confidence_score(
            score=best_score,
            evidence_count=len(evidence),
            contradiction=contradiction,
            unsupported_terms=unsupported_terms,
            status=status,
        )
        warning = temporal_warning(claim)

        result = {
            "status": status, "claim": claim, "score": best_score,
            "confidence": confidence,
            "matched_doc_id": best["doc_id"],
            "matched_source": best["source"],
            "matched_chunk_id": best["chunk_id"],
            "matched_chunk": best["text"],
            "chunk_text": best["text"],
            "unsupported_terms": unsupported_terms,
            "evidence": evidence,
            "reason": "",
            "warning": None,
            **self.diagnostics,
        }
        if warning:
            result["warning"] = warning["warning"]
            result["as_of_year"] = warning["as_of_year"]
        if contradiction:
            result["reason"] = contradiction["reason"]
            result["contradiction_score"] = contradiction_score
            if "claim_numbers" in contradiction:
                result["ai_numbers"] = contradiction["claim_numbers"]
                result["truth_numbers"] = contradiction["truth_numbers"]
            if "claim_years" in contradiction:
                result["ai_years"] = contradiction["claim_years"]
                result["truth_years"] = contradiction["truth_years"]
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def compare_to_docs(self, truth_docs, ai_output, threshold=0.30):
        if not ai_output or not str(ai_output).strip():
            return []
        if isinstance(truth_docs, str):
            truth_docs = [{"file_id": 1, "file_path": "inline_text", "text": truth_docs}]
        elif truth_docs and isinstance(truth_docs[0], str):
            truth_docs = [
                {"file_id": i, "file_path": f"inline_text_{i}", "text": t}
                for i, t in enumerate(truth_docs, 1)
            ]
        elif truth_docs is None:
            raise ValueError("No source documents loaded. Provide non-empty truth_docs.")
        elif not isinstance(truth_docs, list):
            raise ValueError("truth_docs must be a string, list of strings, or list of document dictionaries.")

        all_chunks, all_truth_tokens = [], set()
        for i, doc in enumerate(truth_docs, 1):
            if not isinstance(doc, dict):
                raise ValueError("truth_docs entries must be strings or dictionaries.")
            if "text" not in doc:
                raise ValueError("truth_docs entries must include a 'text' field.")
            text = doc.get("text") or ""
            if not str(text).strip():
                continue
            doc_id = doc.get("file_id", i)
            file_path = doc.get("file_path", f"inline_text_{i}")
            chunks = self.chunk_text(text, doc_id=doc_id, source_name=file_path)
            all_chunks.extend(chunks)
            for chunk in chunks:
                all_truth_tokens.update(chunk["tokens"])
        if not all_chunks:
            raise ValueError("No source documents loaded. Provide at least one non-empty source.")

        results = []
        claims = extract_claims(
            ai_output,
            sentence_splitter=self.split_sentences,
            meaningful_filter=self.is_meaningful_claim,
        )
        for claim_data in claims:
            claim = claim_data["claim"]
            claim_type = self.classify_claim_type(claim)
            if claim_type == "MATH":
                result = self.verify_math_claim(claim)
            else:
                result = self.check_claim_against_chunks(
                    claim=claim,
                    chunks=all_chunks,
                    all_truth_tokens=all_truth_tokens,
                    threshold=threshold,
                )
            result["claim_id"] = claim_data["claim_id"]
            result["sentence_id"] = claim_data["sentence_id"]
            result["source_sentence"] = claim_data["source_sentence"]
            result["type"] = claim_type
            results.append(result)
        return results

    def compare_to_files(self, truth_file_paths, ai_output, threshold=0.30):
        return self.compare_to_docs(self.load_files(truth_file_paths), ai_output, threshold)

    def compare_with_reasoning(self, truth_file_paths, ai_output, threshold=0.30):
        return self.compare_to_files(truth_file_paths, ai_output, threshold)

    def print_report(self, results):
        supported = [r for r in results if r["status"] == "SUPPORTED"]
        weak = [r for r in results if r["status"] == "WEAK_SUPPORT"]
        bad = [r for r in results if r["status"] in {"HALLUCINATION", "CONTRADICTION"}]
        uncertain = [r for r in results if r["status"] == "UNVERIFIABLE_DENIAL"]
        total = len(results)
        confidence = (len(supported) + 0.5 * len(weak)) / total if total else 0

        print("\nHalgorithm Report")
        print("=" * 80)
        print(
            f"Strongly supported: {len(supported)}  Weak: {len(weak)}  "
            f"Unverifiable denials: {len(uncertain)}  Issues: {len(bad)}"
        )
        print(f"Confidence: {round(confidence * 100, 2)}%  —  {'reliable' if not bad else 'not reliable'}")
        print("=" * 80)

        if not bad and not uncertain:
            print("No hallucinations found.\n")
            return

        for r in uncertain + bad:
            print("=" * 80)
            print(f"Claim #{r['claim_id']} | {r['status']} | score {round(r.get('score', 0), 3)}")
            print(f"\n{r['claim']}\n")
            if r.get("reason"):
                print(f"Reason: {r['reason']}")
            if r.get("ai_numbers"):
                print(f"AI numbers: {r['ai_numbers']}  Truth numbers: {r['truth_numbers']}")
            if r.get("unsupported_terms"):
                print(f"Unsupported terms: {', '.join(r['unsupported_terms'])}")
            if r.get("chunk_text"):
                print(f"\nClosest chunk ({r['matched_source']}, chunk {r['matched_chunk_id']}):")
                print(r["chunk_text"])
            print()
