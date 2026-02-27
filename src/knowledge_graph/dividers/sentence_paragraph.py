from typing import List

import spacy

from src.knowledge_graph.dividers import BaseDivider
from src.knowledge_graph.models import Chunk


class SentenceParagraphDivider(BaseDivider):
    """Split text into chunks by paragraph, then by sentence if needed.

    The divider first splits on double-newline paragraph boundaries, then
    uses spaCy sentence segmentation to further split paragraphs that
    exceed *max_tokens*.

    Args:
        spacy_model: Name of the spaCy model to load.
        max_tokens: Maximum number of whitespace-delimited tokens per chunk.
            Paragraphs exceeding this are split into sentence groups.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", max_tokens: int = 256):
        self.max_tokens = max_tokens
        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def _token_count(text: str) -> int:
        """Count whitespace-delimited tokens in *text*."""
        return len(text.split())

    def divide(self, text: str) -> List[Chunk]:
        """Split *text* into semantically coherent chunks.

        1. Split on paragraph boundaries (``\\n\\n``).
        2. If a paragraph exceeds *max_tokens*, further split it into
           sentence groups that stay within the limit.
        3. Assign sequential chunk IDs starting at 0.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[Chunk] = []
        chunk_id = 0

        for paragraph in paragraphs:
            if self._token_count(paragraph) <= self.max_tokens:
                chunks.append(Chunk(id=chunk_id, text=paragraph))
                chunk_id += 1
            else:
                # Use spaCy to split into sentences, then group them
                doc = self.nlp(paragraph)
                current_sentences: List[str] = []
                current_token_count = 0

                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    sent_tokens = self._token_count(sent_text)

                    if (
                        current_token_count + sent_tokens > self.max_tokens
                        and current_sentences
                    ):
                        # Flush the current group
                        chunks.append(
                            Chunk(
                                id=chunk_id,
                                text=" ".join(current_sentences),
                            )
                        )
                        chunk_id += 1
                        current_sentences = []
                        current_token_count = 0

                    current_sentences.append(sent_text)
                    current_token_count += sent_tokens

                # Flush remaining sentences
                if current_sentences:
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=" ".join(current_sentences),
                        )
                    )
                    chunk_id += 1

        return chunks
