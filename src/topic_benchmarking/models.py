import logging
from typing import List, Tuple

# Gensim imports
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

# Scikit-Learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# BERTopic import
# We import inside the method or carefully to avoid errors if not installed yet
# but here we'll assume it's available since it's the target environment.

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.download("stopwords", quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK stopwords: {e}")


class LDAModel:
    """
    Gensim implementation of LDA.
    """

    requires_clean_text = True

    def __init__(self, num_topics: int = 10, passes: int = 10):
        self.num_topics = num_topics
        self.passes = passes

        self.stop_words = set(stopwords.words("english"))
        self.topic_weights = []

    def _preprocess(
        self, texts: List[str]
    ) -> Tuple[List[List[str]], Dictionary, List[List[Tuple[int, int]]]]:
        """
        Tokenizes, removes stop words, and creates a Dictionary and Corpus.
        """
        tokenized_docs = []
        for text in texts:
            # simple_preprocess lowercases, tokenizes, and removes punctuation
            tokens = [
                token
                for token in simple_preprocess(text)
                if token not in self.stop_words
            ]
            tokenized_docs.append(tokens)

        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        return tokenized_docs, dictionary, corpus

    def extract_topics(self, texts: List[str]) -> Tuple[List[List[str]], List[int]]:
        """
        Extracts top 10 words per topic and assigns a topic to each text.
        """
        _, dictionary, corpus = self._preprocess(texts)

        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=42,
        )

        extracted_topics = []
        self.topic_weights = []

        # Extract topics and weights
        for i in range(self.num_topics):
            # show_topic returns a list of (word, weight) tuples
            topic_terms = lda_model.show_topic(i, topn=10)
            extracted_topics.append([term[0] for term in topic_terms])
            self.topic_weights.append([term[1] for term in topic_terms])

        # Assignments
        assignments = []
        for bow in corpus:
            probs = lda_model.get_document_topics(bow)
            if probs:
                best_topic = max(probs, key=lambda x: x[1])[0]
                assignments.append(int(best_topic))
            else:
                assignments.append(-1)

        return extracted_topics, assignments


class NMFModel:
    """
    Scikit-Learn implementation of NMF.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.requires_clean_text = True
        self.topic_weights = []

    def extract_topics(self, texts: List[str]) -> Tuple[List[List[str]], List[int]]:
        """
        Uses TfidfVectorizer and NMF to extract topics and assign them to texts.
        """
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)

        nmf = NMF(n_components=self.n_components, random_state=42, init="nndsvd").fit(
            tfidf
        )

        feature_names = vectorizer.get_feature_names_out()
        extracted_topics = []
        self.topic_weights = []

        for topic_idx, topic in enumerate(nmf.components_):
            # Get indices of top 10 words
            top_indices = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_indices]
            extracted_topics.append(top_words)
            self.topic_weights.append([topic[i] for i in top_indices])

        # Assignments
        W = nmf.transform(tfidf)
        assignments = W.argmax(axis=1).tolist()

        return extracted_topics, assignments


class BERTopicModel:
    """
    BERTopic implementation.
    """

    def __init__(self, nr_topics: int = 10):
        self.nr_topics = nr_topics
        self.requires_clean_text = False  # BERTopic handles raw text
        self.topic_weights = []

    def extract_topics(self, texts: List[str]) -> Tuple[List[List[str]], List[int]]:
        """
        Uses BERTopic to extract topics and assign them to texts.
        """
        from bertopic import BERTopic

        # Initialize BERTopic with a standard embedding model and fixed number of topics
        topic_model = BERTopic(
            nr_topics=self.nr_topics, embedding_model="all-MiniLM-L6-v2"
        )

        raw_topics, _ = topic_model.fit_transform(texts)
        all_topics = topic_model.get_topics()

        # Map raw topic IDs to sequential indices (excluding -1)
        topic_id_to_idx = {}
        valid_topic_ids = sorted([tid for tid in all_topics.keys() if tid != -1])
        for idx, topic_id in enumerate(valid_topic_ids):
            topic_id_to_idx[topic_id] = idx

        assignments = [topic_id_to_idx.get(tid, -1) for tid in raw_topics]

        # Ensure extracted_topics matches the order of topic_id_to_idx
        ordered_extracted_topics = []
        ordered_topic_weights = []
        for tid in valid_topic_ids:
            words_with_weights = all_topics[tid]
            ordered_extracted_topics.append([ww[0] for ww in words_with_weights])
            ordered_topic_weights.append([ww[1] for ww in words_with_weights])

        self.topic_weights = ordered_topic_weights
        return ordered_extracted_topics, assignments


class SLMTopicModel:
    """
    Small Language Model (SLM) implementation using llama-cpp-python.
    """

    def __init__(
        self,
        model_path: str = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        n_threads: int = 8,
    ):
        self.model_path = model_path
        self.n_threads = n_threads
        self.requires_clean_text = False
        self.topic_weights = []

    def extract_topics(self, texts: List[str]) -> Tuple[List[List[str]], List[int]]:
        """
        Uses an LLM via llama-cpp-python to extract topics.
        """
        from llama_cpp import Llama
        import json
        import re

        llm = Llama(
            model_path=self.model_path,
            n_threads=self.n_threads,
            n_ctx=4096,
            verbose=False,
        )

        # Batch texts to fit context window. We'll sample/truncate for the benchmarking engine
        # As topic extraction usually works on a corpus, we feed a representative sample.
        # We limit the total characters to roughly 12000 (~3000-4000 tokens) to avoid context overflow.
        limited_texts = []
        char_count = 0
        for t in texts:
            if char_count + len(t) > 12000:
                break
            limited_texts.append(t)
            char_count += len(t)

        sample_text = "\n---\n".join(limited_texts)

        prompt = f"""<|im_start|>system
You are a topic extraction expert. Analyze the following documents and identify the 10 most distinct topics.
For each topic, provide exactly 10 descriptive keywords.
Return the result as a raw JSON list of lists, where each inner list contains 10 strings.
Example: [["word1", "word2", ...], ["word1", "word2", ...]]
Do not include any other text in your response.<|im_end|>
<|im_start|>user
Documents:
{sample_text}
<|im_end|>
<|im_start|>assistant
"""

        output = llm(prompt, max_tokens=1024, temperature=0.0, stop=["<|im_end|>"])

        response_text = output["choices"][0]["text"].strip()
        logger.info(f"Raw SLM response: {response_text}")

        try:
            # Try to parse JSON from the response
            # First, try to find any list of lists in the text
            json_match = re.search(
                r"(\[[\s\n]*\[.*\][\s\n]*\])", response_text, re.DOTALL
            )
            if json_match:
                extracted_topics = json.loads(json_match.group(1))
            else:
                # If no list of lists found, try parsing the whole response if it's JSON
                extracted_topics = json.loads(response_text)

            # Final validation: Ensure it's a list of lists of strings
            if not isinstance(extracted_topics, list) or not all(
                isinstance(t, list) for t in extracted_topics
            ):
                raise ValueError("Output is not a list of lists")

        except Exception as e:
            logger.error(
                f"Failed to parse LLM response: {e}. Raw response: {response_text}"
            )
            # Fallback: dummy results to prevent complete failure
            extracted_topics = [["parsing_error"] * 10 for _ in range(10)]

        return extracted_topics
