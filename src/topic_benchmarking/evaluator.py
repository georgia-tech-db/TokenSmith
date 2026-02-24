import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import spacy
from nltk.corpus import stopwords
import nltk
import re
import seaborn as sns


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TopicEvaluator:
    """
    Benchmarking engine for comparing multiple topic extraction models.
    """

    def __init__(self, model_classes: List[Any], corpus: List[str], N: int = 3):
        self.model_classes = model_classes
        self.corpus = corpus
        self.N = N
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.stop_words = set(stopwords.words("english"))

        # Preprocess corpus for "Cleaned" version
        self.cleaned_corpus = self._preprocess_corpus(corpus)
        # Prepare tokenized version for coherence calculation
        self.tokenized_corpus = [doc.split() for doc in self.cleaned_corpus]
        self.dictionary = Dictionary(self.tokenized_corpus)

    def _preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        Cleans the corpus: removes stop words, lemmatizes, and keeps only alphabetic tokens.
        """
        cleaned_texts = []
        for text in texts:
            # Lowercase and remove non-alphabetic characters
            text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
            doc = self.nlp(text)
            # Lemmatize and remove stop words
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and token.lemma_ not in self.stop_words
                and len(token.lemma_) > 2
            ]
            cleaned_texts.append(" ".join(tokens))
        return cleaned_texts

    def calculate_metrics(
        self, topics: List[List[str]], model_instance: Any
    ) -> Dict[str, float]:
        """
        Computes various metrics for the extracted topics.
        """
        metrics = {}

        # 1. Topic Coherence (Cv)
        try:
            cm = CoherenceModel(
                topics=topics,
                texts=self.tokenized_corpus,
                dictionary=self.dictionary,
                coherence="c_v",
            )
            metrics["coherence"] = cm.get_coherence()
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            metrics["coherence"] = 0.0

        # 2. Topic Diversity
        unique_words = set()
        total_words = 0
        for topic in topics:
            unique_words.update(topic)
            total_words += len(topic)
        metrics["diversity"] = (
            len(unique_words) / total_words if total_words > 0 else 0.0
        )

        # 3. Topic Redundancy (Average Jaccard Similarity)
        jaccard_sims = []
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                set_i = set(topics[i])
                set_j = set(topics[j])
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                jaccard_sims.append(intersection / union if union > 0 else 0.0)
        metrics["redundancy"] = np.mean(jaccard_sims) if jaccard_sims else 0.0

        # 4. Density Score (Weight Concentration)
        # Assuming the model instance might have weights if supported
        # If the model provides weights, it should be accessible via an attribute or returned by extract_topics
        # Since the interface only guarantees List[List[str]], we look for an optional attribute
        # 'topic_weights' which could be List[List[float]]
        if hasattr(model_instance, "topic_weights") and model_instance.topic_weights:
            densities = []
            for weights in model_instance.topic_weights:
                if len(weights) >= 10:
                    top_3 = sum(sorted(weights, reverse=True)[:3])
                    top_10 = sum(sorted(weights, reverse=True)[:10])
                    densities.append(top_3 / top_10 if top_10 > 0 else 0.0)
            metrics["density"] = np.mean(densities) if densities else 0.0
        else:
            metrics["density"] = 0.0

        return metrics

    def save_model_results(
        self, model_name: str, topics: List[List[str]], assignments: List[int]
    ):
        """
        Saves the extracted topics and assignments for a specific model to a JSON file.
        """
        output_path = f"{model_name}_results.json"
        results = []

        for i, topic_idx in enumerate(assignments):
            results.append(
                {
                    "document_index": i,
                    "content_preview": self.corpus[i][:200] + "..."
                    if len(self.corpus[i]) > 200
                    else self.corpus[i],
                    "assigned_topic_id": int(topic_idx),
                    "topic_keywords": topics[topic_idx] if topic_idx != -1 else [],
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "results": results,
                    "topic_definitions": topics,
                },
                f,
                indent=4,
            )
        logger.info(f"Saved results for {model_name} to {output_path}")

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs the benchmark for all models.
        """
        results = []
        for model_class in self.model_classes:
            model_name = model_class.__name__
            logger.info(f"Evaluating model: {model_name}")

            # Instantiate model
            try:
                model = model_class()
            except Exception as e:
                logger.error(f"Failed to instantiate model {model_name}: {e}")
                continue

            requires_clean = getattr(model, "requires_clean_text", False)
            input_corpus = self.cleaned_corpus if requires_clean else self.corpus

            run_coherences = []
            run_runtimes = []
            last_run_metrics = {}
            num_topics = 0

            for i in range(self.N):
                start_time = time.time()
                try:
                    topics, assignments = model.extract_topics(input_corpus)
                    runtime = time.time() - start_time

                    current_metrics = self.calculate_metrics(topics, model)
                    run_coherences.append(current_metrics["coherence"])
                    run_runtimes.append(runtime)

                    if i == self.N - 1:  # Capture last run
                        last_run_metrics = current_metrics
                        num_topics = len(topics)
                        self.save_model_results(model_name, topics, assignments)

                except Exception as e:
                    logger.error(f"Error during run {i + 1} of model {model_name}: {e}")
                    continue

            if run_coherences:
                results.append(
                    {
                        "Model Name": model_name,
                        "Mean Coherence": np.mean(run_coherences),
                        # "Coherence StdDev": np.std(run_coherences),
                        "Diversity": last_run_metrics.get("diversity", 0.0),
                        "Redundancy": last_run_metrics.get("redundancy", 0.0),
                        # "Density": last_run_metrics.get("density", 0.0),
                        "Runtime": np.mean(run_runtimes),
                        # "Optimal K": num_topics,
                    }
                )

        return pd.DataFrame(results)

    def plot_radar_chart(self, df: pd.DataFrame, output_path: str = "radar_chart.png"):
        """
        Generates a radar chart comparing the top 3 models based on Mean Coherence.
        """
        if df.empty:
            logger.warning("DataFrame is empty. Cannot plot radar chart.")
            return

        # Select top 3 models by Mean Coherence
        top_df = df.sort_values(by="Mean Coherence", ascending=False).head(3).copy()

        # Standardize metrics for radar chart (0-1 scale)
        metrics_to_plot = ["Mean Coherence", "Diversity", "Density"]
        # Redundancy and Runtime (lower is better, so maybe invert or just plot as is)
        # For simplicity in this benchmark, let's use the positive metrics

        # Radar charts need the first point repeated at the end
        labels = metrics_to_plot
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx, row in top_df.iterrows():
            values = row[metrics_to_plot].values.flatten().tolist()
            # Normalize for plotting if needed, but coherence and diversity are already 0-1 usually
            values += values[:1]
            ax.plot(
                angles, values, linewidth=1, linestyle="solid", label=row["Model Name"]
            )
            ax.fill(angles, values, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.savefig(output_path)
        logger.info(f"Radar chart saved to {output_path}")
        plt.close()


if __name__ == "__main__":
    from src.topic_benchmarking.models import (
        LDAModel,
        NMFModel,
        BERTopicModel,
        SLMTopicModel,
    )

    sns.set_theme(style="whitegrid")
    nltk.download("stopwords", quiet=True)

    # Load real textbook data if available, otherwise use sample
    corpus_path = "/home/user/GitHub/TokenSmith/data/extracted_sections.json"
    import json

    with open(corpus_path, "r", encoding="utf-8") as f:
        sections = json.load(f)
        corpus = [section["content"] for section in sections]
        logger.info(f"Loaded {len(corpus)} sections from {corpus_path}")

    evaluator = TopicEvaluator([NMFModel], corpus)
    df = evaluator.run_benchmark()
    print("\nBenchmark Results:")
    print(df)
    evaluator.plot_radar_chart(df)
