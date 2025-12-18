import os
import warnings
from typing import List, Optional
from tests.metrics.base import MetricBase
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIEntailmentMetric(MetricBase):
    """
    NLI-based faithfulness metric using DeBERTa model.

    Measures whether the generated answer is grounded in (entailed by)
    the retrieved chunks. This is a critical RAG metric for detecting
    hallucinations and ensuring answers are faithful to source material.
    """

    def __init__(self):
        self._pipeline = None
        self._available = self._initialize()
    
    @property
    def name(self) -> str:
        return "nli"
    
    @property
    def weight(self) -> float:
        return 0.4  # High weight for faithfulness
    
    def _initialize(self) -> bool:
        """Initialize the NLI pipeline with the best available model."""
        try:
            # Suppress CUDA warnings if running on CPU
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            
            model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # print(f"NLI metric initialized with model: {model_name}")
            return True
            
        except Exception as e:
            print(f"NLI metric initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if NLI pipeline is available."""
        return self._available
    
    def calculate(self, answer: str, chunks: Optional[List[str]] = None, expected: str = "", **kwargs) -> float:
        """
        Calculate NLI faithfulness score by checking if chunks entail the answer.

        This metric checks whether the generated answer is supported by (entailed by)
        the retrieved chunks. A high score means the answer is grounded in the
        source material, while a low score indicates potential hallucination.

        Args:
            answer: Generated answer to evaluate
            chunks: Retrieved chunks to check for entailment
            expected: Fallback if no chunks provided
            **kwargs: Ignored (keywords, question not used)

        Returns:
            Maximum entailment score across all chunks (0.0 to 1.0)
        """
        if not self.is_available():
            return 0.0
        
        if not answer.strip() or not expected.strip():
            return 0.0

        # If no chunks provided, fallback to checking expected answer
        if not chunks or len(chunks) == 0:
            if not expected.strip():
                return 0.0
            chunks = [expected]

        try:
            max_entailment_score = 0.0

            # Check if ANY chunk entails the answer
            for chunk in chunks:
                if not chunk.strip():
                    continue

                # Format input for NLI: premise (chunk) and hypothesis (answer)
                # We check: "Does this chunk entail/support the answer?"
                input_ids = self._tokenizer(
                    chunk,
                    answer,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                output = self._model(input_ids["input_ids"].to('cpu'))

                # Calculate entailment score
                prediction = torch.softmax(output["logits"][0], -1).tolist()
                label_names = ["entailment", "neutral", "contradiction"]
                prediction_dict = {name: pred for pred, name in zip(prediction, label_names)}

                # Weighted scoring: entailment is positive, neutral is slightly positive,
                # contradiction is negative
                chunk_score = (
                    prediction_dict['entailment'] * 1.0 +
                    prediction_dict['neutral'] * 0.3 +
                    prediction_dict['contradiction'] * 0.0
                )

                # Take the maximum score across all chunks
                # If ANY chunk strongly entails the answer, it's grounded
                max_entailment_score = max(max_entailment_score, chunk_score)

            return min(max(max_entailment_score, 0.0), 1.0)

        except Exception as e:
            print(f"NLI faithfulness calculation failed: {e}")
            return 0.0
    