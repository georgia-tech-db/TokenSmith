import os
import warnings
from typing import List, Optional
from tests.metrics.base import MetricBase
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIEntailmentMetric(MetricBase):
    """NLI-based entailment metric using DeBERTa model."""
    
    def __init__(self):
        self._pipeline = None
        self._available = self._initialize()
    
    @property
    def name(self) -> str:
        return "nli"
    
    @property
    def weight(self) -> float:
        return 1.0
    
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
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """ Calculate NLI entailment score between answer and expected text."""
    
        if not self.is_available():
            return 0.0
        
        if not answer.strip() or not expected.strip():
            return 0.0
        
        try:
            # Format input for NLI: premise (expected) and hypothesis (answer)
            input = self._tokenizer(expected, answer, truncation=True, return_tensors="pt")
            output = self._model(input["input_ids"].to('cpu'))
            
            # Calculate entailment score
            prediction = torch.softmax(output["logits"][0], -1).tolist()
            label_names = ["entailment", "neutral", "contradiction"]
            prediction = {name: pred for pred, name in zip(prediction, label_names)}
            
            # Weighted scoring
            final_score = (
                prediction['entailment'] * 1.0 +
                prediction['neutral'] * 0.5 +
                prediction['contradiction'] * -1.0
            )
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            print(f"NLI calculation failed: {e}")
            return 0.0
    