"""
Hallucination detection module for TokenSmith using LettuceDetect.
"""

from typing import List, Dict, Any, Optional
from lettucedetect.models.inference import HallucinationDetector


class HallucinationDetectorWrapper:
    """
    Wrapper for LettuceDetect hallucination detection.
    """

    def __init__(self, model_path: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1", threshold: float = 0.1):
        """
        Initialize the hallucination detector.

        Args:
            model_path: Path to the LettuceDetect model on HuggingFace
            threshold: Threshold for considering answer as hallucinated (fraction of unsupported tokens)
        """
        self.detector = HallucinationDetector(
            method="transformer",
            model_path=model_path,
        )
        self.threshold = threshold

    def detect_hallucinations(self, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Detect hallucinations in the answer given the question and contexts.

        Args:
            question: The question asked
            answer: The generated answer
            contexts: List of context strings (retrieved chunks)

        Returns:
            Dict with 'is_hallucinated' (bool), 'unsupported_fraction' (float), and 'hallucinated_spans' (list)
        """
        try:
            # Get predictions from LettuceDetect
            predictions = self.detector.predict(
                context=contexts,
                question=question,
                answer=answer,
                output_format="spans"
            )

            # Calculate unsupported fraction
            total_answer_tokens = len(answer.split())  # Rough token count
            hallucinated_tokens = 0

            hallucinated_spans = []
            for pred in predictions:
                span_text = pred['text']
                span_tokens = len(span_text.split())
                print(span_text)
                hallucinated_tokens += span_tokens
                hallucinated_spans.append({
                    'text': span_text,
                    'confidence': pred['confidence'],
                    'start': pred['start'],
                    'end': pred['end']
                })

            unsupported_fraction = hallucinated_tokens / total_answer_tokens if total_answer_tokens > 0 else 0
            is_hallucinated = unsupported_fraction > self.threshold

            return {
                'is_hallucinated': is_hallucinated,
                'unsupported_fraction': unsupported_fraction,
                'hallucinated_spans': hallucinated_spans
            }

        except Exception as e:
            # If detection fails, assume no hallucinations to avoid blocking
            return {
                'is_hallucinated': False,
                'unsupported_fraction': 0.0,
                'hallucinated_spans': [],
                'error': str(e)
            }


def create_detector(model_path: Optional[str] = None, threshold: float = 0.1) -> HallucinationDetectorWrapper:
    """
    Factory function to create hallucination detector.

    Args:
        model_path: Path to model, defaults to base English model
        threshold: Detection threshold

    Returns:
        Configured HallucinationDetectorWrapper
    """
    if model_path is None:
        model_path = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

    return HallucinationDetectorWrapper(model_path=model_path, threshold=threshold)