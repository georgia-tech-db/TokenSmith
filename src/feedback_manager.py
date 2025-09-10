"""
Simple Feedback Manager Module

This module handles basic feedback collection and storage:
- Collect thumbs up/down, rating, and comments
- Save feedback to database
"""

import json
import uuid
from typing import Optional
from src.feedback_db import FeedbackDB, FeedbackEntry


class FeedbackManager:
    """Simple manager for basic feedback collection and storage."""
    
    def __init__(self, feedback_db: Optional[FeedbackDB] = None):
        """Initialize the feedback manager with an optional database instance."""
        self.feedback_db = feedback_db or FeedbackDB()
        self.session_id = str(uuid.uuid4())
    
    def collect_feedback(self, query: str, answer: str, retrieved_chunks: list, 
                        prompt_style: str = "default") -> Optional[int]:
        """
        Collect basic feedback from the user for a given query-answer pair.
        
        Args:
            query: The user's query
            answer: The system's answer
            retrieved_chunks: List of retrieved chunks used for the answer
            prompt_style: The prompt style used for generation
            
        Returns:
            Feedback ID if feedback was collected, None otherwise
        """
        print("\nHow was this answer? (Press Enter to skip)")
        
        thumbs_feedback = input("Thumbs up (y) or thumbs down (n): ").strip().lower()
        thumbs_up = None
        if thumbs_feedback in ['y', 'yes']:
            thumbs_up = True
        elif thumbs_feedback in ['n', 'no']:
            thumbs_up = False
        
        rating = None
        if thumbs_up is not None:
            try:
                rating_input = input("Rating (1-5, Enter to skip): ").strip()
                if rating_input:
                    rating = int(rating_input)
                    if not 1 <= rating <= 5:
                        rating = None
            except ValueError:
                pass
        
        comment = input("Any comments or suggestions? (Enter to skip): ").strip()
        
        if thumbs_up is not None or comment:
            feedback = FeedbackEntry(
                query=query,
                answer=answer,
                retrieved_chunks=json.dumps(retrieved_chunks),
                thumbs_up=thumbs_up,
                rating=rating,
                comment=comment,
                session_id=self.session_id,
                prompt_style=prompt_style
            )
            feedback_id = self.feedback_db.add_feedback(feedback)
            print(f"Feedback recorded (ID: {feedback_id})")
            return feedback_id
        else:
            print("Feedback skipped")
            return None
