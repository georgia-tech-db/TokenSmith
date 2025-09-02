

import json
from typing import Dict, List, Optional
from feedback_db import FeedbackDB
from collections import Counter

class PromptSelector:
    
    def __init__(self, feedback_db: FeedbackDB):
        self.db = feedback_db
        self.prompt_styles = ["default", "detailed", "simple", "focused"]
    
    def select_prompt_style(self, query: str) -> str:
        # Get feedback for similar queries
        similar_feedback = self._get_similar_query_feedback(query)
        
        if not similar_feedback:
            return "default"

        issues = self._analyze_feedback_issues(similar_feedback)
        
        return self._choose_style_for_issues(issues)
    
    def _get_similar_query_feedback(self, query: str) -> List[Dict]:
        all_feedback = self.db.get_recent_feedback(limit=1000)
        query_words = set(query.lower().split())
        similar_feedback = []
        
        for fb in all_feedback:
            fb_words = set(fb['query'].lower().split())

            overlap = len(query_words.intersection(fb_words))
            if overlap >= max(1, len(query_words) * 0.5):
                similar_feedback.append(fb)
        
        return similar_feedback
    
    def _analyze_feedback_issues(self, feedback: List[Dict]) -> Dict[str, int]:
        issues = Counter()
        
        for fb in feedback:
            if fb['comment']:
                comment = fb['comment'].lower()
                
                if any(word in comment for word in ['incomplete', 'partial', 'missing', 'insufficient']):
                    issues['incomplete'] += 1
                if any(word in comment for word in ['confusing', 'unclear', 'hard to understand', 'complex']):
                    issues['confusing'] += 1
                if any(word in comment for word in ['irrelevant', 'off-topic', 'not related']):
                    issues['irrelevant'] += 1
                if any(word in comment for word in ['too short', 'brief', 'not enough']):
                    issues['too_short'] += 1
                if any(word in comment for word in ['too long', 'verbose', 'rambling']):
                    issues['too_long'] += 1
                if any(word in comment for word in ['technical', 'jargon', 'difficult']):
                    issues['too_technical'] += 1
            
            if fb['rating'] and fb['rating'] < 3:
                issues['low_rating'] += 1
            if fb['thumbs_up'] is False:
                issues['thumbs_down'] += 1
        
        return dict(issues)
    
    def _choose_style_for_issues(self, issues: Dict[str, int]) -> str:
        if not issues:
            return "default"
        
        most_common_issue = max(issues.items(), key=lambda x: x[1])[0]
        
        issue_to_style = {
            'incomplete': 'detailed',
            'too_short': 'detailed',
            'confusing': 'simple',
            'too_technical': 'simple',
            'irrelevant': 'focused',
            'too_long': 'focused',
            'low_rating': 'detailed',
            'thumbs_down': 'detailed'
        }
        
        return issue_to_style.get(most_common_issue, "default")
    
    def get_prompt_style_stats(self) -> Dict[str, Dict]:
        all_feedback = self.db.get_recent_feedback(limit=1000)
        
        stats = {}
        for style in self.prompt_styles:
            stats[style] = {
                'usage_count': 0,
                'success_rate': 0.0,
                'avg_rating': 0.0
            }
        
        return stats
    
    def suggest_prompt_improvements(self) -> List[str]:
        suggestions = []
        
        all_feedback = self.db.get_recent_feedback(limit=100)
        issues = self._analyze_feedback_issues(all_feedback)
        
        if issues.get('incomplete', 0) > 3:
            suggestions.append("Consider using 'detailed' prompt style more often for comprehensive answers")
        
        if issues.get('confusing', 0) > 3:
            suggestions.append("Consider using 'simple' prompt style more often for clarity")
        
        if issues.get('irrelevant', 0) > 3:
            suggestions.append("Consider using 'focused' prompt style more often for relevance")
        
        return suggestions

_prompt_selector = None

def get_prompt_selector() -> PromptSelector:
    global _prompt_selector
    if _prompt_selector is None:
        _prompt_selector = PromptSelector(FeedbackDB())
    return _prompt_selector

def select_prompt_style_for_query(query: str) -> str:
    selector = get_prompt_selector()
    return selector.select_prompt_style(query)
