

import json
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
from feedback_db import FeedbackDB

@dataclass
class AnalysisResult:
    query_patterns: List[Dict]
    common_issues: List[Dict]
    improvement_suggestions: List[str]
    system_metrics: Dict
    priority_actions: List[str]

class FeedbackAnalyzer:
    
    def __init__(self, feedback_db: FeedbackDB):
        self.db = feedback_db
    
    def analyze_feedback(self) -> AnalysisResult:
        all_feedback = self.db.get_recent_feedback(limit=1000)
        negative_feedback = self.db.get_negative_feedback()
        
        query_patterns = self._analyze_query_patterns(all_feedback)
        common_issues = self._analyze_common_issues(negative_feedback)
        improvement_suggestions = self._generate_improvement_suggestions(all_feedback, negative_feedback)
        system_metrics = self._calculate_system_metrics(all_feedback)
        priority_actions = self._identify_priority_actions(negative_feedback, query_patterns)
        
        return AnalysisResult(
            query_patterns=query_patterns,
            common_issues=common_issues,
            improvement_suggestions=improvement_suggestions,
            system_metrics=system_metrics,
            priority_actions=priority_actions
        )
    
    def _analyze_query_patterns(self, feedback: List[Dict]) -> List[Dict]:
        query_stats = defaultdict(lambda: {'total': 0, 'positive': 0, 'ratings': []})
        
        for fb in feedback:
            query = fb['query'].lower().strip()
            query_stats[query]['total'] += 1
            
            if fb['thumbs_up']:
                query_stats[query]['positive'] += 1
            
            if fb['rating']:
                query_stats[query]['ratings'].append(fb['rating'])
        
        patterns = []
        for query, stats in query_stats.items():
            if stats['total'] >= 2:
                success_rate = stats['positive'] / stats['total']
                avg_rating = sum(stats['ratings']) / len(stats['ratings']) if stats['ratings'] else 0
                
                patterns.append({
                    'query': query,
                    'frequency': stats['total'],
                    'success_rate': success_rate,
                    'avg_rating': avg_rating,
                    'needs_attention': success_rate < 0.5 or avg_rating < 3.0
                })
        
        patterns.sort(key=lambda x: (x['needs_attention'], -x['frequency']), reverse=True)
        return patterns[:10]
    
    def _analyze_common_issues(self, negative_feedback: List[Dict]) -> List[Dict]:
        issue_keywords = {
            'incomplete': ['incomplete', 'partial', 'missing', 'unclear'],
            'irrelevant': ['irrelevant', 'wrong', 'off-topic', 'not related'],
            'inaccurate': ['incorrect', 'wrong', 'false', 'inaccurate', 'mistake'],
            'confusing': ['confusing', 'unclear', 'hard to understand', 'complex'],
            'too_short': ['too short', 'brief', 'insufficient', 'not enough'],
            'too_long': ['too long', 'verbose', 'wordy', 'rambling'],
            'technical': ['technical', 'jargon', 'complex terms', 'difficult']
        }
        
        issue_counts = Counter()
        issue_examples = defaultdict(list)
        
        for fb in negative_feedback:
            if fb['comment']:
                comment = fb['comment'].lower()
                
                for issue_type, keywords in issue_keywords.items():
                    if any(keyword in comment for keyword in keywords):
                        issue_counts[issue_type] += 1
                        if len(issue_examples[issue_type]) < 3:
                            issue_examples[issue_type].append({
                                'query': fb['query'][:50] + '...',
                                'comment': fb['comment'][:100] + '...'
                            })
        
        common_issues = []
        for issue_type, count in issue_counts.most_common():
            common_issues.append({
                'type': issue_type,
                'count': count,
                'examples': issue_examples[issue_type]
            })
        
        return common_issues
    
    def _generate_improvement_suggestions(self, all_feedback: List[Dict], 
                                        negative_feedback: List[Dict]) -> List[str]:
        suggestions = []

        total_feedback = len(all_feedback)
        positive_feedback = sum(1 for fb in all_feedback if fb['thumbs_up'])
        success_rate = positive_feedback / total_feedback if total_feedback > 0 else 0
        
        if success_rate < 0.7:
            suggestions.append(f"Overall success rate is {success_rate:.1%}. Consider reviewing retrieval and generation parameters.")

        ratings = [fb['rating'] for fb in all_feedback if fb['rating']]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            if avg_rating < 3.5:
                suggestions.append(f"Average rating is {avg_rating:.1f}/5. Focus on answer quality and relevance.")

        comment_themes = self._extract_comment_themes(negative_feedback)
        for theme, count in comment_themes.most_common(3):
            if theme == 'incomplete':
                suggestions.append("Users frequently mention incomplete answers. Consider increasing max_tokens or improving context retrieval.")
            elif theme == 'irrelevant':
                suggestions.append("Irrelevant answers are common. Review embedding model and retrieval parameters.")
            elif theme == 'confusing':
                suggestions.append("Answers are often confusing. Consider simplifying the prompt template.")

        complex_queries = self._identify_complex_queries(all_feedback)
        if complex_queries:
            suggestions.append(f"Found {len(complex_queries)} complex queries with low success rates. Consider implementing query decomposition.")
        
        return suggestions
    
    def _calculate_system_metrics(self, feedback: List[Dict]) -> Dict:
        if not feedback:
            return {}
        
        total = len(feedback)
        positive = sum(1 for fb in feedback if fb['thumbs_up'])
        ratings = [fb['rating'] for fb in feedback if fb['rating']]
        comments = [fb for fb in feedback if fb['comment']]
        
        metrics = {
            'total_interactions': total,
            'success_rate': positive / total if total > 0 else 0,
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0,
            'comment_rate': len(comments) / total if total > 0 else 0,
            'engagement_score': (positive + len(comments)) / total if total > 0 else 0
        }
        
        return metrics
    
    def _identify_priority_actions(self, negative_feedback: List[Dict], 
                                 query_patterns: List[Dict]) -> List[str]:
        actions = []

        problematic_queries = [qp for qp in query_patterns if qp['needs_attention'] and qp['frequency'] >= 3]
        if problematic_queries:
            actions.append(f"Address {len(problematic_queries)} frequently asked questions with low success rates")

        recent_negative = [fb for fb in negative_feedback if fb.get('created_at')]
        if len(recent_negative) > 5:
            actions.append("Review recent negative feedback for immediate issues")

        comment_themes = self._extract_comment_themes(negative_feedback)
        if comment_themes:
            top_theme = comment_themes.most_common(1)[0]
            actions.append(f"Focus on addressing '{top_theme[0]}' issues (mentioned {top_theme[1]} times)")
        
        return actions
    
    def _extract_comment_themes(self, negative_feedback: List[Dict]) -> Counter:
        themes = Counter()
        
        theme_keywords = {
            'incomplete': ['incomplete', 'partial', 'missing', 'unclear', 'insufficient'],
            'irrelevant': ['irrelevant', 'wrong', 'off-topic', 'not related', 'doesn\'t answer'],
            'inaccurate': ['incorrect', 'wrong', 'false', 'inaccurate', 'mistake', 'error'],
            'confusing': ['confusing', 'unclear', 'hard to understand', 'complex', 'unclear'],
            'too_short': ['too short', 'brief', 'insufficient', 'not enough detail'],
            'too_long': ['too long', 'verbose', 'wordy', 'rambling', 'too much'],
            'technical': ['technical', 'jargon', 'complex terms', 'difficult', 'advanced']
        }
        
        for fb in negative_feedback:
            if fb['comment']:
                comment = fb['comment'].lower()
                for theme, keywords in theme_keywords.items():
                    if any(keyword in comment for keyword in keywords):
                        themes[theme] += 1
        
        return themes
    
    def _identify_complex_queries(self, feedback: List[Dict]) -> List[Dict]:
        complex_queries = []
        
        for fb in feedback:
            query = fb['query']
            is_complex = (
                len(query.split()) > 10 or
                '?' in query and query.count('?') > 1 or
                any(word in query.lower() for word in ['compare', 'difference', 'relationship', 'explain']) or
                fb['thumbs_up'] == False and fb.get('rating', 5) < 3
            )
            
            if is_complex:
                complex_queries.append(fb)
        
        return complex_queries
    
    def generate_improvement_report(self) -> str:
        analysis = self.analyze_feedback()
        
        report = []
        report.append("TOKENSMITH FEEDBACK ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append("\n SYSTEM METRICS")
        report.append("-" * 30)
        for metric, value in analysis.system_metrics.items():
            if isinstance(value, float):
                report.append(f"{metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                report.append(f"{metric.replace('_', ' ').title()}: {value}")
        
        if analysis.query_patterns:
            report.append("\n QUERY PATTERNS NEEDING ATTENTION")
            report.append("-" * 40)
            for pattern in analysis.query_patterns[:5]:
                report.append(f"• '{pattern['query'][:50]}...'")
                report.append(f"  Frequency: {pattern['frequency']}, Success: {pattern['success_rate']:.1%}")
        
        if analysis.common_issues:
            report.append("\n  COMMON ISSUES")
            report.append("-" * 20)
            for issue in analysis.common_issues[:3]:
                report.append(f"• {issue['type'].replace('_', ' ').title()}: {issue['count']} occurrences")
        
        if analysis.improvement_suggestions:
            report.append("\n IMPROVEMENT SUGGESTIONS")
            report.append("-" * 30)
            for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
                report.append(f"{i}. {suggestion}")
        
        if analysis.priority_actions:
            report.append("\n PRIORITY ACTIONS")
            report.append("-" * 20)
            for i, action in enumerate(analysis.priority_actions, 1):
                report.append(f"{i}. {action}")
        
        return "\n".join(report)
