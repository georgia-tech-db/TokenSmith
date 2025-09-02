

import json
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from feedback_db import FeedbackDB
from feedback_analyzer import FeedbackAnalyzer

@dataclass
class ImprovementAction:
    action_type: str
    description: str
    parameter: str
    old_value: any
    new_value: any
    confidence: float
    reasoning: str

class SystemImprover:
    
    def __init__(self, feedback_db: FeedbackDB, config_path: str = "config.yaml"):
        self.db = feedback_db
        self.config_path = config_path
        self.analyzer = FeedbackAnalyzer(feedback_db)
    
    def analyze_and_improve(self) -> List[ImprovementAction]:
        analysis = self.analyzer.analyze_feedback()
        improvements = []
        

        improvements.extend(self._improve_retrieval_parameters(analysis))
        improvements.extend(self._improve_generation_parameters(analysis))
        improvements.extend(self._improve_prompt_template(analysis))
        improvements.extend(self._suggest_knowledge_base_expansion(analysis))
        
        return improvements
    
    def _improve_retrieval_parameters(self, analysis) -> List[ImprovementAction]:
        improvements = []
        

        success_rate = analysis.system_metrics.get('success_rate', 0)
        avg_rating = analysis.system_metrics.get('avg_rating', 0)
        
        if success_rate < 0.6:

            improvements.append(ImprovementAction(
                action_type="retrieval",
                description="Increase retrieval count for better coverage",
                parameter="top_k",
                old_value=8,
                new_value=12,
                confidence=0.7,
                reasoning=f"Success rate is {success_rate:.1%}, increasing top_k may improve answer quality"
            ))
        elif success_rate > 0.8 and avg_rating > 4.0:

            improvements.append(ImprovementAction(
                action_type="retrieval",
                description="Optimize retrieval count for speed",
                parameter="top_k",
                old_value=8,
                new_value=6,
                confidence=0.6,
                reasoning=f"High success rate ({success_rate:.1%}) allows for optimization"
            ))
        

        problematic_queries = [qp for qp in analysis.query_patterns if qp['needs_attention']]
        if len(problematic_queries) > 3:
            improvements.append(ImprovementAction(
                action_type="retrieval",
                description="Increase chunk size for complex queries",
                parameter="chunk_size_char",
                old_value=20000,
                new_value=25000,
                confidence=0.6,
                reasoning=f"Found {len(problematic_queries)} problematic query patterns"
            ))
        
        return improvements
    
    def _improve_generation_parameters(self, analysis) -> List[ImprovementAction]:
        improvements = []
        

        common_issues = analysis.common_issues
        for issue in common_issues:
            if issue['type'] == 'too_short' and issue['count'] > 2:
                improvements.append(ImprovementAction(
                    action_type="generation",
                    description="Increase max tokens for more detailed answers",
                    parameter="max_gen_tokens",
                    old_value=400,
                    new_value=600,
                    confidence=0.8,
                    reasoning=f"Users frequently complain about short answers ({issue['count']} times)"
                ))
            elif issue['type'] == 'too_long' and issue['count'] > 2:
                improvements.append(ImprovementAction(
                    action_type="generation",
                    description="Decrease max tokens for more concise answers",
                    parameter="max_gen_tokens",
                    old_value=400,
                    new_value=300,
                    confidence=0.7,
                    reasoning=f"Users frequently complain about verbose answers ({issue['count']} times)"
                ))
            elif issue['type'] == 'confusing' and issue['count'] > 2:
                improvements.append(ImprovementAction(
                    action_type="generation",
                    description="Lower temperature for more focused answers",
                    parameter="temperature",
                    old_value=0.3,
                    new_value=0.2,
                    confidence=0.6,
                    reasoning=f"Users find answers confusing ({issue['count']} times)"
                ))
        
        return improvements
    
    def _improve_prompt_template(self, analysis) -> List[ImprovementAction]:
        improvements = []
        

        common_issues = analysis.common_issues
        issue_types = [issue['type'] for issue in common_issues]
        
        if 'incomplete' in issue_types:
            improvements.append(ImprovementAction(
                action_type="prompt",
                description="Enhance prompt to encourage complete answers",
                parameter="prompt_template",
                old_value="current_template",
                new_value="enhanced_completeness_template",
                confidence=0.7,
                reasoning="Users frequently report incomplete answers"
            ))
        
        if 'irrelevant' in issue_types:
            improvements.append(ImprovementAction(
                action_type="prompt",
                description="Improve prompt to focus on relevance",
                parameter="prompt_template",
                old_value="current_template",
                new_value="relevance_focused_template",
                confidence=0.6,
                reasoning="Users report irrelevant answers"
            ))
        
        return improvements
    
    def _suggest_knowledge_base_expansion(self, analysis) -> List[ImprovementAction]:
        improvements = []
        

        problematic_queries = [qp for qp in analysis.query_patterns if qp['needs_attention']]
        
        if len(problematic_queries) > 5:

            topics = self._extract_topics_from_queries([qp['query'] for qp in problematic_queries])
            
            improvements.append(ImprovementAction(
                action_type="knowledge_base",
                description="Expand knowledge base for frequently problematic topics",
                parameter="pdf_sources",
                old_value="current_sources",
                new_value=f"add_sources_for_topics: {topics[:3]}",
                confidence=0.5,
                reasoning=f"Found {len(problematic_queries)} problematic queries on topics: {', '.join(topics[:3])}"
            ))
        
        return improvements
    
    def _extract_topics_from_queries(self, queries: List[str]) -> List[str]:
        all_words = []
        for query in queries:
            words = query.lower().split()

            filtered_words = [w for w in words if len(w) > 4 and w not in ['what', 'how', 'why', 'when', 'where']]
            all_words.extend(filtered_words)
        

        from collections import Counter
        word_counts = Counter(all_words)
        

        return [word for word, count in word_counts.most_common(10) if count > 1]
    
    def apply_improvements(self, improvements: List[ImprovementAction], 
                          dry_run: bool = True) -> Dict:
        results = {
            'applied': [],
            'skipped': [],
            'errors': []
        }
        
        for improvement in improvements:
            try:
                if improvement.confidence < 0.6:
                    results['skipped'].append({
                        'action': improvement.description,
                        'reason': f'Low confidence: {improvement.confidence:.2f}'
                    })
                    continue
                
                if not dry_run:
                    success = self._apply_single_improvement(improvement)
                    if success:
                        results['applied'].append(improvement.description)

                        self.db.log_improvement(
                            improvement_type=improvement.action_type,
                            description=improvement.description,
                            before_value=str(improvement.old_value),
                            after_value=str(improvement.new_value),
                            feedback_count=len(self.db.get_recent_feedback())
                        )
                    else:
                        results['errors'].append(f"Failed to apply: {improvement.description}")
                else:
                    results['applied'].append(f"[DRY RUN] {improvement.description}")
                    
            except Exception as e:
                results['errors'].append(f"Error applying {improvement.description}: {str(e)}")
        
        return results
    
    def _apply_single_improvement(self, improvement: ImprovementAction) -> bool:
        try:
            if improvement.action_type == "retrieval":
                return self._update_config_parameter(improvement.parameter, improvement.new_value)
            elif improvement.action_type == "generation":
                return self._update_config_parameter(improvement.parameter, improvement.new_value)
            elif improvement.action_type == "prompt":
                return self._update_prompt_template(improvement.new_value)
            elif improvement.action_type == "knowledge_base":
                return False
            else:
                return False
        except Exception:
            return False
    
    def _update_config_parameter(self, parameter: str, value: any) -> bool:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config[parameter] = value
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            return True
        except Exception:
            return False
    
    def _update_prompt_template(self, template_type: str) -> bool:
        return False
    
    def generate_improvement_report(self) -> str:
        improvements = self.analyze_and_improve()
        
        if not improvements:
            return "No improvements needed based on current feedback."
        
        report = []
        report.append("SYSTEM IMPROVEMENT SUGGESTIONS")
        report.append("=" * 50)
        

        by_type = {}
        for imp in improvements:
            if imp.action_type not in by_type:
                by_type[imp.action_type] = []
            by_type[imp.action_type].append(imp)
        
        for action_type, actions in by_type.items():
            report.append(f"\n{action_type.upper()} IMPROVEMENTS")
            report.append("-" * 30)
            
            for i, action in enumerate(actions, 1):
                report.append(f"{i}. {action.description}")
                report.append(f"   Parameter: {action.parameter}")
                report.append(f"   Change: {action.old_value} → {action.new_value}")
                report.append(f"   Confidence: {action.confidence:.1%}")
                report.append(f"   Reasoning: {action.reasoning}")
                report.append()
        

        high_confidence = [imp for imp in improvements if imp.confidence >= 0.7]
        report.append("SUMMARY")
        report.append("-" * 15)
        report.append(f"Total suggestions: {len(improvements)}")
        report.append(f"High confidence: {len(high_confidence)}")
        report.append(f"Ready to apply: {len([imp for imp in improvements if imp.confidence >= 0.6])}")
        
        return "\n".join(report)
