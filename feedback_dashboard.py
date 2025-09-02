

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from feedback_db import FeedbackDB
from feedback_analyzer import FeedbackAnalyzer
from system_improver import SystemImprover

class FeedbackDashboard:
    
    def __init__(self, feedback_db: FeedbackDB):
        self.db = feedback_db
        self.analyzer = FeedbackAnalyzer(feedback_db)
        self.improver = SystemImprover(feedback_db)
    
    def show_main_dashboard(self):
        while True:
            print("\n" + "="*60)
            print("TOKENSMITH FEEDBACK DASHBOARD")
            print("="*60)
            print("1. System Overview")
            print("2. Detailed Analytics")
            print("3. Improvement Suggestions")
            print("4. Recent Feedback")
            print("5. Apply Improvements")
            print("6. Export Report")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                self._show_system_overview()
            elif choice == "2":
                self._show_detailed_analytics()
            elif choice == "3":
                self._show_improvement_suggestions()
            elif choice == "4":
                self._show_recent_feedback()
            elif choice == "5":
                self._apply_improvements_interface()
            elif choice == "6":
                self._export_report()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")
    
    def _show_system_overview(self):
        """Display high-level system metrics."""
        print("\n SYSTEM OVERVIEW")
        print("-" * 40)
        
        # Get basic stats
        stats = self.db.get_feedback_stats()
        
        print(f"Total Interactions: {stats['total_feedback']}")
        print(f"Success Rate: {stats['thumbs_up_rate']:.1%}")
        print(f"Average Rating: {stats['avg_rating']:.1f}/5.0")
        print(f"Comments Received: {stats['comments_count']}")
        
        # Calculate engagement score
        if stats['total_feedback'] > 0:
            engagement = (stats['thumbs_up_rate'] + (stats['comments_count'] / stats['total_feedback'])) / 2
            print(f"Engagement Score: {engagement:.1%}")
        
        # Show trend (last 7 days vs previous 7 days)
        self._show_trend_analysis()
        
        # Quick health check
        self._show_health_check(stats)
    
    def _show_trend_analysis(self):
        print("\nTREND ANALYSIS (Last 7 days)")
        print("-" * 35)
        
        # Get recent feedback
        recent_feedback = self.db.get_recent_feedback(limit=100)
        
        if len(recent_feedback) < 10:
            print("Insufficient data for trend analysis")
            return
        
        # Simple trend calculation
        recent_7_days = [fb for fb in recent_feedback if self._is_within_days(fb['created_at'], 7)]
        previous_7_days = [fb for fb in recent_feedback if self._is_within_days(fb['created_at'], 14) and not self._is_within_days(fb['created_at'], 7)]
        
        if recent_7_days and previous_7_days:
            recent_success = sum(1 for fb in recent_7_days if fb['thumbs_up']) / len(recent_7_days)
            previous_success = sum(1 for fb in previous_7_days if fb['thumbs_up']) / len(previous_7_days)
            
            trend = recent_success - previous_success
            trend_emoji = "UP" if trend > 0.05 else "DOWN" if trend < -0.05 else "STABLE"
            
            print(f"Success Rate Trend: {trend_emoji} {trend:+.1%}")
            print(f"Recent (7 days): {recent_success:.1%}")
            print(f"Previous (7 days): {previous_success:.1%}")
    
    def _show_health_check(self, stats: Dict):
        print("\nSYSTEM HEALTH CHECK")
        print("-" * 25)
        
        health_issues = []
        
        if stats['thumbs_up_rate'] < 0.6:
            health_issues.append("Low success rate")
        
        if stats['avg_rating'] < 3.0:
            health_issues.append("Low average rating")
        
        if stats['total_feedback'] > 10 and stats['comments_count'] / stats['total_feedback'] > 0.3:
            health_issues.append("High comment rate (may indicate issues)")
        
        if health_issues:
            print("Issues detected:")
            for issue in health_issues:
                print(f"  {issue}")
        else:
            print("System health looks good!")
    
    def _show_detailed_analytics(self):
        print("\nDETAILED ANALYTICS")
        print("-" * 30)
        
        analysis = self.analyzer.analyze_feedback()
        
        # Query patterns
        if analysis.query_patterns:
            print("\nQUERY PATTERNS")
            print("-" * 20)
            for i, pattern in enumerate(analysis.query_patterns[:5], 1):
                status = "NEEDS_ATTENTION" if pattern['needs_attention'] else "GOOD"
                print(f"{i}. {status} '{pattern['query'][:40]}...'")
                print(f"   Frequency: {pattern['frequency']}, Success: {pattern['success_rate']:.1%}")
        
        # Common issues
        if analysis.common_issues:
            print("\nCOMMON ISSUES")
            print("-" * 18)
            for issue in analysis.common_issues[:5]:
                print(f"• {issue['type'].replace('_', ' ').title()}: {issue['count']} occurrences")
                if issue['examples']:
                    print(f"  Example: {issue['examples'][0]['comment'][:60]}...")
        
        # System metrics
        print("\nDETAILED METRICS")
        print("-" * 22)
        for metric, value in analysis.system_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")
    
    def _show_improvement_suggestions(self):
        print("\nIMPROVEMENT SUGGESTIONS")
        print("-" * 30)
        
        # Get improvement suggestions
        improvements = self.improver.analyze_and_improve()
        
        if not improvements:
            print("No improvements needed based on current feedback.")
            return
        
        # Group by confidence level
        high_confidence = [imp for imp in improvements if imp.confidence >= 0.7]
        medium_confidence = [imp for imp in improvements if 0.6 <= imp.confidence < 0.7]
        low_confidence = [imp for imp in improvements if imp.confidence < 0.6]
        
        if high_confidence:
            print("\nHIGH PRIORITY (Confidence ≥ 70%)")
            for i, imp in enumerate(high_confidence, 1):
                print(f"{i}. {imp.description}")
                print(f"   {imp.parameter}: {imp.old_value} → {imp.new_value}")
                print(f"   Reasoning: {imp.reasoning}")
        
        if medium_confidence:
            print("\nMEDIUM PRIORITY (Confidence 60-70%)")
            for i, imp in enumerate(medium_confidence, 1):
                print(f"{i}. {imp.description}")
                print(f"   {imp.parameter}: {imp.old_value} → {imp.new_value}")
        
        if low_confidence:
            print("\nLOW PRIORITY (Confidence < 60%)")
            for i, imp in enumerate(low_confidence, 1):
                print(f"{i}. {imp.description}")
    
    def _show_recent_feedback(self):
        print("\nRECENT FEEDBACK")
        print("-" * 20)
        
        recent = self.db.get_recent_feedback(limit=10)
        
        if not recent:
            print("No feedback available yet.")
            return
        
        for i, fb in enumerate(recent, 1):
            thumbs = "THUMBS_UP" if fb['thumbs_up'] else "THUMBS_DOWN" if fb['thumbs_up'] is False else "NO_FEEDBACK"
            rating = f"RATING_{fb['rating']}" if fb['rating'] else "NO_RATING"
            
            print(f"{i}. {thumbs} {rating} | {fb['query'][:50]}...")
            if fb['comment']:
                print(f"   COMMENT: {fb['comment'][:80]}...")
            print(f"   DATE: {fb['created_at'][:19]}")
            print()
    
    def _apply_improvements_interface(self):
        print("\nAPPLY IMPROVEMENTS")
        print("-" * 25)
        
        improvements = self.improver.analyze_and_improve()
        
        if not improvements:
            print("No improvements available to apply.")
            return
        
        # Show available improvements
        print("Available improvements:")
        for i, imp in enumerate(improvements, 1):
            if imp.confidence >= 0.6:
                print(f"{i}. {imp.description} (Confidence: {imp.confidence:.1%})")
        
        print("\nOptions:")
        print("1. Apply all high-confidence improvements")
        print("2. Apply specific improvements")
        print("3. Dry run (preview changes)")
        print("4. Back to main menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            high_conf = [imp for imp in improvements if imp.confidence >= 0.7]
            if high_conf:
                results = self.improver.apply_improvements(high_conf, dry_run=False)
                self._show_improvement_results(results)
            else:
                print("No high-confidence improvements available.")
        
        elif choice == "2":
            self._apply_specific_improvements(improvements)
        
        elif choice == "3":
            results = self.improver.apply_improvements(improvements, dry_run=True)
            self._show_improvement_results(results)
        
        elif choice == "4":
            return
        
        else:
            print("Invalid option.")
    
    def _apply_specific_improvements(self, improvements: List):
        print("\nSelect improvements to apply (comma-separated numbers):")
        
        for i, imp in enumerate(improvements, 1):
            if imp.confidence >= 0.6:
                print(f"{i}. {imp.description} (Confidence: {imp.confidence:.1%})")
        
        try:
            selection = input("\nEnter numbers: ").strip()
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            
            selected = [improvements[i] for i in indices if 0 <= i < len(improvements)]
            
            if selected:
                results = self.improver.apply_improvements(selected, dry_run=False)
                self._show_improvement_results(results)
            else:
                print("No valid improvements selected.")
        
        except ValueError:
            print("Invalid input format.")
    
    def _show_improvement_results(self, results: Dict):
        print("\nIMPROVEMENT RESULTS")
        print("-" * 25)
        
        if results['applied']:
            print("Applied improvements:")
            for item in results['applied']:
                print(f"  • {item}")
        
        if results['skipped']:
            print("\nSkipped improvements:")
            for item in results['skipped']:
                print(f"  • {item['action']} - {item['reason']}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  • {error}")
    
    def _export_report(self):
        print("\nEXPORT REPORT")
        print("-" * 20)
        
        # Generate comprehensive report
        analysis_report = self.analyzer.generate_improvement_report()
        improvement_report = self.improver.generate_improvement_report()
        
        # Combine reports
        full_report = f"""
TOKENSMITH FEEDBACK REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{analysis_report}

{improvement_report}

---
Report generated by TokenSmith Feedback System
        """.strip()
        
        # Save to file
        filename = f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(full_report)
        
        print(f"Report exported to: {filename}")
        print(f"Report length: {len(full_report)} characters")
    
    def _is_within_days(self, timestamp: str, days: int) -> bool:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            cutoff = datetime.now() - timedelta(days=days)
            return dt >= cutoff
        except:
            return False

def main():
    db = FeedbackDB()
    dashboard = FeedbackDashboard(db)
    dashboard.show_main_dashboard()

if __name__ == "__main__":
    main()
