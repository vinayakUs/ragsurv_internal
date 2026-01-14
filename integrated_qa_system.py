"""
Integrated Q&A System - Enhanced Reasoning with RAG
==================================================

This module integrates the enhanced Q&A reasoning methodology with your existing
hybrid RAG system to provide structured, step-by-step answers.
"""

import sys
import os

# Add the current directory to the Python path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from enhanced_qa_module import EnhancedQAAnalyzer, QAResponse
from typing import Dict, List, Optional, Any
import json


class IntegratedRAGQASystem:
    """Integrated RAG system with enhanced Q&A reasoning"""
    
    def __init__(self):
        self.qa_analyzer = EnhancedQAAnalyzer()
        
        # Try to import existing backend services
        try:
            # These would be your existing modules
            sys.path.append(os.path.join(current_dir, 'backend', 'services'))
            from hybrid_search import HybridSearchService
            from query_engine import QueryEngineService
            
            self.hybrid_search = HybridSearchService()
            self.query_engine = QueryEngineService()
            self.has_backend = True
        except ImportError as e:
            print(f"Backend services not available: {e}")
            self.has_backend = False
    
    def process_query_with_enhanced_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Process query with enhanced step-by-step reasoning
        
        Args:
            query: User's question
            
        Returns:
            Dict with enhanced reasoning and answer
        """
        
        if self.has_backend:
            # Use existing hybrid search to get context
            try:
                search_results = self.hybrid_search.search(query)
                
                if search_results.get('found') and search_results.get('top_chunks'):
                    # Combine top chunks as context
                    context = ' '.join(search_results['top_chunks'])
                    source_info = search_results.get('sources', [])
                else:
                    context = "No relevant context found in knowledge base."
                    source_info = []
                    
            except Exception as e:
                print(f"Error in hybrid search: {e}")
                context = "Error retrieving context from knowledge base."
                source_info = []
        else:
            # Fallback: use sample context for demonstration
            context = self._get_sample_context()
            source_info = [{"document_name": "sample_document", "combined_score": 0.8}]
        
        # Apply enhanced reasoning to the context
        qa_response = self.qa_analyzer.analyze_question_with_context(query, context)
        
        # Format the complete response
        enhanced_response = {
            "query": query,
            "enhanced_reasoning": {
                "context_summary": qa_response.context_summary,
                "step_by_step_reasoning": [
                    {
                        "step": step.step_number,
                        "description": step.description,
                        "evidence": step.evidence,
                        "key_findings": step.key_findings
                    }
                    for step in qa_response.reasoning_steps
                ],
                "key_parameters": qa_response.key_numbers_and_parameters,
                "explicit_facts": qa_response.explicit_facts,
                "missing_information": qa_response.missing_information,
                "confidence_level": qa_response.confidence_level
            },
            "final_answer": qa_response.final_answer,
            "sources": source_info,
            "formatted_response": self.qa_analyzer.format_qa_response(qa_response)
        }
        
        return enhanced_response
    
    def _get_sample_context(self) -> str:
        """Provide sample context when backend is not available"""
        return """
        Enhanced Surveillance Measure (ESM) Version 1.1 July 2025 provides information on surveillance measures. 
        SEBI and Exchanges implement various surveillance measures like GSM, ASM, price band reduction, call auctions, T+T transfers. 
        The purpose of these measures is to alert investors and advise due diligence. 
        ESM is an additional surveillance measure for companies with market capitalization less than INR 1000 crores, 
        based on objective parameters such as price variation, standard deviation, and PE ratio.
        """
    
    def process_multiple_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions with enhanced reasoning"""
        results = []
        
        for question in questions:
            try:
                result = self.process_query_with_enhanced_reasoning(question)
                results.append(result)
            except Exception as e:
                results.append({
                    "query": question,
                    "error": str(e),
                    "final_answer": "Error processing question"
                })
        
        return results
    
    def create_qa_report(self, questions_and_answers: List[Dict[str, Any]]) -> str:
        """Create a comprehensive Q&A report with reasoning"""
        
        report = "# Enhanced Q&A Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        for i, qa in enumerate(questions_and_answers, 1):
            if 'error' in qa:
                report += f"## Question {i}: {qa['query']}\n"
                report += f"**Error**: {qa['error']}\n\n"
                continue
            
            report += f"## Question {i}: {qa['query']}\n\n"
            
            # Context summary
            reasoning = qa.get('enhanced_reasoning', {})
            if reasoning.get('context_summary'):
                report += f"**Context**: {reasoning['context_summary']}\n\n"
            
            # Step-by-step reasoning
            report += "### Step-by-step reasoning:\n\n"
            for step in reasoning.get('step_by_step_reasoning', []):
                report += f"{step['step']}. **{step['description']}**:\n"
                for evidence in step['evidence']:
                    report += f"   - {evidence}\n"
                if step['key_findings']:
                    report += f"   - **Key findings**: {', '.join(step['key_findings'])}\n"
                report += "\n"
            
            # Key parameters
            if reasoning.get('key_parameters'):
                report += "### Key Numbers/Parameters:\n"
                for param in reasoning['key_parameters']:
                    report += f"- {param}\n"
                report += "\n"
            
            # Explicit facts
            if reasoning.get('explicit_facts'):
                report += "### Explicit Facts from Context:\n"
                for fact in reasoning['explicit_facts']:
                    report += f"- {fact}\n"
                report += "\n"
            
            # Missing information
            if reasoning.get('missing_information'):
                report += "### Information Not Explicitly Mentioned:\n"
                for missing in reasoning['missing_information']:
                    report += f"- {missing}\n"
                report += "\n"
            
            # Final answer
            report += f"### Final Answer:\n"
            report += f"**{qa['final_answer']}**\n\n"
            
            # Confidence
            report += f"**Confidence Level**: {reasoning.get('confidence_level', 'Unknown')}\n\n"
            
            # Sources
            if qa.get('sources'):
                report += "### Sources:\n"
                for source in qa['sources']:
                    doc_name = source.get('document_name', 'Unknown')
                    score = source.get('combined_score', 0)
                    report += f"- {doc_name} (Score: {score:.2f})\n"
                report += "\n"
            
            report += "-" * 50 + "\n\n"
        
        return report


def run_example_analysis():
    """Run example analysis with sample questions"""
    
    qa_system = IntegratedRAGQASystem()
    
    # Sample questions for testing
    test_questions = [
        "What is the purpose of ESM framework?",
        "What is the market capitalization threshold for ESM?",
        "What parameters are used for ESM evaluation?",
        "Who issues ESM guidelines?"
    ]
    
    print("Processing questions with enhanced reasoning...")
    print("=" * 60)
    
    # Process questions
    results = qa_system.process_multiple_questions(test_questions)
    
    # Display individual results
    for result in results:
        if 'formatted_response' in result:
            print(result['formatted_response'])
            print("=" * 60)
    
    # Create comprehensive report
    report = qa_system.create_qa_report(results)
    
    # Save report
    report_path = os.path.join(current_dir, "enhanced_qa_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nComprehensive report saved to: {report_path}")
    
    return results


def test_single_question():
    """Test with a single question for detailed analysis"""
    
    qa_system = IntegratedRAGQASystem()
    
    # Test question matching your example
    question = "What is the purpose for introduction of ESM framework?"
    
    print(f"Analyzing question: {question}")
    print("=" * 80)
    
    result = qa_system.process_query_with_enhanced_reasoning(question)
    
    if 'formatted_response' in result:
        print(result['formatted_response'])
    
    # Also show JSON structure
    print("\n" + "=" * 80)
    print("JSON Response Structure:")
    print("=" * 80)
    
    # Pretty print the structured response (excluding formatted_response for brevity)
    clean_result = {k: v for k, v in result.items() if k != 'formatted_response'}
    print(json.dumps(clean_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("Enhanced Q&A System Test")
    print("=" * 40)
    print("1. Testing single question analysis")
    test_single_question()
    
    print("\n\n" + "=" * 40)
    print("2. Running multiple question analysis")
    run_example_analysis()