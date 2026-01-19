"""
Enhanced Q&A Answer Module with Step-by-Step Reasoning
=====================================================

This module provides structured answers with detailed step-by-step reasoning
based ONLY on the provided context, following the methodology:

1. Read the context carefully
2. Identify relevant sentences or phrases that answer the question
3. Explain step by step how the context supports the answer
4. Highlight key numbers, parameters, or rules mentioned in the context
5. If context does not explicitly state something, do NOT speculate
6. Provide a concise final answer after reasoning
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContextEvidence:
    """Structure to hold evidence found in context"""
    sentence: str
    relevance: str
    key_facts: List[str]
    line_reference: Optional[str] = None


@dataclass
class ReasoningStep:
    """Structure for each reasoning step"""
    step_number: int
    description: str
    evidence: List[str]
    key_findings: List[str]


@dataclass
class QAResponse:
    """Complete structured Q&A response"""
    question: str
    context_summary: str
    reasoning_steps: List[ReasoningStep]
    key_numbers_and_parameters: List[str]
    explicit_facts: List[str]
    missing_information: List[str]
    final_answer: str
    confidence_level: str


class EnhancedQAAnalyzer:
    """Enhanced Q&A Analyzer with step-by-step reasoning"""
    
    def __init__(self):
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
    
    def analyze_question_with_context(self, question: str, context: str) -> QAResponse:
        """
        Analyze question with context using structured methodology
        
        Args:
            question: The question to answer
            context: The context to analyze
            
        Returns:
            QAResponse: Structured response with reasoning
        """
        
        # Step 1: Read context carefully and extract structure
        context_summary = self._summarize_context(context)
        
        # Step 2: Identify relevant sentences
        relevant_evidence = self._identify_relevant_sentences(question, context)
        
        # Step 3: Build step-by-step reasoning
        reasoning_steps = self._build_reasoning_steps(question, relevant_evidence)
        
        # Step 4: Extract key numbers, parameters, rules
        key_params = self._extract_key_parameters(context, question)
        
        # Step 5: Identify explicit facts vs missing information
        explicit_facts, missing_info = self._categorize_information(question, context, relevant_evidence)
        
        # Step 6: Generate final answer with confidence
        final_answer, confidence = self._generate_final_answer(question, reasoning_steps, explicit_facts)
        
        return QAResponse(
            question=question,
            context_summary=context_summary,
            reasoning_steps=reasoning_steps,
            key_numbers_and_parameters=key_params,
            explicit_facts=explicit_facts,
            missing_information=missing_info,
            final_answer=final_answer,
            confidence_level=confidence
        )
    
    def _summarize_context(self, context: str) -> str:
        """Provide brief summary of context"""
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
        key_topics = []
        
        # Extract key topics from first few sentences
        for sentence in sentences[:3]:
            if any(keyword in sentence.lower() for keyword in ['purpose', 'framework', 'measure', 'system']):
                key_topics.append(sentence.strip())
        
        if key_topics:
            return f"Context discusses: {'. '.join(key_topics[:2])}"
        else:
            return f"Context contains {len(sentences)} main statements about the topic"
    
    def _identify_relevant_sentences(self, question: str, context: str) -> List[ContextEvidence]:
        """Identify sentences in context that are relevant to the question"""
        
        # Extract key terms from question
        question_terms = self._extract_question_terms(question)
        
        # Split context into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if len(s.strip()) > 10]
        
        relevant_evidence = []
        
        for i, sentence in enumerate(sentences):
            # Check relevance based on keyword matching
            relevance_score = self._calculate_relevance(question_terms, sentence)
            
            if relevance_score > 0.3:  # Threshold for relevance
                key_facts = self._extract_key_facts_from_sentence(sentence)
                
                evidence = ContextEvidence(
                    sentence=sentence,
                    relevance=f"Contains {len([t for t in question_terms if t.lower() in sentence.lower()])} question keywords",
                    key_facts=key_facts,
                    line_reference=f"Context sentence {i+1}"
                )
                relevant_evidence.append(evidence)
        
        return relevant_evidence
    
    def _extract_question_terms(self, question: str) -> List[str]:
        """Extract key terms from question"""
        # Remove question words
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'the', 'a', 'an']
        
        # Clean and split question
        clean_question = re.sub(r'[^\w\s]', '', question.lower())
        terms = [word for word in clean_question.split() if word not in question_words and len(word) > 2]
        
        return terms
    
    def _calculate_relevance(self, question_terms: List[str], sentence: str) -> float:
        """Calculate relevance score between question terms and sentence"""
        sentence_lower = sentence.lower()
        matches = sum(1 for term in question_terms if term in sentence_lower)
        
        if len(question_terms) == 0:
            return 0.0
        
        return matches / len(question_terms)
    
    def _extract_key_facts_from_sentence(self, sentence: str) -> List[str]:
        """Extract key facts like numbers, names, dates from sentence"""
        key_facts = []
        
        # Extract numbers (including percentages, dates, etc.)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)*(?:%|\.?\s*(?:crores?|lakhs?|billions?|millions?|thousands?))?\b', sentence)
        key_facts.extend([f"Number: {num}" for num in numbers])
        
        # Extract capitalized terms (likely proper nouns/names)
        proper_nouns = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', sentence)
        key_facts.extend([f"Entity: {noun}" for noun in proper_nouns if len(noun) > 2])
        
        # Extract quoted text
        quotes = re.findall(r'"([^"]*)"', sentence)
        key_facts.extend([f"Quote: {quote}" for quote in quotes])
        
        return key_facts[:3]  # Limit to most important facts
    
    def _build_reasoning_steps(self, question: str, evidence: List[ContextEvidence]) -> List[ReasoningStep]:
        """Build step-by-step reasoning based on evidence"""
        reasoning_steps = []
        
        if not evidence:
            reasoning_steps.append(ReasoningStep(
                step_number=1,
                description="Context Analysis",
                evidence=["No directly relevant sentences found in context"],
                key_findings=["Context may not contain specific information to answer the question"]
            ))
            return reasoning_steps
        
        # Step 1: Context overview
        reasoning_steps.append(ReasoningStep(
            step_number=1,
            description="Context Analysis",
            evidence=[f"Found {len(evidence)} relevant sentences in context"],
            key_findings=[f"Evidence strength: {len(evidence)} matching passages"]
        ))
        
        # Step 2-N: Analyze each piece of evidence
        for i, ev in enumerate(evidence[:3]):  # Limit to top 3 pieces of evidence
            reasoning_steps.append(ReasoningStep(
                step_number=i + 2,
                description=f"Evidence Analysis {i + 1}",
                evidence=[f"Text: '{ev.sentence[:100]}...' if relevant",
                         f"Relevance: {ev.relevance}",
                         f"Location: {ev.line_reference}"],
                key_findings=ev.key_facts
            ))
        
        return reasoning_steps
    
    def _extract_key_parameters(self, context: str, question: str) -> List[str]:
        """Extract key numbers, parameters, and rules from context"""
        parameters = []
        
        # Extract numerical values with context
        number_patterns = [
            r'INR\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*crores?',
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d{4})',  # Years
            r'Version\s*(\d+(?:\.\d+)*)',
            r'(\d+(?:,\d+)*)\s*(?:companies?|participants?|measures?)'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                # Find the sentence containing this number for context
                for sentence in context.split('.'):
                    if match in sentence:
                        param_context = sentence.strip()[:100] + "..." if len(sentence) > 100 else sentence.strip()
                        parameters.append(f"{match} (from: {param_context})")
                        break
        
        # Extract rules and criteria
        rule_keywords = ['criteria', 'rule', 'requirement', 'condition', 'threshold', 'limit']
        sentences = context.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in rule_keywords):
                rule_text = sentence.strip()[:150] + "..." if len(sentence) > 150 else sentence.strip()
                if rule_text:
                    parameters.append(f"Rule: {rule_text}")
        
        return parameters[:5]  # Limit to most important parameters
    
    def _categorize_information(self, question: str, context: str, evidence: List[ContextEvidence]) -> Tuple[List[str], List[str]]:
        """Categorize information into explicit facts vs missing information"""
        
        explicit_facts = []
        missing_info = []
        
        # Analyze what is explicitly stated
        for ev in evidence:
            if ev.sentence and len(ev.sentence.strip()) > 10:
                explicit_facts.append(f"Explicitly stated: {ev.sentence}")
        
        # Common question patterns and what they might be looking for
        question_lower = question.lower()
        
        if 'purpose' in question_lower or 'why' in question_lower:
            purpose_found = any('purpose' in ev.sentence.lower() or 'for' in ev.sentence.lower() 
                              for ev in evidence)
            if not purpose_found:
                missing_info.append("Specific purpose or rationale not explicitly mentioned")
        
        if 'when' in question_lower or 'date' in question_lower:
            date_found = any(re.search(r'\d{4}|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', 
                                     ev.sentence.lower()) for ev in evidence)
            if not date_found:
                missing_info.append("Specific dates or timing not mentioned")
        
        if 'who' in question_lower:
            person_found = any(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', ev.sentence) for ev in evidence)
            if not person_found:
                missing_info.append("Specific persons or responsible parties not named")
        
        if 'how much' in question_lower or 'cost' in question_lower:
            amount_found = any(re.search(r'INR|₹|\$|\d+.*(?:crore|lakh|million)', ev.sentence) for ev in evidence)
            if not amount_found:
                missing_info.append("Specific amounts or costs not mentioned")
        
        return explicit_facts[:3], missing_info[:3]  # Limit for clarity
    
    def _generate_final_answer(self, question: str, reasoning_steps: List[ReasoningStep], 
                             explicit_facts: List[str]) -> Tuple[str, str]:
        """Generate final answer with confidence level"""
        
        if not explicit_facts or len(explicit_facts) == 0:
            return ("Insufficient information in knowledge base to answer this question.", "Low")
        
        # Build answer from reasoning steps
        answer_parts = []
        confidence_score = 0
        
        for step in reasoning_steps[1:]:  # Skip overview step
            if step.key_findings:
                for finding in step.key_findings:
                    if 'Number:' in finding or 'Entity:' in finding or 'Quote:' in finding:
                        answer_parts.append(finding.replace('Number:', '').replace('Entity:', '').replace('Quote:', ''))
                        confidence_score += 0.3
        
        # Extract most relevant facts for answer
        key_answer_elements = []
        for fact in explicit_facts:
            if 'Explicitly stated:' in fact:
                clean_fact = fact.replace('Explicitly stated:', '').strip()
                if len(clean_fact) > 20:  # Substantial content
                    key_answer_elements.append(clean_fact)
                    confidence_score += 0.4
        
        if key_answer_elements:
            # Combine the most relevant elements
            final_answer = '. '.join(key_answer_elements[:2])  # Limit to 2 most relevant facts
            
            # Determine confidence level
            if confidence_score >= self.confidence_thresholds["high"]:
                confidence = "High"
            elif confidence_score >= self.confidence_thresholds["medium"]:
                confidence = "Medium"
            else:
                confidence = "Low"
                
        else:
            final_answer = "The context mentions the topic but does not provide specific details to fully answer the question."
            confidence = "Low"
        
        return final_answer, confidence
    
    def format_qa_response(self, response: QAResponse) -> str:
        """Format the complete Q&A response for display"""
        
        output = f"""
Question: {response.question}

Context: {response.context_summary}

Step-by-step reasoning:
"""
        
        for step in response.reasoning_steps:
            output += f"\n{step.step_number}. {step.description}:\n"
            for evidence in step.evidence:
                output += f"   - {evidence}\n"
            if step.key_findings:
                output += f"   Key findings: {', '.join(step.key_findings)}\n"
        
        if response.key_numbers_and_parameters:
            output += f"\nKey Numbers/Parameters:\n"
            for param in response.key_numbers_and_parameters:
                output += f"- {param}\n"
        
        if response.explicit_facts:
            output += f"\nExplicit Facts from Context:\n"
            for fact in response.explicit_facts:
                output += f"- {fact}\n"
        
        if response.missing_information:
            output += f"\nInformation Not Explicitly Mentioned:\n"
            for missing in response.missing_information:
                output += f"- {missing}\n"
        
        output += f"\nFinal Answer:\n- {response.final_answer}\n"
        output += f"\nConfidence Level: {response.confidence_level}\n"
        
        return output


# Example usage and testing
def test_enhanced_qa():
    """Test the enhanced Q&A system with example"""
    
    analyzer = EnhancedQAAnalyzer()
    
    # Example context (simulating ESM document content)
    sample_context = """
    Enhanced Surveillance Measure (ESM) Version 1.1 July 2025 Disclaimer provides general information on queries relating to the topic of enhanced surveillance measures, issued by NSE for member ease understanding. SEBI and Exchanges implement various surveillance measures like GSM, ASM, price band reduction, call auctions, T+T transfers. The purpose of these measures is to alert investors and advise due diligence. ESM is an additional surveillance measure for companies with market capitalization less than INR 1000 crores, based on objective parameters such as price variation, standard deviation, and PE ratio.
    """
    
    # Test question
    test_question = "What is the purpose of ESM framework?"
    
    # Analyze
    response = analyzer.analyze_question_with_context(test_question, sample_context)
    
    # Display formatted response
    print("=" * 80)
    print("ENHANCED Q&A ANALYSIS")
    print("=" * 80)
    print(analyzer.format_qa_response(response))


if __name__ == "__main__":
    test_enhanced_qa()