"""
Evaluation Framework for Tender Pricing RAG System
Comprehensive evaluation metrics and testing suite
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json
import time
import re
from datetime import datetime
import logging
from pathlib import Path

# Statistical imports
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from rag_pipeline import TenderRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TenderRAGEvaluator:
    """Comprehensive evaluation framework for tender pricing RAG system"""
    
    def __init__(self, pipeline: TenderRAGPipeline):
        """
        Initialize evaluator with RAG pipeline
        
        Args:
            pipeline: TenderRAGPipeline instance
        """
        self.pipeline = pipeline
        self.evaluation_results = {}
        self.test_queries = self._create_test_queries()
        self.ground_truth = self._create_ground_truth()
        
    def _create_test_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive test query set"""
        test_queries = [
            # Direct pricing queries
            {
                "query": "What is the average price for healthcare services in Bogotá?",
                "type": "pricing_lookup",
                "expected_elements": ["price", "average", "Bogotá", "healthcare"],
                "difficulty": "easy"
            },
            {
                "query": "Show me contracts for medical equipment over $100,000",
                "type": "pricing_lookup",
                "expected_elements": ["medical equipment", "100000", "contracts"],
                "difficulty": "medium"
            },
            {
                "query": "What are the typical costs for pharmaceutical services in Valle?",
                "type": "pricing_lookup",
                "expected_elements": ["pharmaceutical", "costs", "Valle"],
                "difficulty": "medium"
            },
            
            # Comparative analysis
            {
                "query": "Compare pricing for nursing services across different regions",
                "type": "comparative_analysis",
                "expected_elements": ["nursing", "compare", "regions", "pricing"],
                "difficulty": "hard"
            },
            {
                "query": "How do contract prices vary between Bogotá and Antioquia?",
                "type": "comparative_analysis",
                "expected_elements": ["Bogotá", "Antioquia", "prices", "vary"],
                "difficulty": "hard"
            },
            
            # Budget estimation
            {
                "query": "Estimate budget for laboratory services contract",
                "type": "budget_estimation",
                "expected_elements": ["estimate", "budget", "laboratory", "services"],
                "difficulty": "hard"
            },
            {
                "query": "What should I budget for diagnostic equipment procurement?",
                "type": "budget_estimation",
                "expected_elements": ["budget", "diagnostic", "equipment", "procurement"],
                "difficulty": "hard"
            },
            
            # General inquiries
            {
                "query": "What services does contractor BIBIANA MARCELA ALVARADO BUSTOS provide?",
                "type": "general_inquiry",
                "expected_elements": ["contractor", "services", "BIBIANA"],
                "difficulty": "easy"
            },
            {
                "query": "Show me all cancelled contracts and their reasons",
                "type": "general_inquiry",
                "expected_elements": ["cancelled", "contracts", "reasons"],
                "difficulty": "medium"
            },
            
            # Edge cases
            {
                "query": "Find the most expensive healthcare contract in 2024",
                "type": "pricing_lookup",
                "expected_elements": ["expensive", "healthcare", "2024"],
                "difficulty": "hard"
            },
            {
                "query": "What are the payment terms for contracts in Atlántico?",
                "type": "general_inquiry",
                "expected_elements": ["payment", "terms", "Atlántico"],
                "difficulty": "medium"
            },
            
            # Complex queries
            {
                "query": "Compare the average contract value for direct recruitment vs public procurement",
                "type": "comparative_analysis",
                "expected_elements": ["compare", "direct recruitment", "public procurement", "average"],
                "difficulty": "hard"
            },
            {
                "query": "What percentage of contracts are cancelled and what is the average value?",
                "type": "general_inquiry",
                "expected_elements": ["percentage", "cancelled", "average", "value"],
                "difficulty": "hard"
            }
        ]
        
        return test_queries
    
    def _create_ground_truth(self) -> Dict[str, Any]:
        """Create ground truth data for evaluation"""
        # This would ideally be created from manual annotation
        # For now, create basic expected patterns
        ground_truth = {
            "pricing_patterns": {
                "should_contain_currency": True,
                "should_contain_numbers": True,
                "should_mention_location": True,
                "should_be_specific": True
            },
            "comparative_patterns": {
                "should_compare_multiple_items": True,
                "should_show_differences": True,
                "should_provide_context": True
            },
            "estimation_patterns": {
                "should_provide_range": True,
                "should_justify_estimate": True,
                "should_mention_factors": True
            }
        }
        
        return ground_truth
    
    def evaluate_response_quality(self, query: str, response: str, query_type: str) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics"""
        
        metrics = {}
        
        # Relevance scoring
        metrics['relevance'] = self._score_relevance(query, response)
        
        # Completeness scoring
        metrics['completeness'] = self._score_completeness(response, query_type)
        
        # Accuracy scoring (based on content patterns)
        metrics['accuracy'] = self._score_accuracy(response, query_type)
        
        # Clarity scoring
        metrics['clarity'] = self._score_clarity(response)
        
        # Specificity scoring
        metrics['specificity'] = self._score_specificity(response, query_type)
        
        # Overall score
        metrics['overall'] = (
            metrics['relevance'] * 0.25 +
            metrics['completeness'] * 0.25 +
            metrics['accuracy'] * 0.25 +
            metrics['clarity'] * 0.15 +
            metrics['specificity'] * 0.10
        )
        
        return metrics
    
    def _score_relevance(self, query: str, response: str) -> float:
        """Score how relevant the response is to the query"""
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        query_terms = query_terms - stop_words
        response_terms = response_terms - stop_words
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & response_terms)
        relevance = overlap / len(query_terms)
        
        return min(relevance, 1.0)
    
    def _score_completeness(self, response: str, query_type: str) -> float:
        """Score how complete the response is"""
        
        if query_type == "pricing_lookup":
            # Should contain price information
            has_price = bool(re.search(r'[\$\€\£]\d+|price|cost|value', response, re.IGNORECASE))
            has_context = len(response.split()) > 20  # Reasonable length
            return (has_price + has_context) / 2
        
        elif query_type == "comparative_analysis":
            # Should compare multiple items
            comparison_words = ['compare', 'versus', 'vs', 'difference', 'higher', 'lower', 'more', 'less']
            has_comparison = any(word in response.lower() for word in comparison_words)
            has_multiple_items = len(re.findall(r'\b\w+\b', response)) > 30
            return (has_comparison + has_multiple_items) / 2
        
        elif query_type == "budget_estimation":
            # Should provide estimation with reasoning
            estimation_words = ['estimate', 'approximately', 'around', 'range', 'budget']
            has_estimation = any(word in response.lower() for word in estimation_words)
            has_reasoning = len(response.split()) > 25
            return (has_estimation + has_reasoning) / 2
        
        else:
            # General completeness
            return min(len(response.split()) / 30, 1.0)
    
    def _score_accuracy(self, response: str, query_type: str) -> float:
        """Score accuracy based on content patterns"""
        
        # Check for factual consistency patterns
        accuracy_score = 0.8  # Base score
        
        # Penalty for contradictions
        if 'but' in response.lower() and 'however' in response.lower():
            accuracy_score -= 0.1
        
        # Bonus for specific numbers and dates
        if re.search(r'\d+', response):
            accuracy_score += 0.1
        
        # Bonus for location mentions
        locations = ['Bogotá', 'Antioquia', 'Valle', 'Atlántico']
        if any(loc in response for loc in locations):
            accuracy_score += 0.1
        
        return min(accuracy_score, 1.0)
    
    def _score_clarity(self, response: str) -> float:
        """Score clarity and readability"""
        
        sentences = response.split('.')
        
        # Average sentence length
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Clarity score based on sentence length (optimal 15-20 words)
        if 10 <= avg_sentence_length <= 25:
            clarity_score = 1.0
        elif 5 <= avg_sentence_length <= 35:
            clarity_score = 0.8
        else:
            clarity_score = 0.6
        
        # Bonus for structured response
        if any(marker in response for marker in ['1.', '2.', '•', '-']):
            clarity_score = min(clarity_score + 0.1, 1.0)
        
        return clarity_score
    
    def _score_specificity(self, response: str, query_type: str) -> float:
        """Score how specific the response is"""
        
        # Check for specific numbers, names, dates
        specificity_indicators = [
            r'\$[\d,]+',  # Currency amounts
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimals
        ]
        
        specificity_count = sum(1 for pattern in specificity_indicators if re.search(pattern, response))
        
        # Normalize by number of indicators
        specificity_score = min(specificity_count / len(specificity_indicators), 1.0)
        
        return specificity_score
    
    def evaluate_retrieval_quality(self, query: str, context_docs: List[Dict]) -> Dict[str, float]:
        """Evaluate retrieval quality"""
        
        if not context_docs:
            return {'precision': 0.0, 'recall': 0.0, 'relevance': 0.0}
        
        # Calculate relevance scores
        relevance_scores = []
        for doc in context_docs:
            doc_relevance = self._calculate_doc_relevance(query, doc)
            relevance_scores.append(doc_relevance)
        
        # Retrieval metrics
        avg_relevance = np.mean(relevance_scores)
        precision = sum(1 for score in relevance_scores if score > 0.5) / len(relevance_scores)
        
        # Diversity (different collections)
        collections = set(doc['collection'] for doc in context_docs)
        diversity = len(collections) / len(context_docs)
        
        return {
            'precision': precision,
            'avg_relevance': avg_relevance,
            'diversity': diversity,
            'num_docs': len(context_docs)
        }
    
    def _calculate_doc_relevance(self, query: str, doc: Dict) -> float:
        """Calculate relevance of a document to query"""
        
        query_terms = set(query.lower().split())
        doc_terms = set(doc['content'].lower().split())
        
        # Term overlap
        overlap = len(query_terms & doc_terms)
        term_relevance = overlap / len(query_terms) if query_terms else 0
        
        # Distance-based relevance (lower distance = higher relevance)
        distance_relevance = 1 - doc['distance']
        
        # Metadata relevance
        metadata_relevance = self._calculate_metadata_relevance(query, doc['metadata'])
        
        # Combined relevance
        relevance = (term_relevance * 0.4 + distance_relevance * 0.4 + metadata_relevance * 0.2)
        
        return relevance
    
    def _calculate_metadata_relevance(self, query: str, metadata: Dict) -> float:
        """Calculate relevance based on metadata"""
        
        query_lower = query.lower()
        relevance = 0.0
        
        # Check for location mentions
        if 'entity_department' in metadata:
            dept = metadata['entity_department'].lower()
            if dept in query_lower:
                relevance += 0.3
        
        # Check for contract type mentions
        if 'contract_type' in metadata:
            contract_type = metadata['contract_type'].lower()
            if contract_type in query_lower:
                relevance += 0.2
        
        # Check for pricing mentions
        if 'contract_value' in metadata and any(term in query_lower for term in ['price', 'cost', 'value', 'budget']):
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def evaluate_performance(self, num_queries: int = 10) -> Dict[str, Any]:
        """Evaluate system performance metrics"""
        
        performance_metrics = {
            'response_times': [],
            'memory_usage': [],
            'throughput': 0,
            'error_rate': 0
        }
        
        start_time = time.time()
        errors = 0
        
        # Test queries
        test_queries = self.test_queries[:num_queries]
        
        for query_info in test_queries:
            try:
                query_start = time.time()
                
                # Execute query
                result = self.pipeline.query(query_info['query'])
                
                query_end = time.time()
                response_time = query_end - query_start
                
                performance_metrics['response_times'].append(response_time)
                
            except Exception as e:
                logger.error(f"Error in query: {str(e)}")
                errors += 1
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        performance_metrics['avg_response_time'] = np.mean(performance_metrics['response_times'])
        performance_metrics['max_response_time'] = np.max(performance_metrics['response_times'])
        performance_metrics['min_response_time'] = np.min(performance_metrics['response_times'])
        performance_metrics['throughput'] = num_queries / total_time
        performance_metrics['error_rate'] = errors / num_queries
        
        return performance_metrics
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation suite"""
        
        logger.info("Starting comprehensive evaluation...")
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'query_evaluations': [],
            'aggregate_metrics': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Evaluate each test query
        for query_info in self.test_queries:
            logger.info(f"Evaluating query: {query_info['query']}")
            
            try:
                # Execute query
                result = self.pipeline.query(query_info['query'])
                
                # Evaluate response quality
                quality_metrics = self.evaluate_response_quality(
                    query_info['query'],
                    result['response'],
                    query_info['type']
                )
                
                # Evaluate retrieval quality
                retrieval_metrics = self.evaluate_retrieval_quality(
                    query_info['query'],
                    result['context_docs']
                )
                
                # Combine metrics
                query_evaluation = {
                    'query': query_info['query'],
                    'query_type': query_info['type'],
                    'difficulty': query_info['difficulty'],
                    'response': result['response'],
                    'response_time': result['response_time'],
                    'quality_metrics': quality_metrics,
                    'retrieval_metrics': retrieval_metrics,
                    'num_context_docs': result['num_context_docs']
                }
                
                evaluation_results['query_evaluations'].append(query_evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating query: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        evaluation_results['aggregate_metrics'] = self._calculate_aggregate_metrics(
            evaluation_results['query_evaluations']
        )
        
        # Performance evaluation
        evaluation_results['performance_metrics'] = self.evaluate_performance()
        
        # Generate recommendations
        evaluation_results['recommendations'] = self._generate_recommendations(
            evaluation_results['aggregate_metrics'],
            evaluation_results['performance_metrics']
        )
        
        self.evaluation_results = evaluation_results
        
        logger.info("Comprehensive evaluation completed")
        return evaluation_results
    
    def _calculate_aggregate_metrics(self, query_evaluations: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all queries"""
        
        if not query_evaluations:
            return {}
        
        # Quality metrics
        quality_metrics = [eval['quality_metrics'] for eval in query_evaluations]
        
        aggregate_quality = {}
        for metric in quality_metrics[0].keys():
            values = [qm[metric] for qm in quality_metrics]
            aggregate_quality[f'avg_{metric}'] = np.mean(values)
            aggregate_quality[f'std_{metric}'] = np.std(values)
        
        # Retrieval metrics
        retrieval_metrics = [eval['retrieval_metrics'] for eval in query_evaluations]
        
        aggregate_retrieval = {}
        for metric in retrieval_metrics[0].keys():
            values = [rm[metric] for rm in retrieval_metrics]
            aggregate_retrieval[f'avg_{metric}'] = np.mean(values)
            aggregate_retrieval[f'std_{metric}'] = np.std(values)
        
        # Performance by query type
        query_types = {}
        for eval in query_evaluations:
            q_type = eval['query_type']
            if q_type not in query_types:
                query_types[q_type] = []
            query_types[q_type].append(eval['quality_metrics']['overall'])
        
        type_performance = {}
        for q_type, scores in query_types.items():
            type_performance[q_type] = {
                'avg_score': np.mean(scores),
                'count': len(scores)
            }
        
        return {
            'quality_metrics': aggregate_quality,
            'retrieval_metrics': aggregate_retrieval,
            'type_performance': type_performance,
            'total_queries': len(query_evaluations)
        }
    
    def _generate_recommendations(self, aggregate_metrics: Dict, performance_metrics: Dict) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Quality recommendations
        if aggregate_metrics.get('quality_metrics', {}).get('avg_overall', 0) < 0.7:
            recommendations.append("Overall quality is below threshold (0.7). Consider improving response generation.")
        
        if aggregate_metrics.get('quality_metrics', {}).get('avg_specificity', 0) < 0.6:
            recommendations.append("Responses lack specificity. Improve context retrieval and prompt engineering.")
        
        if aggregate_metrics.get('quality_metrics', {}).get('avg_relevance', 0) < 0.7:
            recommendations.append("Response relevance is low. Consider improving query understanding and retrieval.")
        
        # Performance recommendations
        if performance_metrics.get('avg_response_time', 0) > 5.0:
            recommendations.append("Response times are high. Consider model optimization or caching.")
        
        if performance_metrics.get('error_rate', 0) > 0.1:
            recommendations.append("Error rate is high. Improve error handling and system robustness.")
        
        # Retrieval recommendations
        if aggregate_metrics.get('retrieval_metrics', {}).get('avg_precision', 0) < 0.7:
            recommendations.append("Retrieval precision is low. Consider improving embedding quality or re-ranking.")
        
        if aggregate_metrics.get('retrieval_metrics', {}).get('avg_diversity', 0) < 0.3:
            recommendations.append("Retrieval diversity is low. Consider improving collection balancing.")
        
        return recommendations
    
    def save_evaluation_results(self, output_path: str):
        """Save evaluation results to file"""
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_evaluation_report(self, output_path: str):
        """Generate human-readable evaluation report"""
        
        if not self.evaluation_results:
            logger.error("No evaluation results available. Run evaluation first.")
            return
        
        report = []
        report.append("# AMRA Healthcare Tender Pricing System - Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        aggregate = self.evaluation_results['aggregate_metrics']
        
        if 'quality_metrics' in aggregate:
            overall_score = aggregate['quality_metrics'].get('avg_overall', 0)
            report.append(f"- Overall Quality Score: {overall_score:.3f}/1.000")
            report.append(f"- Total Queries Evaluated: {aggregate['total_queries']}")
        
        if 'performance_metrics' in self.evaluation_results:
            perf = self.evaluation_results['performance_metrics']
            report.append(f"- Average Response Time: {perf.get('avg_response_time', 0):.2f}s")
            report.append(f"- Error Rate: {perf.get('error_rate', 0):.1%}")
        
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        
        if 'quality_metrics' in aggregate:
            report.append("### Quality Metrics")
            quality = aggregate['quality_metrics']
            
            for metric, value in quality.items():
                if metric.startswith('avg_'):
                    metric_name = metric.replace('avg_', '').title()
                    report.append(f"- {metric_name}: {value:.3f}")
        
        # Recommendations
        if 'recommendations' in self.evaluation_results:
            report.append("\n## Recommendations")
            for rec in self.evaluation_results['recommendations']:
                report.append(f"- {rec}")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Evaluation report saved to {output_path}")


def main():
    """Run evaluation framework"""
    print("Evaluation Framework ready for use.")
    print("Initialize with TenderRAGPipeline instance to run evaluations.")


if __name__ == "__main__":
    main()