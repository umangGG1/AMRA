"""
Healthcare Contract Pricing Recommendation System
Advanced multi-collection search with BGE embeddings for accurate pricing recommendations
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import statistics
import requests
import os

from chromadb_manager import HealthcareChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricingRecommendationSystem:
    """Advanced pricing recommendation system using multi-collection vector search with LLM integration"""
    
    def __init__(self, chroma_persist_dir: str = "./chroma_db", use_llm: bool = True, use_mistral_api: bool = True):
        """Initialize the pricing recommendation system"""
        self.chroma_manager = HealthcareChromaManager(chroma_persist_dir)
        self.use_llm = use_llm
        self.use_mistral_api = use_mistral_api
        
        # LLM configuration
        if self.use_llm:
            if self.use_mistral_api:
                self.model_name = "mistral-small-latest"
                self.api_url = "https://api.mistral.ai/v1/chat/completions"
                self.mistral_api_key = os.getenv('MISTRAL_API_KEY')
                self.headers = {"Authorization": f"Bearer {self.mistral_api_key}"} if self.mistral_api_key else {}
            else:
                self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
                self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                self.api_token = os.getenv('HUGGINGFACE_API_TOKEN')
                self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        
        # Recommendation weights for different collection types
        self.collection_weights = {
            'pricing_context': 0.4,      # Highest weight for direct pricing matches
            'service_similarity': 0.3,   # Service similarity for comparison
            'geographic_pricing': 0.2,   # Regional pricing patterns
            'contractor_performance': 0.1 # Contractor-specific insights
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def setup_system(self, data_path: str = "./data/healthcare_enhanced_20250718_181121_streamlined_20250718_182651.json"):
        """Setup the complete pricing recommendation system"""
        logger.info("Setting up pricing recommendation system...")
        
        # Create collections if they don't exist
        self.chroma_manager.create_collections()
        
        # If data_path is provided, load and populate collections
        if data_path:
            # Load and process data
            df = self.chroma_manager.load_healthcare_data(data_path)
            df_processed = self.chroma_manager.preprocess_healthcare_data(df)
            
            # Populate collections
            self.chroma_manager.populate_collections(df_processed)
            
            # Store processed data for reference
            self.processed_data = df_processed
        
        logger.info("Pricing recommendation system setup complete")
        
        # Display system statistics
        stats = self.chroma_manager.get_collection_stats()
        logger.info("Collection Statistics:")
        for collection_name, info in stats.items():
            logger.info(f"  {collection_name}: {info['document_count']} documents")
    
    def get_pricing_recommendations(
        self,
        query_description: str,
        target_department: Optional[str] = None,
        target_entity_type: Optional[str] = None,
        price_range: Optional[str] = None,
        service_category: Optional[str] = None,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Get comprehensive pricing recommendations for a healthcare service
        
        Args:
            query_description: Description of the healthcare service
            target_department: Specific department for geographic filtering
            target_entity_type: Type of entity (ministry, institute, etc.)
            price_range: Target price range (0-50K, 50K-200K, etc.)
            service_category: Service category filter
            n_results: Number of results per collection
            
        Returns:
            Comprehensive pricing recommendation with confidence scores
        """
        logger.info(f"Generating pricing recommendations for: {query_description[:100]}...")
        
        # Build metadata filters
        metadata_filter = self._build_metadata_filter(
            target_department, target_entity_type, price_range, service_category
        )
        
        # Search across all collections
        collection_results = {}
        for collection_name in self.collection_weights.keys():
            try:
                results = self.chroma_manager.query_collection(
                    collection_name=collection_name,
                    query_text=query_description,
                    n_results=n_results,
                    where_filter=metadata_filter.get(collection_name)
                )
                collection_results[collection_name] = results
            except Exception as e:
                logger.warning(f"Error querying {collection_name}: {str(e)}")
                collection_results[collection_name] = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Process and combine results
        pricing_recommendation = self._process_search_results(
            collection_results, query_description, metadata_filter
        )
        
        return pricing_recommendation
    
    def _build_metadata_filter(
        self,
        target_department: Optional[str],
        target_entity_type: Optional[str],
        price_range: Optional[str],
        service_category: Optional[str]
    ) -> Dict[str, Dict]:
        """Build metadata filters for each collection"""
        filters = {}
        
        # Base filter for pricing context (ChromaDB requires single key filters)
        pricing_filter = None
        if target_department:
            pricing_filter = {'entity_department': target_department}
        elif price_range:
            pricing_filter = {'price_bracket': price_range}
        elif service_category:
            pricing_filter = {'service_category': service_category}
        
        filters['pricing_context'] = pricing_filter
        
        # Service similarity filter (less restrictive)
        service_filter = None
        if service_category:
            service_filter = {'service_category': service_category}
        
        filters['service_similarity'] = service_filter
        
        # Geographic filter
        geo_filter = None
        if target_department:
            geo_filter = {'entity_department': target_department}
        elif service_category:
            geo_filter = {'service_category': service_category}
        
        filters['geographic_pricing'] = geo_filter
        
        # Contractor filter
        contractor_filter = None
        if target_department:
            contractor_filter = {'entity_department': target_department}
        
        filters['contractor_performance'] = contractor_filter
        
        return filters
    
    def _process_search_results(
        self,
        collection_results: Dict[str, Any],
        query_description: str,
        metadata_filter: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Process and combine search results from all collections"""
        
        # Extract pricing information from results
        pricing_data = []
        similar_contracts = []
        
        for collection_name, results in collection_results.items():
            if not results['documents'] or not results['documents'][0]:
                continue
            
            weight = self.collection_weights[collection_name]
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Calculate similarity score (ChromaDB uses cosine distance)
                similarity_score = 1 - distance if distance is not None else 0
                weighted_score = similarity_score * weight
                
                contract_info = {
                    'collection': collection_name,
                    'document': doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'weighted_score': weighted_score,
                    'rank': i + 1
                }
                
                similar_contracts.append(contract_info)
                
                # Extract price if available
                if 'contract_value' in metadata and metadata['contract_value']:
                    pricing_data.append({
                        'value': float(metadata['contract_value']),
                        'similarity_score': similarity_score,
                        'weighted_score': weighted_score,
                        'collection': collection_name,
                        'metadata': metadata
                    })
        
        # Calculate pricing recommendations
        pricing_analysis = self._calculate_pricing_analysis(pricing_data)
        
        # Generate confidence assessment
        confidence_assessment = self._assess_confidence(pricing_data, similar_contracts)
        
        # Create recommendation summary
        recommendation = {
            'query': query_description,
            'pricing_analysis': pricing_analysis,
            'confidence_assessment': confidence_assessment,
            'similar_contracts': sorted(similar_contracts, key=lambda x: x['weighted_score'], reverse=True)[:10],
            'metadata_filters_applied': metadata_filter,
            'total_similar_contracts': len(similar_contracts),
            'total_pricing_references': len(pricing_data),
            'recommendation_timestamp': datetime.now().isoformat()
        }
        
        # Generate LLM response if enabled
        if self.use_llm:
            llm_response = self._generate_llm_recommendation(
                query_description, pricing_analysis, similar_contracts, confidence_assessment
            )
            recommendation['llm_response'] = llm_response
        
        return recommendation
    
    def _generate_llm_recommendation(self, query: str, pricing_analysis: Dict, 
                                   similar_contracts: List[Dict], confidence_assessment: Dict) -> str:
        """Generate natural language recommendation using LLM"""
        try:
            # Prepare context for LLM
            context_parts = []
            
            # Add pricing analysis
            if pricing_analysis.get('recommended_price'):
                context_parts.append(f"Recommended Price: ${pricing_analysis['recommended_price']:,.0f}")
            
            if pricing_analysis.get('price_range'):
                range_info = pricing_analysis['price_range']
                if range_info.get('min') and range_info.get('max'):
                    context_parts.append(f"Price Range: ${range_info['min']:,.0f} - ${range_info['max']:,.0f}")
            
            context_parts.append(f"Confidence Level: {pricing_analysis.get('confidence', 'unknown')}")
            context_parts.append(f"Based on: {pricing_analysis.get('basis', 'similar contracts')}")
            
            # Add top similar contracts
            if similar_contracts:
                context_parts.append("\nTop Similar Contracts:")
                for i, contract in enumerate(similar_contracts[:3]):
                    metadata = contract.get('metadata', {})
                    value = metadata.get('contract_value', 'N/A')
                    dept = metadata.get('entity_department', 'Unknown')
                    context_parts.append(f"{i+1}. ${value:,.0f} - {dept} - Score: {contract['weighted_score']:.3f}")
            
            # Add confidence assessment
            if confidence_assessment:
                context_parts.append(f"\nConfidence Assessment:")
                context_parts.append(f"Overall Score: {confidence_assessment.get('overall_score', 0):.1%}")
                context_parts.append(f"Average Similarity: {confidence_assessment.get('average_similarity', 0):.1%}")
            
            context = "\n".join(context_parts)
            
            # Create prompt
            prompt = f"""You are a healthcare tender pricing specialist. Based on the following analysis, provide a comprehensive pricing recommendation.

Query: {query}

Analysis Results:
{context}

Please provide a detailed pricing recommendation that includes:
1. The recommended price and justification
2. Price range considerations
3. Confidence level explanation
4. Key factors affecting the pricing
5. Recommendations for negotiation or further analysis

Recommendation:"""
            
            # Call LLM API
            if self.use_mistral_api:
                # Mistral AI API format
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3,
                    "stream": False
                }
                
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.mistral_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.warning("No response from Mistral API")
                        return self._generate_fallback_recommendation(pricing_analysis, similar_contracts)
                else:
                    logger.warning(f"Mistral API Error: {response.status_code} - {response.text}")
                    return self._generate_fallback_recommendation(pricing_analysis, similar_contracts)
            else:
                # HuggingFace API format
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 512,
                        "temperature": 0.3,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        return result.get('generated_text', '')
                    else:
                        return str(result)
                else:
                    logger.warning(f"HuggingFace API Error: {response.status_code}")
                    return self._generate_fallback_recommendation(pricing_analysis, similar_contracts)
                
        except Exception as e:
            logger.error(f"Error generating LLM recommendation: {str(e)}")
            return self._generate_fallback_recommendation(pricing_analysis, similar_contracts)
    
    def _generate_fallback_recommendation(self, pricing_analysis: Dict, similar_contracts: List[Dict]) -> str:
        """Generate fallback recommendation without LLM"""
        recommendation = []
        
        if pricing_analysis.get('recommended_price'):
            recommendation.append(f"Based on the analysis of {len(similar_contracts)} similar contracts, the recommended price is ${pricing_analysis['recommended_price']:,.0f}.")
        
        if pricing_analysis.get('price_range'):
            range_info = pricing_analysis['price_range']
            if range_info.get('min') and range_info.get('max'):
                recommendation.append(f"The expected price range is between ${range_info['min']:,.0f} and ${range_info['max']:,.0f}.")
        
        recommendation.append(f"Confidence level: {pricing_analysis.get('confidence', 'unknown')}")
        
        if similar_contracts:
            recommendation.append(f"This recommendation is based on {len(similar_contracts)} similar healthcare contracts in the database.")
        
        return " ".join(recommendation)
    
    def _calculate_pricing_analysis(self, pricing_data: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive pricing analysis"""
        if not pricing_data:
            return {
                'recommended_price': None,
                'price_range': {'min': None, 'max': None},
                'confidence': 'low',
                'basis': 'insufficient_data'
            }
        
        # Extract prices and weights
        prices = [item['value'] for item in pricing_data]
        weights = [item['weighted_score'] for item in pricing_data]
        
        # Calculate weighted statistics
        weighted_avg = np.average(prices, weights=weights)
        
        # Calculate percentiles for range estimation
        price_25th = np.percentile(prices, 25)
        price_75th = np.percentile(prices, 75)
        
        # Price distribution analysis
        distribution_analysis = {
            'mean': np.mean(prices),
            'median': np.median(prices),
            'weighted_mean': weighted_avg,
            'std_dev': np.std(prices),
            'min': min(prices),
            'max': max(prices),
            'q25': price_25th,
            'q75': price_75th,
            'count': len(prices)
        }
        
        # Determine recommended price (weighted average with adjustments)
        recommended_price = self._adjust_price_recommendation(weighted_avg, distribution_analysis)
        
        # Determine confidence level
        confidence = self._determine_price_confidence(pricing_data, distribution_analysis)
        
        return {
            'recommended_price': round(recommended_price, 0),
            'price_range': {
                'min': round(price_25th, 0),
                'max': round(price_75th, 0)
            },
            'distribution': distribution_analysis,
            'confidence': confidence,
            'basis': f'{len(pricing_data)}_similar_contracts'
        }
    
    def _adjust_price_recommendation(self, base_price: float, distribution: Dict) -> float:
        """Adjust price recommendation based on distribution characteristics"""
        
        # If there's high variance, be more conservative
        cv = distribution['std_dev'] / distribution['mean'] if distribution['mean'] > 0 else 0
        
        if cv > 0.5:  # High variance
            # Use median instead of mean for more stability
            adjusted_price = distribution['median']
        else:
            # Use weighted mean for stable distributions
            adjusted_price = base_price
        
        # Apply market adjustments (could be configurable)
        # For now, no additional adjustments
        
        return adjusted_price
    
    def _determine_price_confidence(self, pricing_data: List[Dict], distribution: Dict) -> str:
        """Determine confidence level for price recommendation"""
        
        # Factors for confidence:
        # 1. Number of similar contracts
        # 2. Quality of similarity scores
        # 3. Price distribution consistency
        
        count_score = min(len(pricing_data) / 10, 1.0)  # Up to 10 contracts gives full score
        
        # Average similarity score
        avg_similarity = np.mean([item['similarity_score'] for item in pricing_data])
        similarity_score = avg_similarity
        
        # Distribution consistency (lower CV = higher confidence)
        cv = distribution['std_dev'] / distribution['mean'] if distribution['mean'] > 0 else 1.0
        consistency_score = max(0, 1 - cv)
        
        # Combined confidence score
        confidence_score = (count_score * 0.4 + similarity_score * 0.4 + consistency_score * 0.2)
        
        if confidence_score >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_confidence(self, pricing_data: List[Dict], similar_contracts: List[Dict]) -> Dict[str, Any]:
        """Assess overall confidence in the recommendation"""
        
        total_contracts = len(similar_contracts)
        contracts_with_pricing = len(pricing_data)
        
        # Collection coverage
        collections_used = set(contract['collection'] for contract in similar_contracts)
        coverage_score = len(collections_used) / len(self.collection_weights)
        
        # Average similarity across all results
        if similar_contracts:
            avg_similarity = np.mean([contract['similarity_score'] for contract in similar_contracts])
        else:
            avg_similarity = 0
        
        # Data completeness
        data_completeness = contracts_with_pricing / max(total_contracts, 1)
        
        return {
            'overall_score': (coverage_score * 0.3 + avg_similarity * 0.4 + data_completeness * 0.3),
            'collection_coverage': coverage_score,
            'average_similarity': avg_similarity,
            'data_completeness': data_completeness,
            'collections_used': list(collections_used),
            'contracts_analyzed': total_contracts,
            'contracts_with_pricing': contracts_with_pricing
        }
    
    def get_comparative_analysis(
        self,
        service_description: str,
        comparison_filters: List[Dict[str, str]],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Get comparative pricing analysis across different scenarios
        
        Args:
            service_description: Description of the service
            comparison_filters: List of filter scenarios to compare
            n_results: Number of results per scenario
            
        Returns:
            Comparative analysis across scenarios
        """
        comparisons = {}
        
        for i, filters in enumerate(comparison_filters):
            scenario_name = f"scenario_{i+1}"
            
            recommendation = self.get_pricing_recommendations(
                query_description=service_description,
                target_department=filters.get('department'),
                target_entity_type=filters.get('entity_type'),
                price_range=filters.get('price_range'),
                service_category=filters.get('service_category'),
                n_results=n_results
            )
            
            comparisons[scenario_name] = {
                'filters': filters,
                'recommendation': recommendation
            }
        
        return {
            'service_description': service_description,
            'comparisons': comparisons,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def export_recommendations(self, recommendations: Dict[str, Any], output_path: str):
        """Export recommendations to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Recommendations exported to {output_path}")


def main():
    """Example usage of the pricing recommendation system"""
    
    # Initialize system
    pricing_system = PricingRecommendationSystem()
    
    # Setup with healthcare data (use existing data if available)
    data_path = "/home/umanggod/AMRA-healthcare-POC/data/healthcare_contracts_cleaned_20250717_230006.json"
    pricing_system.setup_system(data_path)
    
    # Example query
    query = "Professional healthcare support services financial advisory and management"
    
    print(f"\n=== Pricing Recommendation for: {query} ===")
    
    # Get pricing recommendations
    recommendations = pricing_system.get_pricing_recommendations(
        query_description=query,
        target_department="Bogotá D.C.",
        service_category="professional_services",
        n_results=5
    )
    
    # Display results
    pricing = recommendations['pricing_analysis']
    confidence = recommendations['confidence_assessment']
    
    print(f"\nRecommended Price: ${pricing['recommended_price']:,.0f}")
    print(f"Price Range: ${pricing['price_range']['min']:,.0f} - ${pricing['price_range']['max']:,.0f}")
    print(f"Confidence: {pricing['confidence']}")
    print(f"Based on: {pricing['basis']}")
    
    print(f"\nConfidence Assessment:")
    print(f"  Overall Score: {confidence['overall_score']:.2f}")
    print(f"  Collection Coverage: {confidence['collection_coverage']:.2f}")
    print(f"  Average Similarity: {confidence['average_similarity']:.2f}")
    print(f"  Data Completeness: {confidence['data_completeness']:.2f}")
    
    print(f"\nTop Similar Contracts:")
    for i, contract in enumerate(recommendations['similar_contracts'][:3]):
        print(f"  {i+1}. Score: {contract['weighted_score']:.3f} | {contract['document'][:100]}...")
    
    # Export results
    output_path = "/home/umanggod/AMRA-healthcare-POC/data/processed/pricing_recommendation_example.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    pricing_system.export_recommendations(recommendations, output_path)


if __name__ == "__main__":
    main()