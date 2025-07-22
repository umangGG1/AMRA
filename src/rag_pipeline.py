"""
Enhanced RAG Pipeline for Healthcare Contract Pricing
Implements high-priority improvements: larger model, data validation, 
structured responses, and enhanced context management
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from datetime import datetime
import requests
import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re

# Core ML and NLP
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# RAG components
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager

# Local imports
from chromadb_manager import HealthcareChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enhanced query type classification"""
    SPECIFIC_PRICING = "specific_pricing"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    BUDGET_ESTIMATION = "budget_estimation"
    MARKET_RESEARCH = "market_research"
    TREND_ANALYSIS = "trend_analysis"
    GENERAL_INQUIRY = "general_inquiry"

@dataclass
class PricingResponse:
    """Structured pricing response format"""
    primary_price: Optional[float] = None
    price_range: Optional[Tuple[float, float]] = None
    comparable_contracts: List[Dict] = None
    factors_affecting_price: List[str] = None
    confidence_score: float = 0.0
    data_source_count: int = 0
    response_text: str = ""
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.comparable_contracts is None:
            self.comparable_contracts = []
        if self.factors_affecting_price is None:
            self.factors_affecting_price = []
        if self.recommendations is None:
            self.recommendations = []

class DataValidator:
    """Enhanced data validation and preprocessing"""
    
    @staticmethod
    def validate_pricing_data(df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        logger.info("Starting data validation and cleaning...")
        
        # Create copy to avoid modifying original
        df_clean = df.copy()
        
        # Standardize currency and numeric fields
        df_clean = DataValidator._standardize_currency(df_clean)
        
        # Remove outliers
        df_clean = DataValidator._remove_price_outliers(df_clean)
        
        # Normalize service categories
        df_clean = DataValidator._normalize_service_categories(df_clean)
        
        # Handle missing values
        df_clean = DataValidator._handle_missing_values(df_clean)
        
        # Add computed fields
        df_clean = DataValidator._add_computed_fields(df_clean)
        
        logger.info(f"Data validation complete. Records: {len(df_clean)}")
        return df_clean
    
    @staticmethod
    def _standardize_currency(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize currency formats"""
        if 'contract_value' in df.columns:
            # Remove currency symbols and convert to numeric
            df['contract_value'] = df['contract_value'].astype(str).str.replace(r'[$,€£¥]', '', regex=True)
            df['contract_value'] = pd.to_numeric(df['contract_value'], errors='coerce')
            
            # Convert to USD if needed (simplified - in production use live rates)
            df['contract_value_usd'] = df['contract_value']
        
        return df
    
    @staticmethod
    def _remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """Remove price outliers using IQR method"""
        if 'contract_value' in df.columns:
            Q1 = df['contract_value'].quantile(0.25)
            Q3 = df['contract_value'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_removed = len(df) - len(df[(df['contract_value'] >= lower_bound) & 
                                               (df['contract_value'] <= upper_bound)])
            
            df = df[(df['contract_value'] >= lower_bound) & (df['contract_value'] <= upper_bound)]
            
            if outliers_removed > 0:
                logger.info(f"Removed {outliers_removed} price outliers")
        
        return df
    
    @staticmethod
    def _normalize_service_categories(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and standardize service categories"""
        service_mapping = {
            'medical equipment': ['medical devices', 'hospital equipment', 'diagnostic tools'],
            'nursing services': ['healthcare personnel', 'nursing staff', 'patient care'],
            'pharmaceutical': ['drugs', 'medications', 'medical supplies'],
            'consulting': ['advisory services', 'management consulting', 'technical assistance'],
            'maintenance': ['equipment maintenance', 'facility maintenance', 'support services']
        }
        
        if 'service_category' in df.columns:
            df['service_category_normalized'] = df['service_category'].str.lower()
            
            for main_category, variations in service_mapping.items():
                mask = df['service_category_normalized'].str.contains('|'.join(variations), na=False)
                df.loc[mask, 'service_category_normalized'] = main_category
        
        return df
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values strategically"""
        # Fill missing contract values with median by service type
        if 'contract_value' in df.columns and 'service_category_normalized' in df.columns:
            df['contract_value'] = df.groupby('service_category_normalized')['contract_value'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Fill missing locations
        if 'entity_department' in df.columns:
            df['entity_department'] = df['entity_department'].fillna('Unknown Location')
        
        return df
    
    @staticmethod
    def _add_computed_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Add computed fields for enhanced analysis"""
        # Price per unit calculations (if applicable)
        if 'contract_value' in df.columns and 'quantity' in df.columns:
            df['price_per_unit'] = df['contract_value'] / df['quantity'].replace(0, np.nan)
        
        # Contract size category
        if 'contract_value' in df.columns:
            df['contract_size'] = pd.cut(
                df['contract_value'], 
                bins=[0, 10000, 50000, 200000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Extra Large']
            )
        
        # Add hash for deduplication
        text_columns = ['contract_title', 'entity_department', 'service_category']
        available_columns = [col for col in text_columns if col in df.columns]
        if available_columns:
            df['content_hash'] = df[available_columns].apply(
                lambda x: hashlib.md5(''.join(x.astype(str)).encode()).hexdigest(), axis=1
            )
        
        return df

class EnhancedQueryClassifier:
    """ML-enhanced query classification"""
    
    def __init__(self):
        self.classification_patterns = {
            QueryType.SPECIFIC_PRICING: [
                r'\b(price|cost|value|amount)\s+of\b',
                r'\bhow much\b',
                r'\bwhat.*cost',
                r'\bprice for\b'
            ],
            QueryType.COMPARATIVE_ANALYSIS: [
                r'\bcompare|comparison|versus|vs\b',
                r'\bdifference between\b',
                r'\bwhich is cheaper\b',
                r'\bbetter value\b'
            ],
            QueryType.BUDGET_ESTIMATION: [
                r'\bestimate|estimation|approximate\b',
                r'\bbudget for\b',
                r'\bwhat should.*cost\b',
                r'\bexpected cost\b'
            ],
            QueryType.MARKET_RESEARCH: [
                r'\bmarket rate|market price\b',
                r'\bindustry standard\b',
                r'\baverage cost\b',
                r'\btypical price\b'
            ],
            QueryType.TREND_ANALYSIS: [
                r'\btrend|trending\b',
                r'\bover time\b',
                r'\bhistorical\b',
                r'\bchange in price\b'
            ]
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query using pattern matching and ML scoring"""
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        
        # Return highest scoring type, default to general inquiry
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return QueryType.GENERAL_INQUIRY

class EnhancedTenderRAGPipeline:
    """Complete enhanced RAG pipeline with high-priority improvements"""
    
    def __init__(self, 
                 model_name: str = "mistral-small-latest",  # Mistral AI model
                 chroma_persist_dir: str = "./chroma_db",
                 max_tokens: int = 1024,  # Increased for better responses
                 temperature: float = 0.1,
                 context_window: int = 4000,  # Enhanced context window
                 use_api: bool = True,
                 use_mistral_api: bool = True,  # Use Mistral API instead of HF
                 api_token: Optional[str] = None,
                 mistral_api_key: Optional[str] = None):
        """
        Initialize enhanced RAG pipeline
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window = context_window
        self.use_api = use_api
        self.use_mistral_api = use_mistral_api
        self.api_token = api_token or os.getenv('HUGGINGFACE_API_TOKEN')
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
        
        # Initialize components
        self.chroma_manager = HealthcareChromaManager(chroma_persist_dir)
        self.data_validator = DataValidator()
        self.query_classifier = EnhancedQueryClassifier()
        
        # Model fallback hierarchy
        if self.use_mistral_api:
            self.model_hierarchy = [
                "mistral-small-latest",
                "mistral-tiny",
                "open-mistral-7b"
            ]
        else:
            self.model_hierarchy = [
                "mistralai/Mistral-7B-Instruct-v0.1",
                "microsoft/DialoGPT-medium",
                "gpt2"
            ]
        self.current_model_index = 0
        
        # API configuration
        if self.use_api:
            if self.use_mistral_api:
                self.api_url = "https://api.mistral.ai/v1/chat/completions"
                self.headers = {"Authorization": f"Bearer {self.mistral_api_key}"} if self.mistral_api_key else {}
                logger.info(f"Enhanced RAG Pipeline initialized with Mistral API: {model_name}")
            else:
                self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
                self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
                logger.info(f"Enhanced RAG Pipeline initialized with HuggingFace API: {model_name}")
        else:
            self.tokenizer = None
            self.model = None
            self.pipeline = None
            logger.info(f"Enhanced RAG Pipeline initialized with local model: {model_name}")
        
        # Enhanced query templates
        self.query_templates = self._create_enhanced_templates()
        
        # Response cache
        self.response_cache = {}
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_response_time': 0,
            'cache_hits': 0,
            'model_fallbacks': 0
        }
    
    def _create_enhanced_templates(self) -> Dict[QueryType, PromptTemplate]:
        """Create enhanced query templates with structured output"""
        templates = {
            QueryType.SPECIFIC_PRICING: PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert healthcare procurement analyst specializing in the Colombian market. Analyze the contract data to provide specific pricing information.

IMPORTANT: All prices are in Colombian Pesos (COP). Always format monetary values as "COP $XX,XXX,XXX" and never use USD.

CONTRACT DATA:
{context}

QUERY: {question}

Provide a structured response with:
1. SPECIFIC PRICE: The exact price if available
2. PRICE RANGE: Minimum to maximum range from similar contracts
3. KEY FACTORS: What influences this pricing
4. CONFIDENCE: How reliable this information is (High/Medium/Low)
5. DATA SOURCES: Number of contracts used for this analysis

FORMAT YOUR RESPONSE AS:
**Primary Price:** $X,XXX
**Price Range:** $X,XXX - $X,XXX
**Confidence:** [High/Medium/Low] based on [X] contracts
**Key Factors:** 
- Factor 1
- Factor 2
**Recommendations:**
- Recommendation 1
- Recommendation 2

Response:"""
            ),
            
            QueryType.COMPARATIVE_ANALYSIS: PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a healthcare contract analyst specializing in comparative pricing analysis.

CONTRACT DATA:
{context}

COMPARISON QUERY: {question}

Provide a detailed comparison including:
1. PRICE COMPARISON: Direct price comparisons
2. VALUE ANALYSIS: Which offers better value
3. REGIONAL DIFFERENCES: Location-based pricing variations
4. SERVICE DIFFERENCES: What accounts for price differences

FORMAT AS:
**Comparison Summary:**
- Option A: $X,XXX [Location/Service details]
- Option B: $Y,YYY [Location/Service details]

**Value Analysis:**
[Best value recommendation with reasoning]

**Key Differences:**
- Difference 1
- Difference 2

Response:"""
            ),
            
            QueryType.BUDGET_ESTIMATION: PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a healthcare budget planning specialist for the Colombian market. Use the contract data to provide accurate budget estimates.

IMPORTANT: All prices are in Colombian Pesos (COP). Format all monetary values as "COP $XX,XXX,XXX" and never use USD.

CONTRACT DATA:
{context}

BUDGET QUERY: {question}

Provide a comprehensive budget estimate:
1. ESTIMATED COST: Based on similar contracts
2. COST RANGE: Conservative to optimistic estimates
3. RISK FACTORS: What could affect the budget
4. PLANNING RECOMMENDATIONS: How to prepare the budget

FORMAT AS:
**Budget Estimate:** COP $X,XXX,XXX - COP $Y,YYY,YYY
**Recommended Budget:** COP $Z,ZZZ,ZZZ (includes 15% contingency)
**Based on:** [X] similar contracts
**Risk Factors:**
- Risk 1
- Risk 2
**Budget Planning Tips:**
- Tip 1
- Tip 2

Response:"""
            ),
            
            QueryType.MARKET_RESEARCH: PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a healthcare market research analyst. Analyze the contract data to provide market insights.

CONTRACT DATA:
{context}

MARKET QUERY: {question}

Provide market analysis including:
1. MARKET RATES: Current market pricing
2. PRICE TRENDS: How prices are changing
3. MARKET FACTORS: What drives pricing in this market
4. BENCHMARKING: How to compare against market rates

Response:"""
            ),
            
            QueryType.GENERAL_INQUIRY: PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a healthcare contract expert. Answer the query using the provided contract data.

CONTRACT DATA:
{context}

QUERY: {question}

Provide a comprehensive answer including relevant contract details, pricing information, and actionable insights.

Response:"""
            )
        }
        
        return templates
    
    def setup_enhanced_database(self, data_path: str, reset: bool = False) -> Dict:
        """Setup database with enhanced data validation"""
        try:
            if reset:
                self.chroma_manager.reset_collections()
            
            # Create collections
            self.chroma_manager.create_collections()
            
            # Load and validate data
            logger.info("Loading contract data...")
            df = self.chroma_manager.load_healthcare_data(data_path)
            
            logger.info("Validating and preprocessing data...")
            df_validated = self.data_validator.validate_pricing_data(df)
            
            # Enhanced preprocessing
            df_processed = self.chroma_manager.preprocess_healthcare_data(df_validated)
            
            # Populate collections with enhanced data
            self.chroma_manager.populate_collections(df_processed)
            
            # Get statistics
            stats = self.chroma_manager.get_collection_stats()
            
            # Add validation stats
            stats['validation'] = {
                'original_records': len(df),
                'processed_records': len(df_processed),
                'data_quality_score': self._calculate_data_quality_score(df_processed)
            }
            
            logger.info(f"Enhanced database setup complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error setting up enhanced database: {str(e)}")
            raise
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        scores = []
        
        # Completeness score
        completeness = df.notna().mean().mean()
        scores.append(completeness)
        
        # Price consistency score
        if 'contract_value' in df.columns:
            price_cv = df['contract_value'].std() / df['contract_value'].mean()
            consistency = max(0, 1 - price_cv / 2)  # Normalize coefficient of variation
            scores.append(consistency)
        
        # Uniqueness score (based on content hash if available)
        if 'content_hash' in df.columns and len(df) > 0:
            uniqueness = df['content_hash'].nunique() / len(df)
            scores.append(uniqueness)
        
        return np.mean(scores)
    
    def enhanced_retrieval(self, 
                          query: str, 
                          n_results: int = 15,  # Increased for better context
                          filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced multi-stage retrieval with reranking"""
        try:
            # Stage 1: Broad semantic search
            broad_results = self._stage1_semantic_search(query, n_results * 2)
            
            # Stage 2: Filter by metadata if filters provided
            filtered_results = self._stage2_metadata_filtering(broad_results, filters)
            
            # Stage 3: Rerank by relevance and recency
            reranked_results = self._stage3_reranking(filtered_results, query, n_results)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {str(e)}")
            return []
    
    def _stage1_semantic_search(self, query: str, n_results: int) -> List[Dict]:
        """Stage 1: Broad semantic search across collections"""
        all_results = []
        collections = ['pricing_context', 'service_similarity']
        
        # Ensure collections are loaded
        self.chroma_manager.create_collections()
        
        for collection_name in collections:
            try:
                results = self.chroma_manager.query_collection(
                    collection_name=collection_name,
                    query_text=query,
                    n_results=n_results // len(collections),
                    where_filter=None
                )
                
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    all_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'collection': collection_name,
                        'stage1_rank': i
                    })
            except Exception as e:
                logger.warning(f"Error searching collection {collection_name}: {str(e)}")
        
        return all_results
    
    def _stage2_metadata_filtering(self, results: List[Dict], filters: Optional[Dict]) -> List[Dict]:
        """Stage 2: Filter by metadata constraints"""
        if not filters:
            return results
        
        filtered = []
        for result in results:
            metadata = result['metadata']
            include = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            include = False
                            break
                    elif metadata[key] != value:
                        include = False
                        break
            
            if include:
                filtered.append(result)
        
        return filtered
    
    def _stage3_reranking(self, results: List[Dict], query: str, n_results: int) -> List[Dict]:
        """Stage 3: Rerank by combined relevance score"""
        for result in results:
            # Calculate combined relevance score
            distance_score = 1 / (1 + result['distance'])  # Lower distance = higher score
            recency_score = self._calculate_recency_score(result['metadata'])
            completeness_score = self._calculate_completeness_score(result['metadata'])
            
            # Weighted combination
            result['combined_score'] = (
                0.5 * distance_score + 
                0.3 * completeness_score + 
                0.2 * recency_score
            )
        
        # Sort by combined score and return top results
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:n_results]
    
    def _calculate_recency_score(self, metadata: Dict) -> float:
        """Calculate recency score based on contract date"""
        if 'contract_date' in metadata and metadata['contract_date']:
            try:
                contract_date = pd.to_datetime(metadata['contract_date'])
                days_old = (datetime.now() - contract_date).days
                # More recent contracts get higher scores (decay over 3 years)
                return max(0, 1 - days_old / (3 * 365))
            except:
                return 0.5
        return 0.5  # Default score if no date
    
    def _calculate_completeness_score(self, metadata: Dict) -> float:
        """Calculate completeness score based on available metadata"""
        important_fields = ['contract_value', 'entity_department', 'service_category', 'contractor_name']
        available_fields = sum(1 for field in important_fields if field in metadata and metadata[field])
        return available_fields / len(important_fields)
    
    def generate_structured_response(self, 
                                   query: str, 
                                   context_docs: List[Dict],
                                   query_type: QueryType) -> PricingResponse:
        """Generate structured response with validation"""
        try:
            # Prepare enhanced context
            context = self._prepare_enhanced_context(context_docs)
            
            # Get appropriate template
            template = self.query_templates[query_type]
            prompt = template.format(context=context, question=query)
            
            # Generate response with fallback
            generated_text = self._generate_with_fallback(prompt)
            
            # Parse response into structured format
            structured_response = self._parse_structured_response(
                generated_text, context_docs, query_type
            )
            
            # Calculate confidence score
            structured_response.confidence_score = self._calculate_confidence_score(
                context_docs, structured_response
            )
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error generating structured response: {str(e)}")
            return PricingResponse(
                response_text=f"I encountered an error processing your query: {str(e)}",
                confidence_score=0.0
            )
    
    def _generate_with_fallback(self, prompt: str) -> str:
        """Generate response with model fallback hierarchy"""
        for attempt in range(len(self.model_hierarchy)):
            try:
                if self.use_api:
                    if self.use_mistral_api:
                        return self._call_mistral_api(prompt)
                    else:
                        return self._call_enhanced_api(prompt)
                else:
                    if not self.pipeline:
                        self.initialize_model()
                    response = self.pipeline(prompt)
                    return response[0]['generated_text']
                    
            except Exception as e:
                logger.warning(f"Model attempt {attempt + 1} failed: {str(e)}")
                if attempt < len(self.model_hierarchy) - 1:
                    # Try next model in hierarchy
                    self.current_model_index = attempt + 1
                    self.model_name = self.model_hierarchy[self.current_model_index]
                    if self.use_mistral_api:
                        # Keep same URL for Mistral API, just change model name
                        pass
                    else:
                        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                    self.query_stats['model_fallbacks'] += 1
                    continue
                else:
                    # All models failed, return error
                    return f"Unable to process query after trying all available models: {str(e)}"
        
        return "Unable to generate response"
    
    def _call_enhanced_api(self, prompt: str) -> str:
        """Enhanced API call with better error handling"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True,
                    "return_full_text": False,
                    "stop": ["Question:", "Query:", "Context:"]  # Stop sequences
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60  # Increased timeout
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
                raise Exception(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("API request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def _call_mistral_api(self, prompt: str) -> str:
        """Call Mistral AI API directly"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.mistral_api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception("No response from Mistral API")
            else:
                raise Exception(f"Mistral API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Mistral API request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Mistral API request failed: {str(e)}")
    
    def _prepare_enhanced_context(self, context_docs: List[Dict]) -> str:
        """Prepare enhanced context with better formatting"""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(context_docs):
            doc_content = doc['content']
            metadata = doc['metadata']
            
            # Enhanced context formatting
            context_part = f"=== CONTRACT {i+1} ===\n"
            context_part += f"Content: {doc_content}\n"
            
            # Add structured metadata
            if 'contract_value' in metadata and metadata['contract_value']:
                try:
                    value = float(metadata['contract_value'])
                    context_part += f"Value: ${value:,.2f}\n"
                except:
                    context_part += f"Value: {metadata['contract_value']}\n"
            
            if 'entity_department' in metadata and metadata['entity_department']:
                context_part += f"Location: {metadata['entity_department']}\n"
            
            if 'service_category' in metadata and metadata['service_category']:
                context_part += f"Service: {metadata['service_category']}\n"
            
            if 'contractor_name' in metadata and metadata['contractor_name']:
                context_part += f"Contractor: {metadata['contractor_name']}\n"
            
            # Add relevance score
            if 'combined_score' in doc:
                context_part += f"Relevance: {doc['combined_score']:.2f}\n"
            
            context_part += "\n"
            
            # Check length constraint
            if current_length + len(context_part) > self.context_window:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def _parse_structured_response(self, 
                                 response_text: str, 
                                 context_docs: List[Dict],
                                 query_type: QueryType) -> PricingResponse:
        """Parse generated text into structured response"""
        structured = PricingResponse()
        structured.response_text = response_text.strip()
        structured.data_source_count = len(context_docs)
        
        # Extract pricing information using regex
        price_patterns = [
            r'\$([0-9,]+(?:\.[0-9]{2})?)',
            r'([0-9,]+(?:\.[0-9]{2})?) dollars?',
            r'USD ([0-9,]+(?:\.[0-9]{2})?)'
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    prices.append(price)
                except:
                    continue
        
        if prices:
            structured.primary_price = prices[0]
            if len(prices) > 1:
                structured.price_range = (min(prices), max(prices))
        
        # Extract factors and recommendations
        factors = self._extract_bullet_points(response_text, ['factors', 'influences', 'affects'])
        recommendations = self._extract_bullet_points(response_text, ['recommend', 'suggest', 'tip'])
        
        structured.factors_affecting_price = factors
        structured.recommendations = recommendations
        
        # Extract comparable contracts
        structured.comparable_contracts = self._extract_comparable_contracts(context_docs)
        
        return structured
    
    def _extract_bullet_points(self, text: str, keywords: List[str]) -> List[str]:
        """Extract bullet points related to keywords"""
        points = []
        lines = text.split('\n')
        
        in_relevant_section = False
        for line in lines:
            line = line.strip()
            
            # Check if we're entering a relevant section
            if any(keyword in line.lower() for keyword in keywords):
                in_relevant_section = True
                continue
            
            # Extract bullet points in relevant sections
            if in_relevant_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                points.append(line[1:].strip())
            elif in_relevant_section and line and not line.startswith(('-', '•', '*')):
                # End of bullet point section
                break
        
        return points[:5]  # Limit to 5 points
    
    def _extract_comparable_contracts(self, context_docs: List[Dict]) -> List[Dict]:
        """Extract comparable contract information"""
        comparables = []
        
        for doc in context_docs[:3]:  # Top 3 most relevant
            metadata = doc['metadata']
            comparable = {}
            
            if 'contract_value' in metadata and metadata['contract_value']:
                comparable['value'] = metadata['contract_value']
            
            if 'entity_department' in metadata and metadata['entity_department']:
                comparable['location'] = metadata['entity_department']
            
            if 'service_category' in metadata and metadata['service_category']:
                comparable['service'] = metadata['service_category']
            
            if 'contractor_name' in metadata and metadata['contractor_name']:
                comparable['contractor'] = metadata['contractor_name']
            
            if comparable:  # Only add if we have some data
                comparables.append(comparable)
        
        return comparables
    
    def _calculate_confidence_score(self, 
                                  context_docs: List[Dict], 
                                  response: PricingResponse) -> float:
        """Calculate confidence score based on data quality and completeness"""
        factors = []
        
        # Data source count factor
        source_factor = min(1.0, len(context_docs) / 5)  # Full confidence with 5+ sources
        factors.append(source_factor)
        
        # Relevance factor (average of top 3 documents)
        if context_docs:
            relevance_scores = [doc.get('combined_score', 0.5) for doc in context_docs[:3]]
            relevance_factor = np.mean(relevance_scores)
            factors.append(relevance_factor)
        
        # Data completeness factor
        completeness_scores = []
        for doc in context_docs[:5]:
            metadata = doc['metadata']
            important_fields = ['contract_value', 'entity_department', 'service_category']
            available = sum(1 for field in important_fields if field in metadata and metadata[field])
            completeness_scores.append(available / len(important_fields))
        
        if completeness_scores:
            completeness_factor = np.mean(completeness_scores)
            factors.append(completeness_factor)
        
        # Price consistency factor (if multiple prices found)
        if response.comparable_contracts:
            prices = []
            for contract in response.comparable_contracts:
                if 'value' in contract:
                    try:
                        price = float(str(contract['value']).replace(',', '').replace('$', ''))
                        prices.append(price)
                    except:
                        continue
            
            if len(prices) > 1:
                cv = np.std(prices) / np.mean(prices)  # Coefficient of variation
                consistency_factor = max(0, 1 - cv)  # Lower variation = higher confidence
                factors.append(consistency_factor)
        
        # Calculate weighted average
        return np.mean(factors) if factors else 0.5
    
    def enhanced_query(self, 
                      query: str, 
                      filters: Optional[Dict] = None,
                      n_results: int = 15) -> Dict[str, Any]:
        """Enhanced query processing with structured responses"""
        start_time = datetime.now()
        
        # Check cache first
        cache_key = hashlib.md5(f"{query}{filters}".encode()).hexdigest()
        if cache_key in self.response_cache:
            self.query_stats['cache_hits'] += 1
            cached_result = self.response_cache[cache_key].copy()
            cached_result['cached'] = True
            return cached_result
        
        try:
            # Classify query
            query_type = self.query_classifier.classify_query(query)
            
            # Enhanced retrieval
            context_docs = self.enhanced_retrieval(query, n_results, filters)
            
            # Generate structured response
            structured_response = self.generate_structured_response(query, context_docs, query_type)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'query': query,
                'query_type': query_type.value,
                'structured_response': structured_response,
                'context_docs': context_docs,
                'response_time': response_time,
                'num_context_docs': len(context_docs),
                'cached': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result (limit cache size)
            if len(self.response_cache) < 100:
                self.response_cache[cache_key] = result.copy()
            
            # Update statistics
            self.query_stats['total_queries'] += 1
            self.query_stats['successful_queries'] += 1
            self.query_stats['avg_response_time'] = (
                (self.query_stats['avg_response_time'] * (self.query_stats['total_queries'] - 1) + response_time) /
                self.query_stats['total_queries']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {str(e)}")
            
            # Update error statistics
            self.query_stats['total_queries'] += 1
            
            return {
                'query': query,
                'query_type': 'error',
                'structured_response': PricingResponse(
                    response_text=f"I encountered an error processing your query: {str(e)}",
                    confidence_score=0.0
                ),
                'context_docs': [],
                'response_time': (datetime.now() - start_time).total_seconds(),
                'num_context_docs': 0,
                'cached': False,
                'error': str(e)
            }
    
    async def async_batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Async batch processing for multiple queries"""
        tasks = []
        
        for query in queries:
            task = asyncio.create_task(self._async_single_query(query))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'query': queries[i],
                    'error': str(result),
                    'query_type': 'error'
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _async_single_query(self, query: str) -> Dict[str, Any]:
        """Single async query processing"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.enhanced_query, query)
        return result
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        chroma_stats = self.chroma_manager.get_collection_stats()
        
        success_rate = (
            self.query_stats['successful_queries'] / max(1, self.query_stats['total_queries'])
        ) * 100
        
        cache_hit_rate = (
            self.query_stats['cache_hits'] / max(1, self.query_stats['total_queries'])
        ) * 100
        
        return {
            'query_stats': {
                **self.query_stats,
                'success_rate': success_rate,
                'cache_hit_rate': cache_hit_rate
            },
            'database_stats': chroma_stats,
            'model_info': {
                'current_model': self.model_name,
                'model_hierarchy': self.model_hierarchy,
                'current_model_index': self.current_model_index
            },
            'configuration': {
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'context_window': self.context_window,
                'use_api': self.use_api
            },
            'cache_size': len(self.response_cache)
        }
    
    def export_query_results(self, results: List[Dict], output_path: str):
        """Export query results with enhanced formatting"""
        # Prepare data for export
        export_data = []
        
        for result in results:
            structured_resp = result.get('structured_response')
            if structured_resp:
                export_item = {
                    'query': result['query'],
                    'query_type': result['query_type'],
                    'response_text': structured_resp.response_text,
                    'primary_price': structured_resp.primary_price,
                    'price_range': structured_resp.price_range,
                    'confidence_score': structured_resp.confidence_score,
                    'data_source_count': structured_resp.data_source_count,
                    'factors_affecting_price': structured_resp.factors_affecting_price,
                    'recommendations': structured_resp.recommendations,
                    'comparable_contracts': structured_resp.comparable_contracts,
                    'response_time': result['response_time'],
                    'timestamp': result.get('timestamp')
                }
            else:
                export_item = result
            
            export_data.append(export_item)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Enhanced query results exported to {output_path}")

def main():
    """Test the enhanced RAG pipeline"""
    # Initialize enhanced pipeline with Mistral API
    pipeline = EnhancedTenderRAGPipeline(
        model_name="mistral-small-latest",  # Mistral AI model
        context_window=4000,  # Enhanced context
        use_api=True,
        use_mistral_api=True  # Use Mistral API instead of HuggingFace
    )
    
    # Setup enhanced database
    print("Setting up enhanced database...")
    stats = pipeline.setup_enhanced_database(
        "./data/healthcare_contracts_cleaned_20250717_230006.json",
        reset=True
    )
    
    print(f"Database setup complete: {stats}")
    
    # Test queries with structured responses
    test_queries = [
        "What is the average price for medical equipment in Bogotá?",
        "Compare nursing service costs between different regions in Colombia",
        "Estimate budget for pharmaceutical services for a medium-sized hospital",
        "Show me the most expensive healthcare contracts over $200,000",
        "What factors affect pricing for healthcare consulting services?"
    ]
    
    print("\nTesting Enhanced RAG Pipeline...")
    print("=" * 60)
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 50)
        
        result = pipeline.enhanced_query(query)
        results.append(result)
        
        structured_resp = result['structured_response']
        
        print(f"Query Type: {result['query_type']}")
        print(f"Response Time: {result['response_time']:.2f}s")
        print(f"Confidence Score: {structured_resp.confidence_score:.2f}")
        print(f"Data Sources: {structured_resp.data_source_count}")
        
        if structured_resp.primary_price:
            print(f"Primary Price: ${structured_resp.primary_price:,.2f}")
        
        if structured_resp.price_range:
            print(f"Price Range: ${structured_resp.price_range[0]:,.2f} - ${structured_resp.price_range[1]:,.2f}")
        
        print(f"\nResponse:\n{structured_resp.response_text}")
        
        if structured_resp.factors_affecting_price:
            print(f"\nKey Factors:")
            for factor in structured_resp.factors_affecting_price:
                print(f"  - {factor}")
        
        if structured_resp.recommendations:
            print(f"\nRecommendations:")
            for rec in structured_resp.recommendations:
                print(f"  - {rec}")
        
        print("=" * 60)
    
    # Show enhanced statistics
    stats = pipeline.get_enhanced_stats()
    print(f"\nEnhanced Pipeline Statistics:")
    print(f"Success Rate: {stats['query_stats']['success_rate']:.1f}%")
    print(f"Cache Hit Rate: {stats['query_stats']['cache_hit_rate']:.1f}%")
    print(f"Average Response Time: {stats['query_stats']['avg_response_time']:.2f}s")
    print(f"Model Fallbacks: {stats['query_stats']['model_fallbacks']}")
    print(f"Current Model: {stats['model_info']['current_model']}")
    
    # Export results
    pipeline.export_query_results(results, "./enhanced_query_results.json")
    print(f"\nResults exported to enhanced_query_results.json")

if __name__ == "__main__":
    main()
