# AMRA Healthcare RAG System - Complete Architecture Documentation

## Overview

This document provides a comprehensive overview of the Enhanced RAG (Retrieval-Augmented Generation) Pipeline implemented for the AMRA Healthcare Tender Pricing System. The system is designed to provide intelligent pricing recommendations for healthcare contracts using advanced vector search and LLM integration.



## Core Components

### 1. Enhanced RAG Pipeline (`rag_pipeline.py`)

The main RAG pipeline class `EnhancedTenderRAGPipeline` provides advanced query processing with structured responses.

#### Key Features Implemented:

##### Query Classification System
- **Enum**: `QueryType` with 6 categories:
  - `SPECIFIC_PRICING`: Direct price queries
  - `COMPARATIVE_ANALYSIS`: Price comparisons
  - `BUDGET_ESTIMATION`: Budget planning queries
  - `MARKET_RESEARCH`: Market analysis
  - `TREND_ANALYSIS`: Price trend analysis
  - `GENERAL_INQUIRY`: Generic questions

- **Classifier**: `EnhancedQueryClassifier` using regex patterns for query type detection

##### Structured Response System
- **Data Class**: `PricingResponse` with structured output:
  ```python
  @dataclass
  class PricingResponse:
      primary_price: Optional[float]
      price_range: Optional[Tuple[float, float]]
      comparable_contracts: List[Dict]
      factors_affecting_price: List[str]
      confidence_score: float
      data_source_count: int
      response_text: str
      recommendations: List[str]
  ```

##### Multi-Stage Retrieval System
1. **Stage 1**: Broad semantic search across collections
2. **Stage 2**: Metadata filtering
3. **Stage 3**: Reranking by combined relevance score

##### Data Validation System
- **Class**: `DataValidator` with comprehensive data cleaning:
  - Currency standardization
  - Outlier removal using IQR method
  - Service category normalization
  - Missing value handling
  - Computed field generation

##### Enhanced Query Templates
- Specialized prompts for each query type
- Structured output formatting
- Context-aware instructions

##### Model Fallback Hierarchy
- Primary: `mistralai/Mistral-7B-Instruct-v0.1`
- Fallback 1: `microsoft/DialoGPT-medium`
- Fallback 2: `gpt2`

#### Configuration Parameters:
- **Model**: Mistral-7B-Instruct-v0.1 (upgraded from 3B)
- **Max Tokens**: 1024 (increased from 512)
- **Context Window**: 4000 characters
- **Temperature**: 0.1
- **Timeout**: 60 seconds
- **Cache Size**: 100 queries

#### Performance Features:
- **Response Caching**: MD5-based query caching
- **Async Support**: Batch query processing
- **Statistics Tracking**: Success rates, response times, cache hits
- **Error Handling**: Comprehensive exception management

---

### 2. ChromaDB Manager (`chromadb_manager.py`)

Manages the multi-collection vector database using advanced BGE embeddings.

#### Key Features Implemented:

##### BGE Embedding Function
- **Model**: `BAAI/bge-base-en-v1.5`
- **Custom Implementation**: `BGEEmbeddingFunction` class
- **Features**: 
  - Query instruction optimization
  - FP16 precision
  - Batch processing

##### Collection Schema System
Four specialized collections with optimized schemas:

1. **pricing_context** (40% weight)
   - Direct pricing recommendations
   - Fields: contract_object, contract_value, entity_name, entity_department
   - Metadata: price_bracket, execution_year, service_category

2. **service_similarity** (30% weight)
   - Service-based comparisons
   - Fields: contract_object, process_object, searchable_content
   - Metadata: complexity_level, contract_modality

3. **geographic_pricing** (20% weight)
   - Location-based pricing patterns
   - Fields: entity_department, entity_municipality, contract_value
   - Metadata: region, entity_type

4. **contractor_performance** (10% weight)
   - Contractor-specific insights
   - Fields: contractor_name, contract_history, performance_metrics
   - Metadata: performance_rating, specialization

##### Database Operations
- **Persistence**: ChromaDB with file-based storage
- **Settings**: Anonymous telemetry disabled, reset enabled
- **Collection Management**: Automatic creation and population
- **Statistics**: Real-time document counts and metadata

---


## Data Pipeline Architecture

### Data Processing Flow

1. **Data Ingestion**
   - Source: Healthcare contract JSON files
   - Format: Structured contract records
   - Volume: ~1900+ contract records

2. **Data Validation** (`DataValidator`)
   - Currency standardization
   - Outlier detection and removal
   - Service category normalization
   - Missing value imputation
   - Quality score calculation

3. **Data Preprocessing**
   - Text cleaning and normalization
   - Metadata extraction
   - Content hash generation
   - Price bracket categorization

4. **Embedding Generation**
   - BGE-base-en-v1.5 model
   - 768-dimensional vectors
   - Query-optimized instructions
   - Batch processing

5. **Collection Population**
   - Multi-collection strategy
   - Specialized content per collection
   - Metadata indexing
   - Persistence management

### Database Schema

#### Collection Structure
```
chroma_db/
├── pricing_context/     # Direct pricing data
├── service_similarity/  # Service comparisons
├── geographic_pricing/  # Location-based data
└── contractor_performance/ # Contractor insights
```

#### Metadata Fields
- **contract_value**: Numeric pricing data
- **entity_department**: Geographic information
- **service_category**: Service classification
- **contract_type**: Contract categorization
- **price_bracket**: Price range classification
- **execution_year**: Temporal information

---

## Query Processing Architecture

### Query Flow

1. **Query Reception**
   - User input via Streamlit interface
   - Query preprocessing and validation

2. **Query Classification**
   - Pattern matching against query types
   - Confidence scoring for classification
   - Template selection

3. **Multi-Stage Retrieval**
   - **Stage 1**: Broad semantic search (2x results)
   - **Stage 2**: Metadata filtering
   - **Stage 3**: Relevance reranking (final N results)

4. **Context Preparation**
   - Enhanced formatting with metadata
   - Length constraint management (4000 chars)
   - Relevance score inclusion

5. **Response Generation**
   - LLM API call with structured prompts
   - Fallback model hierarchy
   - Timeout and error handling

6. **Response Parsing**
   - Structured data extraction
   - Price information parsing
   - Bullet point extraction
   - Confidence calculation

### Scoring System

#### Combined Relevance Score
```python
combined_score = (
    0.5 * distance_score +      # Semantic similarity
    0.3 * completeness_score +  # Metadata completeness
    0.2 * recency_score        # Contract recency
)
```

#### Confidence Score Factors
- Data source count (5+ sources = full confidence)
- Average relevance of top 3 documents
- Metadata completeness percentage
- Price consistency (coefficient of variation)

---

## Performance Features

### Caching System
- **Query Caching**: MD5-based cache keys
- **Cache Size**: 100 most recent queries
- **Cache Hits**: Tracked for performance metrics

### Statistics Tracking
- **Query Metrics**: Total, successful, failed queries
- **Performance Metrics**: Average response time, cache hit rate
- **Model Metrics**: Fallback usage, current model status

### Error Handling
- **API Failures**: Model fallback hierarchy
- **Database Errors**: Graceful degradation
- **Validation Errors**: User-friendly messages
- **Timeout Handling**: Configurable timeouts

### Async Support
- **Batch Processing**: Multiple queries simultaneously
- **Concurrent Execution**: Thread pool execution
- **Exception Handling**: Per-query error isolation

---

## API Integration

### Hugging Face Integration
- **Primary Model**: mistralai/Mistral-7B-Instruct-v0.1
- **Authentication**: Bearer token authentication
- **Parameters**:
  - max_new_tokens: 1024
  - temperature: 0.1
  - do_sample: True
  - return_full_text: False
  - stop: ["Question:", "Query:", "Context:"]

### Token Management
- **Environment Variable**: HUGGINGFACE_API_TOKEN
- **Validation**: Token format and authentication testing
- **Error Messages**: Clear guidance for token issues

---

## Configuration Management

### Environment Variables
- `HUGGINGFACE_API_TOKEN`: API authentication
- Database paths and model configurations via code

### Default Parameters
- **Retrieval**: 15 results per query (increased from 10)
- **Context Window**: 4000 characters
- **Temperature**: 0.1 for deterministic responses
- **Timeout**: 60 seconds for API calls
- **Cache Size**: 100 queries

---

## Export and Reporting

### Query Result Export
- **Format**: JSON with structured data
- **Fields**: Query metadata, structured responses, timing
- **Encoding**: UTF-8 with proper escaping

### Statistics Export
- **Pipeline Statistics**: Success rates, performance metrics
- **Database Statistics**: Collection counts, data quality scores
- **Model Information**: Current model, fallback usage

---

## Security Features

### Data Protection
- **API Token Security**: Environment variable storage
- **Error Message Sanitization**: No sensitive data exposure
- **Input Validation**: Query and parameter validation

### Error Handling
- **Graceful Degradation**: System continues with limited functionality
- **User Feedback**: Clear error messages and guidance
- **Logging**: Comprehensive logging without sensitive data

---

## Deployment Architecture

### File Structure
```
AMRA-healthcare-POC/
├── src/
│   ├── rag_pipeline.py          # Main RAG pipeline
│   ├── chromadb_manager.py      # Database management
│   ├── pricing_recommendation_system.py  # Pricing system
│   └── [other components]
├── app.py                       # Streamlit application
├── chroma_db/                   # Vector database storage
├── data/                        # Source data files
├── .env                         # Environment variables
└── venv/                        # Python virtual environment
```

### Dependencies
- **Core**: streamlit, chromadb, pandas, numpy
- **ML**: transformers, sentence-transformers, FlagEmbedding
- **API**: requests, langchain
- **Utilities**: python-dotenv, plotly

---



This documentation reflects the current state of the RAG system as implemented in the codebase. All features described are actively implemented and functional, providing a robust foundation for healthcare contract pricing analysis and recommendations.