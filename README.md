# AMRA Healthcare POC: RAG-Powered Tender Pricing System

## 🎯 Project Overview

A production-ready healthcare contract pricing bot that leverages advanced RAG (Retrieval-Augmented Generation) technology to analyze 2,000 healthcare tender records and provide accurate pricing insights for client acquisition.

## 🏗️ Technical Architecture

### Core Components
- **ChromaDB**: Multi-collection vector database with advanced retrieval features
- **Mistral-3B-Instruct**: Efficient 3.8B parameter model optimized for tender pricing
- **Sentence-Transformers**: `all-mpnet-base-v2` for superior semantic embeddings
- **LangChain**: Advanced RAG orchestration with enhanced retrieval for smaller models

### System Design
```
[Tender Data] → [Preprocessing] → [ChromaDB Collections] → [RAG Pipeline] → [Mistral-3B] → [Response]
```

## 📊 Implementation Roadmap

### Phase 1: Data Architecture (Days 1-3)
- **Day 1**: Environment setup and data analysis
- **Day 2**: ChromaDB schema design and data preprocessing
- **Day 3**: Intelligent chunking and metadata extraction

### Phase 2: Enhanced RAG Pipeline (Days 4-7)
- **Day 4**: Multi-vector retrieval system implementation
- **Day 5**: Hybrid search with semantic + keyword matching
- **Day 6**: Mistral-3B integration with structured prompts
- **Day 7**: Re-ranking and confidence scoring

### Phase 3: Query Intelligence (Days 8-10)
- **Day 8**: Industry-standard query patterns implementation
- **Day 9**: Chain-of-thought reasoning for complex pricing
- **Day 10**: Response validation and consistency checks

### Phase 4: Production Deployment (Days 11-14)
- **Day 11**: Evaluation framework and metrics
- **Day 12**: Performance optimization and context compression
- **Day 13**: Streamlit interface development
- **Day 14**: Testing and client demo preparation

## 🎯 ChromaDB Multi-Collection Schema

### Collection Structure
1. **tender_documents**: Full tender documents with metadata
2. **price_components**: Granular pricing breakdowns
3. **market_segments**: Categorized by industry/region
4. **historical_trends**: Time-series pricing data

### Advanced ChromaDB Features
- **Metadata Filtering**: Filter by tender_type, region, date_range, budget_range
- **Hybrid Search**: Combine semantic similarity with keyword matching
- **Re-ranking**: Built-in re-ranking for precision improvement
- **Custom Distance Functions**: Weighted similarity for pricing relevance

## 🔧 3B Model Optimization Strategies

### Enhanced Retrieval
- Top-10 ChromaDB results with sophisticated re-ranking
- Price-focused similarity functions in ChromaDB
- Aggressive metadata filtering for context relevance

### Structured Prompts
- Few-shot examples for tender pricing scenarios
- Step-by-step reasoning templates
- Chain-of-thought prompts for complex calculations

### Context Optimization
- Context prioritization for pricing information
- Context compression techniques
- Validation layer for numerical accuracy

## 📈 Industry-Standard Query Patterns

1. **Direct Price Lookup**: "What's the price for X service/product?"
2. **Comparative Analysis**: "Compare prices across similar tenders"
3. **Budget Estimation**: "Estimate total cost for project scope Y"
4. **Price Breakdown**: "Show cost components for tender Z"
5. **Market Analysis**: "Average pricing for category A in region B"
6. **Trend Analysis**: "Price trends for service X over time"
7. **Compliance Check**: "Pricing aligned with government regulations?"

## 🎯 Expected Performance Metrics

### Quantitative Metrics
- **Speed**: Sub-500ms response times for pricing queries
- **Resource Usage**: Minimal compute requirements, laptop-deployable
- **Accuracy**: 85-90% pricing accuracy with enhanced RAG compensation
- **Scalability**: Easy horizontal scaling and deployment flexibility

### Qualitative Assessment
- **Relevance Scoring**: 5-point scale for answer appropriateness
- **Completeness**: Coverage of pricing components
- **Clarity**: Understandability of explanations
- **Trustworthiness**: Confidence in pricing recommendations

## 🔍 Evaluation Framework

### A/B Testing Components
- Different retrieval strategies comparison
- Chunk size optimization (256/512/1024 tokens)
- Embedding model performance comparison
- Mistral prompt engineering variations

### Success Criteria
- 85% pricing accuracy on validation set
- Sub-500ms query response time
- 90% user satisfaction on relevance scoring
- Comprehensive coverage of tender pricing scenarios

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- 8GB+ RAM
- CUDA-compatible GPU (optional, for faster inference)

### Quick Setup (Recommended - with Virtual Environment)
```bash
# Clone and navigate to the project
cd AMRA-healthcare-POC

# Run the automated setup script with virtual environment
python setup_venv.py

# Activate virtual environment
# On Unix/Linux/macOS:
source activate.sh

# On Windows:
activate.bat
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Initialize ChromaDB
python test_chromadb.py

# Start the application
streamlit run app.py
```

### Usage
```bash
# Make sure virtual environment is activated first
source venv/bin/activate  # Unix/Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Start the web interface
streamlit run app.py

# Run complete demo
python demo.py

# Test the system directly
python -c "
from src.rag_pipeline import TenderRAGPipeline
pipeline = TenderRAGPipeline()
result = pipeline.query('What is the average price for healthcare services?')
print(result['response'])
"
```

## 📁 Project Structure
```
AMRA-healthcare-POC/
├── data/
│   ├── tender_data.csv
│   └── processed/
├── src/
│   ├── data_processing.py
│   ├── chromadb_manager.py
│   ├── rag_pipeline.py
│   ├── mistral_integration.py
│   └── query_processor.py
├── app.py
├── requirements.txt
└── README.md
```

## 🎯 Client Acquisition Strategy

### Demo Focus
- Showcase 5-7 successful queries with accurate pricing
- Highlight real-time responses and medical context understanding
- Demonstrate scalability potential for larger datasets

### Key Selling Points
- **Speed**: Real-time pricing responses
- **Accuracy**: Validated against tender records
- **Scalability**: Ready for enterprise deployment
- **Cost-Effective**: Efficient 3B model reduces operational costs

## 📞 Next Steps Post-POC

1. **Phase 2 Enhancement**: Upgrade to larger models for improved accuracy
2. **Enterprise Integration**: API development for system integration
3. **Advanced Features**: Multi-modal support, real-time data updates
4. **Compliance**: Full regulatory compliance implementation

---

**Ready to revolutionize healthcare tender pricing with advanced RAG technology!**