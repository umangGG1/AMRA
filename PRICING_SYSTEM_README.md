# Healthcare Contract Pricing Recommendation System

Advanced vector database system using BGE-base-en-v1.5 embeddings for accurate healthcare contract pricing recommendations.

## Overview

This system analyzes 539 healthcare contracts to provide pricing recommendations using:
- **BGE-base-en-v1.5 embeddings** for semantic understanding
- **Multi-collection ChromaDB architecture** for specialized search
- **Advanced data preprocessing** to reduce noise and enhance accuracy
- **Confidence scoring** for recommendation reliability

## System Architecture

### 1. Data Processing Pipeline (`healthcare_data_processor.py`)
- Filters extreme values and data quality issues
- Creates categorical features (price brackets, regions, service categories)
- Extracts service keywords and complexity scores
- Generates comprehensive data reports

### 2. Vector Database Manager (`chromadb_manager.py`)
- **BGE-base-en-v1.5 integration** with custom embedding function
- **4 specialized collections**:
  - `pricing_context`: Direct pricing recommendations
  - `service_similarity`: Service matching for comparisons
  - `geographic_pricing`: Location-based pricing patterns
  - `contractor_performance`: Contractor pricing history

### 3. Pricing Recommendation Engine (`pricing_recommendation_system.py`)
- Multi-collection weighted search
- Confidence assessment and pricing analysis
- Comparative analysis across scenarios
- Export functionality for recommendations

## Key Features

### Data Preprocessing
- **Value filtering**: Removes contracts <$1K or >$10B (data quality issues)
- **Service categorization**: medical_services, professional_services, equipment_supply, etc.
- **Geographic standardization**: Regional groupings for analysis
- **Complexity scoring**: Based on value, content length, and keyword diversity

### Vector Collections Schema
```python
pricing_context:
  - Content: "Healthcare service: [description] | Price: $[value] | Location: [dept] | Entity: [name]"
  - Metadata: price_bracket, entity_department, contract_type, execution_year
  
service_similarity:
  - Content: Focus on contract_object and searchable_content
  - Metadata: service_category, contract_modality, complexity_level
  
geographic_pricing:
  - Content: Location + service + pricing context
  - Metadata: region, municipality, price_bracket, service_category
  
contractor_performance:
  - Content: Contractor + service + performance context  
  - Metadata: contractor_name, performance_category, value_range
```

### BGE Optimization
- **Query instruction**: "Represent this sentence for searching relevant passages: "
- **FP16 precision** for faster inference
- **Batch processing** for efficiency
- **Custom embedding function** integrated with ChromaDB

## Usage

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Data Processing
```python
from src.healthcare_data_processor import HealthcareDataProcessor

processor = HealthcareDataProcessor()
df = processor.load_data("data/healthcare_contracts_cleaned_20250717_230006.json")
df_processed = processor.add_derived_features(processor.clean_and_filter(df))
```

### 3. Setup Pricing System
```python
from src.pricing_recommendation_system import PricingRecommendationSystem

pricing_system = PricingRecommendationSystem()
pricing_system.setup_system("data/healthcare_contracts_cleaned_20250717_230006.json")
```

### 4. Get Pricing Recommendations
```python
recommendations = pricing_system.get_pricing_recommendations(
    query_description="Professional healthcare support services financial advisory",
    target_department="Bogotá D.C.",
    service_category="professional_services",
    n_results=10
)

print(f"Recommended Price: ${recommendations['pricing_analysis']['recommended_price']:,.0f}")
print(f"Confidence: {recommendations['pricing_analysis']['confidence']}")
```

### 5. Run Tests
```bash
python test_pricing_system.py
```

## Output Structure

### Pricing Recommendation
```json
{
  "query": "Service description",
  "pricing_analysis": {
    "recommended_price": 45000000,
    "price_range": {"min": 30000000, "max": 60000000},
    "confidence": "high",
    "basis": "8_similar_contracts"
  },
  "confidence_assessment": {
    "overall_score": 0.85,
    "collection_coverage": 0.75,
    "average_similarity": 0.82,
    "data_completeness": 0.90
  },
  "similar_contracts": [...],
  "total_similar_contracts": 25,
  "total_pricing_references": 18
}
```

## Data Insights

From the 539 healthcare contracts:
- **Value range**: $1,000 - $1.69 trillion (extreme values filtered)
- **Service categories**: 6 main types (medical, professional, equipment, training, maintenance, research)
- **Geographic coverage**: 7 regions across Colombia
- **Entity types**: Ministry, institutes, funds, healthcare facilities, universities

## Noise Reduction Strategy

**Excluded fields** (noise reduction):
- Technical IDs: contract_id, process_id, entity_code, entity_nit
- Processing metadata: source, dataset, cleaned_at, collection_time
- Non-discriminative: Always same values or too specific

**Optimized fields** (signal enhancement):
- contract_object: Primary service description
- contract_value: Core pricing target
- entity_department: Geographic context
- entity_name: Organization context
- searchable_content: Pre-processed combined text
- keywords: Extracted semantic features

## Performance Optimizations

1. **BGE-specific optimizations**: Query instructions, FP16, batch processing
2. **Multi-collection strategy**: Specialized schemas for different use cases
3. **Metadata filtering**: Efficient pre-filtering before similarity search
4. **Weighted scoring**: Different collection importance for final recommendations
5. **Confidence assessment**: Multiple factors for recommendation reliability

## Files Structure

```
src/
├── healthcare_data_processor.py     # Data preprocessing pipeline
├── chromadb_manager.py             # BGE + ChromaDB integration  
├── pricing_recommendation_system.py # Main recommendation engine
└── chunking_system.py              # (existing) Text chunking utilities

test_pricing_system.py              # Complete system tests
requirements.txt                    # Updated with FlagEmbedding
```

This system provides accurate, confidence-scored pricing recommendations for healthcare contracts using state-of-the-art embeddings and multi-dimensional similarity search.