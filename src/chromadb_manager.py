"""
ChromaDB Manager for Healthcare Tender Pricing System
Manages multi-collection vector database with advanced retrieval features
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

try:
    from FlagEmbedding import FlagModel
    FLAGEMBEDDING_AVAILABLE = True
except ImportError:
    FLAGEMBEDDING_AVAILABLE = False
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGEEmbeddingFunction:
    """Custom embedding function for BGE-base-en-v1.5"""
    
    def __init__(self):
        """Initialize BGE model"""
        if FLAGEMBEDDING_AVAILABLE:
            logger.info("Loading BGE-base-en-v1.5 model with FlagEmbedding...")
            self.model = FlagModel(
                'BAAI/bge-base-en-v1.5',
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                use_fp16=True
            )
        else:
            logger.info("Loading BGE-base-en-v1.5 model with SentenceTransformers...")
            self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        logger.info("BGE model loaded successfully!")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts"""
        embeddings = self.model.encode(input)
        return embeddings.tolist()


class HealthcareChromaManager:
    """Manages ChromaDB collections for healthcare contract pricing RAG system"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with BGE embeddings"""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Initialize BGE embedding function
        self.embedding_function = BGEEmbeddingFunction()
        
        # Collection references
        self.collections = {}
        self.collection_schemas = self._define_collection_schemas()
        
        logger.info(f"ChromaDB initialized with persistence at {self.persist_directory}")
    
    def _define_collection_schemas(self) -> Dict[str, Dict]:
        """Define optimized collection schemas for healthcare contract pricing"""
        return {
            "pricing_context": {
                "description": "Direct pricing recommendations based on service similarity",
                "fields": [
                    "contract_object", "contract_value", "entity_name", 
                    "entity_department", "contract_type", "process_status"
                ],
                "metadata_fields": [
                    "contract_value", "price_bracket", "entity_department", "contract_type", 
                    "execution_year", "service_category", "contract_modality"
                ],
                "embedding_strategy": "pricing_focused"
            },
            "service_similarity": {
                "description": "Service similarity for comparative analysis",
                "fields": [
                    "contract_object", "process_object", "searchable_content"
                ],
                "metadata_fields": [
                    "contract_value", "service_category", "contract_modality", "complexity_level",
                    "entity_department", "contract_type", "price_bracket"
                ],
                "embedding_strategy": "service_focused"
            },
            "geographic_pricing": {
                "description": "Location-based pricing patterns",
                "fields": [
                    "entity_department", "entity_municipality", "contract_object",
                    "contract_value", "entity_name"
                ],
                "metadata_fields": [
                    "contract_value", "region", "entity_municipality", "price_bracket", 
                    "service_category", "entity_type", "entity_department"
                ],
                "embedding_strategy": "geographic_focused"
            },
            "contractor_performance": {
                "description": "Contractor pricing history and performance",
                "fields": [
                    "contractor_name", "contract_object", "contract_value",
                    "process_status", "entity_department"
                ],
                "metadata_fields": [
                    "contract_value", "contractor_name", "performance_category", "value_range",
                    "entity_department", "success_rate", "price_bracket"
                ],
                "embedding_strategy": "contractor_focused"
            }
        }
    
    def create_collections(self):
        """Create all collections with proper configuration"""
        for collection_name, schema in self.collection_schemas.items():
            try:
                # Create collection with embedding function
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": schema["description"]}
                )
                
                self.collections[collection_name] = collection
                logger.info(f"Created collection: {collection_name}")
                
            except Exception as e:
                # Collection might already exist
                try:
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    self.collections[collection_name] = collection
                    logger.info(f"Loaded existing collection: {collection_name}")
                except Exception as e2:
                    logger.error(f"Error with collection {collection_name}: {str(e2)}")
                    raise
    
    def load_healthcare_data(self, data_path: str) -> pd.DataFrame:
        """Load healthcare contract data from JSON file"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} healthcare contract records from {data_path}")
        return df
    
    def preprocess_healthcare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess healthcare contract data for ChromaDB ingestion"""
        logger.info(f"Starting preprocessing of {len(df)} records")
        logger.info(f"Original columns: {list(df.columns)}")
        
        # Map new data structure to expected structure
        df_processed = df.copy()
        
        # The data already has correct field names, no mapping needed
        # Just ensure we have the contract_number field
        if 'contract_number' not in df_processed.columns:
            if 'contract_id' in df_processed.columns:
                df_processed['contract_number'] = df_processed['contract_id']
            else:
                df_processed['contract_number'] = df_processed.index.astype(str)
        
        # Add required fields that might be missing
        required_fields = [
            'entity_department', 'entity_municipality', 'contract_modality',
            'end_date', 'execution_end_date', 'process_object'
        ]
        
        for field in required_fields:
            if field not in df_processed.columns:
                logger.info(f"Adding default value for missing field: {field}")
                if field in ['end_date', 'execution_end_date']:
                    df_processed[field] = pd.Timestamp.now()
                else:
                    df_processed[field] = 'Not specified'
        
        # Extract department from entity name if available
        if 'entity_name' in df_processed.columns:
            df_processed['entity_department'] = df_processed['entity_name'].apply(self._extract_department)
        
        # Extract municipality (simplified)
        df_processed['entity_municipality'] = 'Not defined'
        
        # Set contract modality based on type
        if 'contract_type' in df_processed.columns:
            df_processed['contract_modality'] = df_processed['contract_type'].apply(self._map_contract_modality)
        
        # Set process object same as contract object initially
        df_processed['process_object'] = df_processed['contract_object']
        
        # Convert contract_value to numeric first
        df_processed['contract_value'] = pd.to_numeric(df_processed['contract_value'], errors='coerce')
        
        # Filter extreme values (likely data quality issues) - now that contract_value is numeric
        df_processed = df_processed[(df_processed['contract_value'] >= 1000) & (df_processed['contract_value'] <= 10000000000)]
        
        # Parse end_date
        df_processed['end_date'] = pd.to_datetime(df_processed['end_date'], errors='coerce')
        df_processed['execution_year'] = df_processed['end_date'].dt.year
        df_processed['execution_quarter'] = df_processed['end_date'].dt.quarter
        
        # Create price brackets for metadata filtering
        df_processed['price_bracket'] = pd.cut(
            df_processed['contract_value'],
            bins=[0, 50000, 200000, 1000000, 5000000, float('inf')],
            labels=['0-50K', '50K-200K', '200K-1M', '1M-5M', '5M+']
        )
        
        # Create regional categories
        df_processed['region'] = df_processed['entity_department'].apply(self._categorize_region)
        
        # Extract service categories from contract objects
        df_processed['service_category'] = df_processed['contract_object'].apply(self._categorize_service)
        
        # Categorize entity types
        df_processed['entity_type'] = df_processed['entity_name'].apply(self._categorize_entity)
        
        # Create performance categories
        df_processed['performance_category'] = df_processed.apply(self._categorize_performance, axis=1)
        
        # Calculate complexity level based on content length and value
        df_processed['complexity_level'] = df_processed.apply(self._calculate_complexity, axis=1)
        
        # Calculate success rate (simplified based on status)
        df_processed['success_rate'] = df_processed['process_status'].apply(self._calculate_success_rate)
        
        # Create value ranges for contractor analysis
        df_processed['value_range'] = df_processed['contract_value'].apply(self._categorize_value_range)
        
        # Remove rows with critical missing data
        df_processed = df_processed.dropna(subset=['contract_object', 'contract_value'])
        
        logger.info(f"Preprocessed data: {len(df_processed)} records ready for ingestion")
        logger.info(f"Final columns: {list(df_processed.columns)}")
        return df_processed
    
    def _extract_department(self, entity_name: str) -> str:
        """Extract department from entity name"""
        if pd.isna(entity_name) or entity_name == 'Not specified':
            return 'Unknown'
        
        # Common department patterns
        dept_patterns = {
            'antioquia': 'Antioquia',
            'bogotá': 'Bogotá D.C.',
            'valle': 'Valle del Cauca',
            'atlántico': 'Atlántico',
            'santander': 'Santander',
            'boyacá': 'Boyacá',
            'cundinamarca': 'Cundinamarca',
            'tolima': 'Tolima',
            'huila': 'Huila',
            'cauca': 'Cauca',
            'nariño': 'Nariño',
            'magdalena': 'Magdalena',
            'córdoba': 'Córdoba',
            'risaralda': 'Risaralda',
            'caldas': 'Caldas'
        }
        
        entity_lower = entity_name.lower()
        for pattern, dept in dept_patterns.items():
            if pattern in entity_lower:
                return dept
        
        return 'Other'
    
    def _map_contract_modality(self, contract_type: str) -> str:
        """Map contract type to modality"""
        if pd.isna(contract_type) or contract_type == 'Not specified':
            return 'Direct Contracting'
        
        type_lower = contract_type.lower()
        
        if any(word in type_lower for word in ['direct', 'directa']):
            return 'Direct Contracting'
        elif any(word in type_lower for word in ['public', 'pública', 'tender', 'licitación']):
            return 'Public Tender'
        elif any(word in type_lower for word in ['framework', 'marco']):
            return 'Framework Agreement'
        else:
            return 'Other'
    
    def _categorize_region(self, department: str) -> str:
        """Categorize regions for metadata filtering"""
        if pd.isna(department):
            return 'unknown'
        
        dept_lower = department.lower()
        if 'bogotá' in dept_lower or 'distrito' in dept_lower or 'capital' in dept_lower:
            return 'capital'
        elif any(word in dept_lower for word in ['valle', 'cauca', 'nariño']):
            return 'pacific'
        elif any(word in dept_lower for word in ['antioquia', 'caldas', 'risaralda']):
            return 'central'
        elif any(word in dept_lower for word in ['atlántico', 'magdalena', 'córdoba']):
            return 'atlantic'
        elif any(word in dept_lower for word in ['santander', 'boyacá', 'cundinamarca']):
            return 'central_east'
        elif any(word in dept_lower for word in ['tolima', 'huila', 'caquetá']):
            return 'central_south'
        else:
            return 'other'
    
    def _categorize_service(self, contract_object: str) -> str:
        """Extract service category from contract description"""
        if pd.isna(contract_object):
            return 'unknown'
        
        text_lower = contract_object.lower()
        if any(word in text_lower for word in ['medical', 'health', 'clinical', 'hospital', 'patient']):
            return 'medical_services'
        elif any(word in text_lower for word in ['professional', 'technical', 'support', 'advisory']):
            return 'professional_services'
        elif any(word in text_lower for word in ['equipment', 'supply', 'material', 'device']):
            return 'equipment_supply'
        elif any(word in text_lower for word in ['training', 'education', 'workshop', 'capacity']):
            return 'training_education'
        elif any(word in text_lower for word in ['maintenance', 'repair', 'installation']):
            return 'maintenance_technical'
        elif any(word in text_lower for word in ['research', 'study', 'analysis', 'evaluation']):
            return 'research_analysis'
        else:
            return 'general_services'
    
    def _categorize_entity(self, entity_name: str) -> str:
        """Categorize entity type from entity name"""
        if pd.isna(entity_name):
            return 'unknown'
        
        entity_lower = entity_name.lower()
        if any(word in entity_lower for word in ['ministry', 'ministerio']):
            return 'ministry'
        elif any(word in entity_lower for word in ['institute', 'instituto']):
            return 'institute'
        elif any(word in entity_lower for word in ['fund', 'fondo']):
            return 'fund'
        elif any(word in entity_lower for word in ['hospital', 'clinic']):
            return 'healthcare_facility'
        elif any(word in entity_lower for word in ['university', 'universidad']):
            return 'university'
        else:
            return 'other_entity'
    
    def _calculate_complexity(self, row) -> str:
        """Calculate complexity level based on value and content"""
        value = row['contract_value']
        content_length = row.get('content_length', 0)
        
        if value > 5000000 or content_length > 500:
            return 'high'
        elif value > 1000000 or content_length > 300:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_success_rate(self, status: str) -> str:
        """Calculate success rate based on process status"""
        if pd.isna(status):
            return 'unknown'
        
        status_lower = status.lower()
        if any(word in status_lower for word in ['completed', 'finalized', 'sent']):
            return 'high'
        elif 'draft' in status_lower:
            return 'medium'
        elif any(word in status_lower for word in ['cancelled', 'terminated']):
            return 'low'
        else:
            return 'medium'
    
    def _categorize_value_range(self, value: float) -> str:
        """Categorize contract value into ranges"""
        if pd.isna(value):
            return 'unknown'
        
        if value < 50000:
            return 'micro'
        elif value < 200000:
            return 'small'
        elif value < 1000000:
            return 'medium'
        elif value < 5000000:
            return 'large'
        else:
            return 'mega'
    
    def _categorize_performance(self, row) -> str:
        """Categorize contractor performance"""
        if pd.isna(row['contract_value']) or pd.isna(row['process_status']):
            return 'unknown'
        
        value = row['contract_value']
        status = row['process_status'].lower()
        
        if any(word in status for word in ['cancelled', 'terminated']):
            return 'cancelled'
        elif 'draft' in status:
            return 'pending'
        elif any(word in status for word in ['completed', 'finalized', 'sent']):
            if value > 1000000:
                return 'high_value_completed'
            elif value > 200000:
                return 'medium_value_completed'
            else:
                return 'low_value_completed'
        else:
            return 'in_progress'
    
    def populate_collections(self, df: pd.DataFrame):
        """Populate all collections with processed data"""
        logger.info(f"Starting to populate {len(self.collection_schemas)} collections with {len(df)} records")
        
        # Log sample data for debugging
        if not df.empty:
            sample_row = df.iloc[0]
            logger.info(f"Sample contract_value: {sample_row.get('contract_value', 'MISSING')}")
            logger.info(f"Sample columns: {list(df.columns)[:10]}...")
        
        for collection_name, schema in self.collection_schemas.items():
            logger.info(f"Populating collection: {collection_name}")
            logger.info(f"Schema metadata fields: {schema['metadata_fields']}")
            self._populate_single_collection(collection_name, schema, df)
            
        logger.info("All collections populated successfully")
    
    def _populate_single_collection(self, collection_name: str, schema: Dict, df: pd.DataFrame):
        """Populate a single collection with appropriate data"""
        collection = self.collections[collection_name]
        
        documents = []
        metadatas = []
        ids = []
        pricing_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Generate document text based on embedding strategy
                doc_text = self._generate_document_text(row, schema)
                
                # Generate metadata
                metadata = self._generate_metadata(row, schema)
                
                # Count documents with pricing information
                if 'contract_value' in metadata and metadata['contract_value'] > 0:
                    pricing_count += 1
                
                # Generate unique ID
                doc_id = f"{collection_name}_{idx}"
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(doc_id)
                
            except Exception as e:
                logger.warning(f"Error processing row {idx} for {collection_name}: {e}")
                continue
        
        # Add to collection in batches
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            try:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                total_added += len(batch_docs)
            except Exception as e:
                logger.error(f"Error adding batch to {collection_name}: {e}")
                # Try adding documents one by one to identify the problematic ones
                for j, (doc, meta, doc_id) in enumerate(zip(batch_docs, batch_metadata, batch_ids)):
                    try:
                        collection.add(
                            documents=[doc],
                            metadatas=[meta],
                            ids=[doc_id]
                        )
                        total_added += 1
                    except Exception as e2:
                        logger.warning(f"Failed to add document {doc_id}: {e2}")
                        logger.warning(f"Metadata: {meta}")
        
        logger.info(f"Added {total_added} documents to {collection_name} ({pricing_count} with pricing data)")
        
        # Log sample metadata for debugging
        if metadatas:
            sample_metadata = metadatas[0]
            logger.info(f"Sample metadata for {collection_name}: {sample_metadata}")
    
    def _generate_document_text(self, row: pd.Series, schema: Dict) -> str:
        """Generate document text based on embedding strategy"""
        strategy = schema.get("embedding_strategy", "pricing_focused")
        
        if strategy == "pricing_focused":
            return self._generate_pricing_context(row)
        elif strategy == "service_focused":
            return self._generate_service_context(row)
        elif strategy == "geographic_focused":
            return self._generate_geographic_context(row)
        elif strategy == "contractor_focused":
            return self._generate_contractor_context(row)
        else:
            return self._generate_pricing_context(row)
    
    def _generate_pricing_context(self, row: pd.Series) -> str:
        """Generate pricing-focused context optimized for BGE embeddings"""
        parts = []
        
        # Service description (primary content)
        if pd.notna(row['contract_object']):
            parts.append(f"Healthcare service: {row['contract_object']}")
        
        # Price information - ensure it's always included if available
        if pd.notna(row['contract_value']):
            try:
                value = float(row['contract_value'])
                if value > 0:
                    parts.append(f"Price: ${value:,.0f}")
            except (ValueError, TypeError):
                pass
        
        # Entity and location
        if pd.notna(row['entity_name']):
            parts.append(f"Entity: {row['entity_name']}")
        
        if pd.notna(row['entity_department']):
            parts.append(f"Location: {row['entity_department']}")
        
        # Contract details
        if pd.notna(row['contract_type']):
            parts.append(f"Type: {row['contract_type']}")
        
        if pd.notna(row['process_status']):
            parts.append(f"Status: {row['process_status']}")
        
        # Temporal context
        if pd.notna(row.get('execution_year')):
            try:
                year = int(row['execution_year'])
                parts.append(f"Year: {year}")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(parts) if parts else "Healthcare service: No description available"
    
    def _generate_service_context(self, row: pd.Series) -> str:
        """Generate service-focused context for similarity matching"""
        parts = []
        
        # Primary service description
        if pd.notna(row['contract_object']):
            parts.append(row['contract_object'])
        
        # Additional context from process object if different
        if pd.notna(row['process_object']) and row['process_object'] != row['contract_object']:
            parts.append(row['process_object'])
        
        # Use searchable content if available
        if pd.notna(row.get('searchable_content')):
            # Extract the service part from searchable content
            content = row['searchable_content']
            if len(content) > len(row['contract_object']) * 1.5:
                parts.append(content[:500])  # Limit to first 500 chars
        
        return " ".join(parts)
    
    
    def _generate_geographic_context(self, row: pd.Series) -> str:
        """Generate geographic-focused context"""
        parts = []
        
        # Location hierarchy
        if pd.notna(row['entity_department']):
            parts.append(f"Department: {row['entity_department']}")
        
        if pd.notna(row['entity_municipality']) and row['entity_municipality'] != 'Not Defined':
            parts.append(f"Municipality: {row['entity_municipality']}")
        
        # Entity context
        if pd.notna(row['entity_name']):
            parts.append(f"Entity: {row['entity_name']}")
        
        # Service and pricing in geographic context
        if pd.notna(row['contract_object']):
            # Truncate long descriptions for geographic focus
            service_desc = row['contract_object'][:200] + "..." if len(row['contract_object']) > 200 else row['contract_object']
            parts.append(f"Service: {service_desc}")
        
        if pd.notna(row['contract_value']):
            parts.append(f"Value: ${row['contract_value']:,.0f}")
        
        return " | ".join(parts)
    
    def _generate_contractor_context(self, row: pd.Series) -> str:
        """Generate contractor-focused context"""
        parts = []
        
        # Contractor identification
        if pd.notna(row['contractor_name']):
            parts.append(f"Contractor: {row['contractor_name']}")
        
        # Service provided
        if pd.notna(row['contract_object']):
            # Truncate for contractor focus
            service_desc = row['contract_object'][:300] + "..." if len(row['contract_object']) > 300 else row['contract_object']
            parts.append(f"Service: {service_desc}")
        
        # Financial and performance context
        if pd.notna(row['contract_value']):
            parts.append(f"Value: ${row['contract_value']:,.0f}")
        
        if pd.notna(row['process_status']):
            parts.append(f"Status: {row['process_status']}")
        
        # Geographic context for contractor
        if pd.notna(row['entity_department']):
            parts.append(f"Department: {row['entity_department']}")
        
        return " | ".join(parts)
    
    def _generate_metadata(self, row: pd.Series, schema: Dict) -> Dict[str, Any]:
        """Generate metadata for filtering"""
        metadata = {}
        
        for field in schema["metadata_fields"]:
            if field in row.index and pd.notna(row[field]):
                try:
                    if field == 'contract_value':
                        # Ensure contract_value is properly converted to float
                        value = float(row[field])
                        if value > 0:  # Only include positive values
                            metadata[field] = value
                    elif field == 'execution_year':
                        metadata[field] = int(row[field]) if pd.notna(row[field]) else None
                    elif field in ['price_bracket', 'service_category', 'region', 'entity_type', 
                                 'complexity_level', 'performance_category', 'value_range', 'success_rate']:
                        # Ensure categorical fields are strings
                        metadata[field] = str(row[field])
                    else:
                        # Handle other string fields
                        metadata[field] = str(row[field])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing metadata field {field}: {e}")
                    continue
        
        # Ensure we always have contract_value if it exists in the row
        if 'contract_value' in row.index and pd.notna(row['contract_value']):
            try:
                value = float(row['contract_value'])
                if value > 0:
                    metadata['contract_value'] = value
            except (ValueError, TypeError):
                pass
        
        return metadata
    
    def query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        where_filter: Optional[Dict] = None
    ) -> Dict:
        """Query a specific collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def multi_collection_search(
        self,
        query_text: str,
        collections: Optional[List[str]] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Search across multiple collections"""
        if collections is None:
            collections = list(self.collections.keys())
        
        results = {}
        
        for collection_name in collections:
            if collection_name in self.collections:
                collection_results = self.query_collection(
                    collection_name, query_text, n_results
                )
                results[collection_name] = collection_results
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        for collection_name, collection in self.collections.items():
            count = collection.count()
            stats[collection_name] = {
                'document_count': count,
                'description': self.collection_schemas[collection_name]['description']
            }
        
        return stats
    
    def reset_collections(self):
        """Reset all collections (for development/testing)"""
        logger.info("Resetting all collections...")
        
        for collection_name in list(self.collections.keys()):
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete collection {collection_name}: {str(e)}")
        
        self.collections = {}
        
        # Recreate collections
        self.create_collections()
        logger.info("Collections reset and recreated successfully")
    
    def verify_collections_have_pricing_data(self) -> Dict[str, int]:
        """Verify that collections contain pricing metadata"""
        results = {}
        
        for collection_name, collection in self.collections.items():
            # Query a few documents to check for pricing metadata
            try:
                sample_results = collection.query(
                    query_texts=["healthcare service"],
                    n_results=5
                )
                
                pricing_count = 0
                if sample_results['metadatas'] and sample_results['metadatas'][0]:
                    for metadata in sample_results['metadatas'][0]:
                        if 'contract_value' in metadata and metadata['contract_value'] is not None:
                            pricing_count += 1
                
                results[collection_name] = pricing_count
                logger.info(f"Collection {collection_name}: {pricing_count}/{len(sample_results['metadatas'][0])} samples have pricing data")
                
            except Exception as e:
                logger.error(f"Error verifying {collection_name}: {e}")
                results[collection_name] = -1
        
        return results


def main():
    """Initialize and populate ChromaDB with healthcare contract data"""
    # Initialize ChromaDB manager
    chroma_manager = HealthcareChromaManager()
    
    # Create collections
    chroma_manager.create_collections()
    
    # Load and preprocess data
    df = chroma_manager.load_healthcare_data(
        "/home/umanggod/AMRA-healthcare-POC/data/healthcare_contracts_cleaned_20250717_230006.json"
    )
    df_processed = chroma_manager.preprocess_healthcare_data(df)
    
    # Populate collections
    chroma_manager.populate_collections(df_processed)
    
    # Show statistics
    stats = chroma_manager.get_collection_stats()
    print("\nCollection Statistics:")
    for collection_name, info in stats.items():
        print(f"  {collection_name}: {info['document_count']} documents")
    
    # Test pricing query
    print("\nTesting pricing query...")
    results = chroma_manager.query_collection(
        "pricing_context",
        "professional healthcare support services financial advisory",
        n_results=3
    )
    
    print(f"Sample pricing results: {len(results['documents'][0])} documents found")
    for i, doc in enumerate(results['documents'][0][:2]):
        print(f"  {i+1}. {doc[:150]}...")
    
    # Test service similarity query
    print("\nTesting service similarity query...")
    results = chroma_manager.query_collection(
        "service_similarity",
        "technical support healthcare management",
        n_results=3
    )
    
    print(f"Sample service results: {len(results['documents'][0])} documents found")
    for i, doc in enumerate(results['documents'][0][:2]):
        print(f"  {i+1}. {doc[:150]}...")


if __name__ == "__main__":
    main()