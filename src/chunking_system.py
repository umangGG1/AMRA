"""
Intelligent Chunking System for Tender Documents
Optimized for pricing context preservation and semantic coherence
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TenderChunkingSystem:
    """Advanced chunking system optimized for tender pricing context"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 102):
        """
        Initialize chunking system
        
        Args:
            chunk_size: Target chunk size in tokens (default 512 for 3B model)
            chunk_overlap: Overlap between chunks (20% of chunk_size)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with custom separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "; ",    # Clause separators
                ", ",    # Comma separators
                " ",     # Word boundaries
                ""       # Character level
            ]
        )
        
        # Pricing keywords for context detection
        self.pricing_keywords = [
            'value', 'cost', 'price', 'amount', 'budget', 'payment',
            'valor', 'costo', 'precio', 'monto', 'presupuesto', 'pago',
            'contract_value', 'estimated_value', 'total_value'
        ]
        
        # Contract structure keywords
        self.structure_keywords = [
            'contract', 'service', 'delivery', 'requirements', 'specifications',
            'contrato', 'servicio', 'entrega', 'requisitos', 'especificaciones'
        ]
    
    def chunk_tender_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Chunk tender documents with pricing context preservation
        
        Args:
            df: DataFrame with tender data
            
        Returns:
            List of Document objects with chunks and metadata
        """
        documents = []
        
        for idx, row in df.iterrows():
            # Create chunks for this tender
            tender_chunks = self._chunk_single_tender(row, idx)
            documents.extend(tender_chunks)
        
        logger.info(f"Created {len(documents)} chunks from {len(df)} tender documents")
        return documents
    
    def _chunk_single_tender(self, row: pd.Series, tender_idx: int) -> List[Document]:
        """Process a single tender document into chunks"""
        chunks = []
        
        # Primary content: contract object
        primary_content = self._prepare_primary_content(row)
        
        # Secondary content: process details
        secondary_content = self._prepare_secondary_content(row)
        
        # Pricing context
        pricing_context = self._prepare_pricing_context(row)
        
        # Administrative context
        admin_context = self._prepare_administrative_context(row)
        
        # Combine all content
        full_content = self._combine_content_sections(
            primary_content, secondary_content, pricing_context, admin_context
        )
        
        # Create chunks from combined content
        text_chunks = self.text_splitter.split_text(full_content)
        
        # Convert to Document objects with metadata
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_metadata = self._create_chunk_metadata(row, tender_idx, chunk_idx, chunk_text)
            
            document = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            chunks.append(document)
        
        return chunks
    
    def _prepare_primary_content(self, row: pd.Series) -> str:
        """Prepare primary contract content"""
        sections = []
        
        # Contract object (main service description)
        if pd.notna(row['contract_object']):
            sections.append(f"Contract Object: {row['contract_object']}")
        
        # Process object if different
        if pd.notna(row['process_object']) and row['process_object'] != row['contract_object']:
            sections.append(f"Process Description: {row['process_object']}")
        
        return "\n\n".join(sections)
    
    def _prepare_secondary_content(self, row: pd.Series) -> str:
        """Prepare secondary contract details"""
        sections = []
        
        # Contract type and modality
        if pd.notna(row['contract_type']):
            sections.append(f"Contract Type: {row['contract_type']}")
        
        if pd.notna(row['contract_modality']):
            sections.append(f"Contract Modality: {row['contract_modality']}")
        
        # Status
        if pd.notna(row['process_status']):
            sections.append(f"Status: {row['process_status']}")
        
        return "\n".join(sections)
    
    def _prepare_pricing_context(self, row: pd.Series) -> str:
        """Prepare pricing-specific context"""
        sections = []
        
        # Contract value
        if pd.notna(row['contract_value']):
            try:
                value = float(row['contract_value'])
                sections.append(f"Contract Value: ${value:,.2f}")
            except (ValueError, TypeError):
                sections.append(f"Contract Value: {row['contract_value']}")
        
        # Execution timeline
        if pd.notna(row['execution_end_date']):
            sections.append(f"Execution End Date: {row['execution_end_date']}")
        
        return "\n".join(sections)
    
    def _prepare_administrative_context(self, row: pd.Series) -> str:
        """Prepare administrative context"""
        sections = []
        
        # Geographic location
        if pd.notna(row['entity_department']):
            sections.append(f"Department: {row['entity_department']}")
        
        if pd.notna(row['entity_municipality']) and row['entity_municipality'] != 'Not defined':
            sections.append(f"Municipality: {row['entity_municipality']}")
        
        # Contractor
        if pd.notna(row['contractor_name']):
            sections.append(f"Contractor: {row['contractor_name']}")
        
        # Contract identifiers
        if pd.notna(row['contract_number']):
            sections.append(f"Contract Number: {row['contract_number']}")
        
        return "\n".join(sections)
    
    def _combine_content_sections(self, *sections: str) -> str:
        """Combine content sections with proper formatting"""
        valid_sections = [section for section in sections if section.strip()]
        return "\n\n".join(valid_sections)
    
    def _create_chunk_metadata(self, row: pd.Series, tender_idx: int, chunk_idx: int, chunk_text: str) -> Dict[str, Any]:
        """Create comprehensive metadata for chunk"""
        metadata = {
            # Identifiers
            'tender_id': tender_idx,
            'chunk_id': chunk_idx,
            'contract_number': str(row.get('contract_number', '')),
            'process_number': str(row.get('process_number', '')),
            
            # Content classification
            'content_type': self._classify_chunk_content(chunk_text),
            'has_pricing': self._contains_pricing_info(chunk_text),
            'has_technical': self._contains_technical_info(chunk_text),
            
            # Contract details
            'contract_type': str(row.get('contract_type', '')),
            'contract_modality': str(row.get('contract_modality', '')),
            'process_status': str(row.get('process_status', '')),
            
            # Geographic
            'entity_department': str(row.get('entity_department', '')),
            'entity_municipality': str(row.get('entity_municipality', '')),
            
            # Pricing
            'contract_value': float(row['contract_value']) if pd.notna(row['contract_value']) else None,
            'price_range': self._categorize_price_range(row.get('contract_value')),
            
            # Temporal
            'execution_end_date': str(row.get('execution_end_date', '')),
            'execution_year': self._extract_year(row.get('execution_end_date')),
            
            # Language
            'original_language': str(row.get('original_language', '')),
            'translated_language': str(row.get('translated_language', '')),
            
            # Contractor
            'contractor_name': str(row.get('contractor_name', '')),
            
            # Chunk characteristics
            'chunk_length': len(chunk_text),
            'chunk_word_count': len(chunk_text.split()),
        }
        
        return metadata
    
    def _classify_chunk_content(self, chunk_text: str) -> str:
        """Classify the type of content in the chunk"""
        text_lower = chunk_text.lower()
        
        # Check for pricing content
        if any(keyword in text_lower for keyword in self.pricing_keywords):
            return 'pricing'
        
        # Check for technical specifications
        if any(keyword in text_lower for keyword in ['specification', 'technical', 'requirement']):
            return 'technical'
        
        # Check for contract details
        if any(keyword in text_lower for keyword in self.structure_keywords):
            return 'contractual'
        
        # Check for administrative content
        if any(keyword in text_lower for keyword in ['department', 'municipality', 'contractor']):
            return 'administrative'
        
        return 'general'
    
    def _contains_pricing_info(self, chunk_text: str) -> bool:
        """Check if chunk contains pricing information"""
        text_lower = chunk_text.lower()
        return any(keyword in text_lower for keyword in self.pricing_keywords)
    
    def _contains_technical_info(self, chunk_text: str) -> bool:
        """Check if chunk contains technical information"""
        text_lower = chunk_text.lower()
        technical_keywords = [
            'technical', 'specification', 'requirement', 'standard',
            'técnico', 'especificación', 'requisito', 'estándar'
        ]
        return any(keyword in text_lower for keyword in technical_keywords)
    
    def _categorize_price_range(self, contract_value: Any) -> str:
        """Categorize contract value into price ranges"""
        if pd.isna(contract_value):
            return 'unknown'
        
        try:
            value = float(contract_value)
            if value < 50000:
                return '0-50K'
            elif value < 200000:
                return '50K-200K'
            elif value < 1000000:
                return '200K-1M'
            elif value < 5000000:
                return '1M-5M'
            else:
                return '5M+'
        except (ValueError, TypeError):
            return 'unknown'
    
    def _extract_year(self, date_str: Any) -> Optional[int]:
        """Extract year from date string"""
        if pd.isna(date_str):
            return None
        
        try:
            # Try to parse as datetime
            date_obj = pd.to_datetime(date_str)
            return date_obj.year
        except:
            # Try to extract year with regex
            year_match = re.search(r'(\d{4})', str(date_str))
            if year_match:
                return int(year_match.group(1))
            return None
    
    def create_specialized_chunks(self, df: pd.DataFrame) -> Dict[str, List[Document]]:
        """Create specialized chunks for different collection types"""
        specialized_chunks = {
            'pricing_focused': [],
            'geographic_focused': [],
            'contractor_focused': [],
            'technical_focused': []
        }
        
        # Process each tender
        for idx, row in df.iterrows():
            # Pricing-focused chunks
            pricing_chunks = self._create_pricing_chunks(row, idx)
            specialized_chunks['pricing_focused'].extend(pricing_chunks)
            
            # Geographic-focused chunks
            geographic_chunks = self._create_geographic_chunks(row, idx)
            specialized_chunks['geographic_focused'].extend(geographic_chunks)
            
            # Contractor-focused chunks
            contractor_chunks = self._create_contractor_chunks(row, idx)
            specialized_chunks['contractor_focused'].extend(contractor_chunks)
            
            # Technical-focused chunks
            technical_chunks = self._create_technical_chunks(row, idx)
            specialized_chunks['technical_focused'].extend(technical_chunks)
        
        return specialized_chunks
    
    def _create_pricing_chunks(self, row: pd.Series, tender_idx: int) -> List[Document]:
        """Create chunks optimized for pricing queries"""
        chunks = []
        
        # Primary pricing content
        pricing_content = []
        
        if pd.notna(row['contract_value']):
            pricing_content.append(f"Contract Value: ${float(row['contract_value']):,.2f}")
        
        if pd.notna(row['contract_object']):
            pricing_content.append(f"Service: {row['contract_object']}")
        
        if pd.notna(row['contract_type']):
            pricing_content.append(f"Type: {row['contract_type']}")
        
        if pd.notna(row['entity_department']):
            pricing_content.append(f"Location: {row['entity_department']}")
        
        if pd.notna(row['execution_end_date']):
            pricing_content.append(f"Duration: {row['execution_end_date']}")
        
        if pricing_content:
            content = "\n".join(pricing_content)
            metadata = self._create_chunk_metadata(row, tender_idx, 0, content)
            metadata['chunk_type'] = 'pricing_focused'
            
            chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks
    
    def _create_geographic_chunks(self, row: pd.Series, tender_idx: int) -> List[Document]:
        """Create chunks optimized for geographic queries"""
        chunks = []
        
        geo_content = []
        
        if pd.notna(row['entity_department']):
            geo_content.append(f"Department: {row['entity_department']}")
        
        if pd.notna(row['entity_municipality']) and row['entity_municipality'] != 'Not defined':
            geo_content.append(f"Municipality: {row['entity_municipality']}")
        
        if pd.notna(row['contract_object']):
            geo_content.append(f"Service: {row['contract_object']}")
        
        if pd.notna(row['contract_value']):
            geo_content.append(f"Value: ${float(row['contract_value']):,.2f}")
        
        if geo_content:
            content = "\n".join(geo_content)
            metadata = self._create_chunk_metadata(row, tender_idx, 0, content)
            metadata['chunk_type'] = 'geographic_focused'
            
            chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks
    
    def _create_contractor_chunks(self, row: pd.Series, tender_idx: int) -> List[Document]:
        """Create chunks optimized for contractor queries"""
        chunks = []
        
        if pd.notna(row['contractor_name']):
            contractor_content = []
            
            contractor_content.append(f"Contractor: {row['contractor_name']}")
            
            if pd.notna(row['contract_object']):
                contractor_content.append(f"Service: {row['contract_object']}")
            
            if pd.notna(row['contract_value']):
                contractor_content.append(f"Value: ${float(row['contract_value']):,.2f}")
            
            if pd.notna(row['process_status']):
                contractor_content.append(f"Status: {row['process_status']}")
            
            if pd.notna(row['entity_department']):
                contractor_content.append(f"Location: {row['entity_department']}")
            
            content = "\n".join(contractor_content)
            metadata = self._create_chunk_metadata(row, tender_idx, 0, content)
            metadata['chunk_type'] = 'contractor_focused'
            
            chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks
    
    def _create_technical_chunks(self, row: pd.Series, tender_idx: int) -> List[Document]:
        """Create chunks optimized for technical queries"""
        chunks = []
        
        # Focus on detailed contract and process objects
        technical_content = []
        
        if pd.notna(row['contract_object']):
            technical_content.append(f"Contract Object: {row['contract_object']}")
        
        if pd.notna(row['process_object']) and row['process_object'] != row['contract_object']:
            technical_content.append(f"Process Object: {row['process_object']}")
        
        if pd.notna(row['contract_type']):
            technical_content.append(f"Contract Type: {row['contract_type']}")
        
        if pd.notna(row['contract_modality']):
            technical_content.append(f"Contract Modality: {row['contract_modality']}")
        
        if technical_content:
            content = "\n".join(technical_content)
            metadata = self._create_chunk_metadata(row, tender_idx, 0, content)
            metadata['chunk_type'] = 'technical_focused'
            
            chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks
    
    def get_chunking_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about the chunking process"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        word_counts = [len(chunk.page_content.split()) for chunk in chunks]
        
        # Content type distribution
        content_types = [chunk.metadata.get('content_type', 'unknown') for chunk in chunks]
        content_type_counts = pd.Series(content_types).value_counts().to_dict()
        
        # Pricing chunk distribution
        pricing_chunks = sum(1 for chunk in chunks if chunk.metadata.get('has_pricing', False))
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunks),
            'avg_word_count': sum(word_counts) / len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'content_type_distribution': content_type_counts,
            'pricing_chunks': pricing_chunks,
            'pricing_chunk_ratio': pricing_chunks / len(chunks)
        }


def main():
    """Test the chunking system"""
    print("TenderChunkingSystem ready for use.")
    print("Use with DataFrame of tender data to create optimized chunks.")


if __name__ == "__main__":
    main()