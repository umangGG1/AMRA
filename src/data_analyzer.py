"""
Data Analysis Module for Healthcare Tender Data
Analyzes structure, quality, and pricing patterns in tender documents
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TenderDataAnalyzer:
    """Analyzes tender data structure and quality for RAG optimization"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.analysis_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load tender data from various formats"""
        try:
            if self.data_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(self.data_path)
            elif self.data_path.suffix.lower() == '.json':
                self.df = pd.read_json(self.data_path)
            elif self.data_path.suffix.lower() in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            logger.info(f"Loaded {len(self.df)} tender records from {self.data_path}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_structure(self) -> Dict:
        """Analyze data structure and identify key fields"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        structure_analysis = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
        }
        
        # Identify potential pricing fields
        pricing_fields = self._identify_pricing_fields()
        structure_analysis['pricing_fields'] = pricing_fields
        
        # Identify text fields for RAG
        text_fields = self._identify_text_fields()
        structure_analysis['text_fields'] = text_fields
        
        # Identify categorical fields
        categorical_fields = self._identify_categorical_fields()
        structure_analysis['categorical_fields'] = categorical_fields
        
        self.analysis_results['structure'] = structure_analysis
        return structure_analysis
    
    def _identify_pricing_fields(self) -> List[str]:
        """Identify columns that likely contain pricing information"""
        pricing_keywords = [
            'price', 'cost', 'amount', 'budget', 'value', 'fee', 'rate',
            'total', 'sum', 'payment', 'charge', 'tariff', 'quote'
        ]
        
        pricing_fields = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in pricing_keywords):
                pricing_fields.append(col)
            elif self.df[col].dtype in ['int64', 'float64']:
                # Check if numeric field contains price-like values
                if self._is_likely_price_field(col):
                    pricing_fields.append(col)
        
        return pricing_fields
    
    def _is_likely_price_field(self, column: str) -> bool:
        """Check if numeric column likely contains prices"""
        try:
            values = self.df[column].dropna()
            if len(values) == 0:
                return False
            
            # Check for reasonable price ranges (assuming currency values)
            min_val, max_val = values.min(), values.max()
            
            # Prices should be positive and in reasonable ranges
            if min_val < 0 or max_val > 1e12:  # Extremely large values unlikely to be prices
                return False
            
            # Check for decimal places (common in pricing)
            decimal_count = sum(1 for val in values if val != int(val))
            if decimal_count > len(values) * 0.1:  # >10% have decimals
                return True
            
            return False
        except:
            return False
    
    def _identify_text_fields(self) -> List[str]:
        """Identify columns suitable for RAG text processing"""
        text_fields = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check average text length
                avg_length = self.df[col].dropna().str.len().mean()
                if avg_length > 50:  # Longer text fields more suitable for RAG
                    text_fields.append(col)
        
        return text_fields
    
    def _identify_categorical_fields(self) -> List[str]:
        """Identify categorical fields for metadata filtering"""
        categorical_fields = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_count = self.df[col].nunique()
                total_count = len(self.df[col].dropna())
                
                # If <20% unique values, likely categorical
                if unique_count < total_count * 0.2:
                    categorical_fields.append(col)
        
        return categorical_fields
    
    def analyze_pricing_patterns(self) -> Dict:
        """Analyze pricing patterns and distributions"""
        if 'structure' not in self.analysis_results:
            self.analyze_structure()
        
        pricing_fields = self.analysis_results['structure']['pricing_fields']
        pricing_analysis = {}
        
        for field in pricing_fields:
            values = self.df[field].dropna()
            if len(values) > 0:
                pricing_analysis[field] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'percentiles': {
                        '25th': float(values.quantile(0.25)),
                        '75th': float(values.quantile(0.75)),
                        '90th': float(values.quantile(0.9)),
                        '95th': float(values.quantile(0.95))
                    }
                }
        
        self.analysis_results['pricing_patterns'] = pricing_analysis
        return pricing_analysis
    
    def analyze_text_quality(self) -> Dict:
        """Analyze text quality for RAG optimization"""
        if 'structure' not in self.analysis_results:
            self.analyze_structure()
        
        text_fields = self.analysis_results['structure']['text_fields']
        text_analysis = {}
        
        for field in text_fields:
            text_values = self.df[field].dropna()
            if len(text_values) > 0:
                text_analysis[field] = {
                    'count': len(text_values),
                    'avg_length': float(text_values.str.len().mean()),
                    'max_length': int(text_values.str.len().max()),
                    'min_length': int(text_values.str.len().min()),
                    'empty_rate': float((text_values == '').sum() / len(text_values)),
                    'unique_rate': float(text_values.nunique() / len(text_values))
                }
                
                # Analyze language characteristics
                sample_text = ' '.join(text_values.head(100))
                text_analysis[field]['language_stats'] = self._analyze_language(sample_text)
        
        self.analysis_results['text_quality'] = text_analysis
        return text_analysis
    
    def _analyze_language(self, text: str) -> Dict:
        """Basic language analysis for chunking optimization"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1)
        }
    
    def generate_rag_recommendations(self) -> Dict:
        """Generate recommendations for RAG optimization"""
        if not self.analysis_results:
            self.analyze_structure()
            self.analyze_pricing_patterns()
            self.analyze_text_quality()
        
        recommendations = {
            'chunking_strategy': self._recommend_chunking_strategy(),
            'metadata_fields': self._recommend_metadata_fields(),
            'embedding_focus': self._recommend_embedding_focus(),
            'collection_schema': self._recommend_collection_schema()
        }
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def _recommend_chunking_strategy(self) -> Dict:
        """Recommend chunking strategy based on text analysis"""
        text_analysis = self.analysis_results.get('text_quality', {})
        
        # Determine optimal chunk size based on text characteristics
        avg_lengths = [stats['avg_length'] for stats in text_analysis.values()]
        if avg_lengths:
            avg_text_length = sum(avg_lengths) / len(avg_lengths)
            
            if avg_text_length < 200:
                chunk_size = 256
            elif avg_text_length < 500:
                chunk_size = 512
            else:
                chunk_size = 1024
        else:
            chunk_size = 512  # Default
        
        return {
            'recommended_chunk_size': chunk_size,
            'overlap_ratio': 0.2,
            'preserve_structure': True,
            'split_on_sentences': True
        }
    
    def _recommend_metadata_fields(self) -> List[str]:
        """Recommend fields for metadata filtering"""
        categorical_fields = self.analysis_results['structure']['categorical_fields']
        pricing_fields = self.analysis_results['structure']['pricing_fields']
        
        # Prioritize categorical fields with good coverage
        metadata_fields = []
        for field in categorical_fields:
            missing_rate = self.analysis_results['structure']['missing_values'][field] / len(self.df)
            if missing_rate < 0.5:  # Less than 50% missing
                metadata_fields.append(field)
        
        # Add primary pricing field if available
        if pricing_fields:
            metadata_fields.append(pricing_fields[0])
        
        return metadata_fields
    
    def _recommend_embedding_focus(self) -> Dict:
        """Recommend embedding focus areas"""
        text_fields = self.analysis_results['structure']['text_fields']
        pricing_fields = self.analysis_results['structure']['pricing_fields']
        
        return {
            'primary_text_fields': text_fields[:3],  # Top 3 text fields
            'pricing_context_fields': pricing_fields[:2],  # Top 2 pricing fields
            'embedding_strategy': 'combined',  # Combine text and pricing context
            'weight_pricing_higher': True
        }
    
    def _recommend_collection_schema(self) -> Dict:
        """Recommend ChromaDB collection schema"""
        return {
            'collections': {
                'tender_documents': {
                    'description': 'Full tender documents with complete context',
                    'fields': self.analysis_results['structure']['text_fields']
                },
                'price_components': {
                    'description': 'Pricing-focused chunks with cost breakdowns',
                    'fields': self.analysis_results['structure']['pricing_fields']
                },
                'market_segments': {
                    'description': 'Categorized by industry/region for filtering',
                    'fields': self.analysis_results['structure']['categorical_fields']
                }
            }
        }
    
    def save_analysis(self, output_path: str):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")
    
    def print_summary(self):
        """Print analysis summary"""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_structure() first.")
            return
        
        structure = self.analysis_results['structure']
        print(f"\n{'='*50}")
        print(f"TENDER DATA ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total Records: {structure['total_records']:,}")
        print(f"Total Columns: {structure['total_columns']}")
        print(f"Memory Usage: {structure['memory_usage'] / 1024 / 1024:.2f} MB")
        
        print(f"\nPricing Fields ({len(structure['pricing_fields'])}):")
        for field in structure['pricing_fields']:
            print(f"  - {field}")
        
        print(f"\nText Fields for RAG ({len(structure['text_fields'])}):")
        for field in structure['text_fields']:
            print(f"  - {field}")
        
        print(f"\nCategorical Fields ({len(structure['categorical_fields'])}):")
        for field in structure['categorical_fields']:
            print(f"  - {field}")
        
        if 'recommendations' in self.analysis_results:
            rec = self.analysis_results['recommendations']
            print(f"\nRAG RECOMMENDATIONS:")
            print(f"  Chunk Size: {rec['chunking_strategy']['recommended_chunk_size']}")
            print(f"  Metadata Fields: {rec['metadata_fields']}")
            print(f"  Primary Text Fields: {rec['embedding_focus']['primary_text_fields']}")


def main():
    """Example usage of TenderDataAnalyzer"""
    # This would be run when actual data is available
    print("TenderDataAnalyzer ready for use.")
    print("Usage:")
    print("  analyzer = TenderDataAnalyzer('path/to/tender_data.csv')")
    print("  analyzer.load_data()")
    print("  analyzer.analyze_structure()")
    print("  analyzer.analyze_pricing_patterns()")
    print("  analyzer.generate_rag_recommendations()")
    print("  analyzer.print_summary()")


if __name__ == "__main__":
    main()