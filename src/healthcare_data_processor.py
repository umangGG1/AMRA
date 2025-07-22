"""
Healthcare Contract Data Processor
Specialized preprocessing for healthcare contract data to optimize vector database performance
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDataProcessor:
    """Specialized processor for healthcare contract data"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.service_keywords = {
            'medical_services': ['medical', 'health', 'clinical', 'hospital', 'patient', 'care', 'treatment'],
            'professional_services': ['professional', 'technical', 'support', 'advisory', 'consulting', 'management'],
            'equipment_supply': ['equipment', 'supply', 'material', 'device', 'instrument', 'apparatus'],
            'training_education': ['training', 'education', 'workshop', 'capacity', 'course', 'learning'],
            'maintenance_technical': ['maintenance', 'repair', 'installation', 'calibration', 'servicing'],
            'research_analysis': ['research', 'study', 'analysis', 'evaluation', 'assessment', 'investigation']
        }
        
        self.entity_keywords = {
            'ministry': ['ministry', 'ministerio'],
            'institute': ['institute', 'instituto'],
            'fund': ['fund', 'fondo'],
            'healthcare_facility': ['hospital', 'clinic', 'health center', 'medical center'],
            'university': ['university', 'universidad'],
            'department': ['department', 'departamento', 'secretariat']
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load healthcare contract data from JSON file"""
        logger.info(f"Loading healthcare contract data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} healthcare contract records")
        
        # Display basic info about the data
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Contract value range: ${df['contract_value'].min():,.0f} - ${df['contract_value'].max():,.0f}")
        
        return df
    
    def clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the data to remove noise and outliers"""
        logger.info("Starting data cleaning and filtering...")
        original_count = len(df)
        
        # Filter extreme values (likely data quality issues)
        df = df[
            (df['contract_value'] >= 1000) & 
            (df['contract_value'] <= 10_000_000_000)
        ].copy()
        
        # Remove records with missing critical fields
        df = df.dropna(subset=['contract_object', 'contract_value', 'entity_name'])
        
        # Clean text fields
        df['contract_object'] = df['contract_object'].str.strip()
        df['entity_name'] = df['entity_name'].str.strip()
        df['contractor_name'] = df['contractor_name'].str.strip()
        
        # Standardize department names
        df['entity_department'] = df['entity_department'].apply(self._standardize_department)
        
        filtered_count = len(df)
        logger.info(f"Filtered data: {original_count} -> {filtered_count} records ({original_count-filtered_count} removed)")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better categorization and filtering"""
        logger.info("Adding derived features...")
        
        # Convert and process dates
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df['execution_year'] = df['end_date'].dt.year
        df['execution_quarter'] = df['end_date'].dt.quarter
        
        # Create price brackets
        df['price_bracket'] = pd.cut(
            df['contract_value'],
            bins=[0, 50_000, 200_000, 1_000_000, 5_000_000, float('inf')],
            labels=['0-50K', '50K-200K', '200K-1M', '1M-5M', '5M+']
        ).astype(str)
        
        # Regional categorization
        df['region'] = df['entity_department'].apply(self._categorize_region)
        
        # Service categorization
        df['service_category'] = df['contract_object'].apply(self._categorize_service)
        
        # Entity type categorization
        df['entity_type'] = df['entity_name'].apply(self._categorize_entity_type)
        
        # Complexity level
        df['complexity_level'] = df.apply(self._calculate_complexity, axis=1)
        
        # Performance indicators
        df['performance_category'] = df.apply(self._categorize_performance, axis=1)
        df['success_rate'] = df['process_status'].apply(self._calculate_success_rate)
        df['value_range'] = df['contract_value'].apply(self._categorize_value_range)
        
        # Content-based features
        df['service_keywords'] = df['contract_object'].apply(self._extract_service_keywords)
        df['contract_complexity_score'] = df.apply(self._calculate_contract_complexity, axis=1)
        
        logger.info("Derived features added successfully")
        return df
    
    def _standardize_department(self, department: str) -> str:
        """Standardize department names"""
        if pd.isna(department):
            return 'Unknown'
        
        dept = department.strip()
        
        # Common standardizations
        standardizations = {
            'Capital District of Bogotá': 'Bogotá D.C.',
            'Distrito Capital de Bogotá': 'Bogotá D.C.',
            'Bogotá': 'Bogotá D.C.'
        }
        
        return standardizations.get(dept, dept)
    
    def _categorize_region(self, department: str) -> str:
        """Categorize regions for geographic analysis"""
        if pd.isna(department):
            return 'unknown'
        
        dept_lower = department.lower()
        
        if any(word in dept_lower for word in ['bogotá', 'distrito', 'capital']):
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
        """Categorize service type based on contract description"""
        if pd.isna(contract_object):
            return 'unknown'
        
        text_lower = contract_object.lower()
        
        for category, keywords in self.service_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general_services'
    
    def _categorize_entity_type(self, entity_name: str) -> str:
        """Categorize entity type"""
        if pd.isna(entity_name):
            return 'unknown'
        
        entity_lower = entity_name.lower()
        
        for entity_type, keywords in self.entity_keywords.items():
            if any(keyword in entity_lower for keyword in keywords):
                return entity_type
        
        return 'other_entity'
    
    def _calculate_complexity(self, row) -> str:
        """Calculate service complexity based on multiple factors"""
        value = row['contract_value']
        content_length = row.get('content_length', len(str(row['contract_object'])))
        keyword_count = row.get('keyword_count', 0)
        
        complexity_score = 0
        
        # Value-based complexity
        if value > 5_000_000:
            complexity_score += 3
        elif value > 1_000_000:
            complexity_score += 2
        elif value > 200_000:
            complexity_score += 1
        
        # Content-based complexity
        if content_length > 500:
            complexity_score += 2
        elif content_length > 300:
            complexity_score += 1
        
        # Keyword diversity
        if keyword_count > 30:
            complexity_score += 2
        elif keyword_count > 20:
            complexity_score += 1
        
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_performance(self, row) -> str:
        """Categorize performance based on status and value"""
        status = row['process_status']
        value = row['contract_value']
        
        if pd.isna(status):
            return 'unknown'
        
        status_lower = status.lower()
        
        if any(word in status_lower for word in ['cancelled', 'terminated']):
            return 'cancelled'
        elif 'draft' in status_lower:
            return 'pending'
        elif any(word in status_lower for word in ['completed', 'finalized', 'sent']):
            if value > 1_000_000:
                return 'high_value_completed'
            elif value > 200_000:
                return 'medium_value_completed'
            else:
                return 'low_value_completed'
        else:
            return 'in_progress'
    
    def _calculate_success_rate(self, status: str) -> str:
        """Calculate success indicator based on process status"""
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
        
        if value < 50_000:
            return 'micro'
        elif value < 200_000:
            return 'small'
        elif value < 1_000_000:
            return 'medium'
        elif value < 5_000_000:
            return 'large'
        else:
            return 'mega'
    
    def _extract_service_keywords(self, contract_object: str) -> List[str]:
        """Extract relevant keywords from contract description"""
        if pd.isna(contract_object):
            return []
        
        # Simple keyword extraction
        text_lower = contract_object.lower()
        keywords = []
        
        for category, category_keywords in self.service_keywords.items():
            for keyword in category_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_contract_complexity(self, row) -> float:
        """Calculate a numerical complexity score"""
        score = 0.0
        
        # Value component (normalized)
        value_score = min(row['contract_value'] / 5_000_000, 1.0)
        score += value_score * 0.4
        
        # Content length component
        content_length = row.get('content_length', len(str(row['contract_object'])))
        content_score = min(content_length / 1000, 1.0)
        score += content_score * 0.3
        
        # Keyword diversity component
        keyword_count = row.get('keyword_count', 0)
        keyword_score = min(keyword_count / 50, 1.0)
        score += keyword_score * 0.3
        
        return round(score, 3)
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary report of the processed data"""
        report = {
            'total_contracts': len(df),
            'total_value': df['contract_value'].sum(),
            'avg_value': df['contract_value'].mean(),
            'median_value': df['contract_value'].median(),
            'value_distribution': {
                'micro': len(df[df['value_range'] == 'micro']),
                'small': len(df[df['value_range'] == 'small']),
                'medium': len(df[df['value_range'] == 'medium']),
                'large': len(df[df['value_range'] == 'large']),
                'mega': len(df[df['value_range'] == 'mega'])
            },
            'service_categories': df['service_category'].value_counts().to_dict(),
            'regions': df['region'].value_counts().to_dict(),
            'entity_types': df['entity_type'].value_counts().to_dict(),
            'complexity_levels': df['complexity_level'].value_counts().to_dict(),
            'year_distribution': df['execution_year'].value_counts().sort_index().to_dict()
        }
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to JSON file"""
        logger.info(f"Saving processed data to {output_path}")
        
        # Convert to records for JSON serialization
        records = df.to_dict('records')
        
        # Handle datetime serialization
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat() if not pd.isna(value) else None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed data saved successfully")


def main():
    """Main processing pipeline"""
    processor = HealthcareDataProcessor()
    
    # Load data
    df = processor.load_data(
        "/home/umanggod/AMRA-healthcare-POC/data/healthcare_contracts_cleaned_20250717_230006.json"
    )
    
    # Process data
    df_cleaned = processor.clean_and_filter(df)
    df_processed = processor.add_derived_features(df_cleaned)
    
    # Generate report
    report = processor.generate_summary_report(df_processed)
    
    print("\n=== Healthcare Contract Data Processing Report ===")
    print(f"Total contracts: {report['total_contracts']}")
    print(f"Total value: ${report['total_value']:,.0f}")
    print(f"Average value: ${report['avg_value']:,.0f}")
    print(f"Median value: ${report['median_value']:,.0f}")
    
    print("\nValue Distribution:")
    for range_name, count in report['value_distribution'].items():
        print(f"  {range_name}: {count}")
    
    print("\nService Categories:")
    for category, count in list(report['service_categories'].items())[:5]:
        print(f"  {category}: {count}")
    
    print("\nRegional Distribution:")
    for region, count in report['regions'].items():
        print(f"  {region}: {count}")
    
    # Save processed data
    output_path = "/home/umanggod/AMRA-healthcare-POC/data/processed/healthcare_contracts_processed.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    processor.save_processed_data(df_processed, output_path)
    
    print(f"\nProcessed data saved to: {output_path}")


if __name__ == "__main__":
    main()