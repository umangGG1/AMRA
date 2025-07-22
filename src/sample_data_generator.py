"""
Sample Tender Data Generator
Creates realistic healthcare tender data for testing and development
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict
import json

class TenderDataGenerator:
    """Generates realistic tender data for healthcare contracts"""
    
    def __init__(self, num_records: int = 2000):
        self.num_records = num_records
        self.setup_data_pools()
    
    def setup_data_pools(self):
        """Setup data pools for realistic generation"""
        self.healthcare_categories = [
            'Medical Equipment', 'Pharmaceutical Services', 'Laboratory Services',
            'Radiology Equipment', 'Surgical Instruments', 'IT Healthcare Systems',
            'Diagnostic Equipment', 'Patient Monitoring', 'Rehabilitation Services',
            'Emergency Services', 'Dental Equipment', 'Mental Health Services'
        ]
        
        self.regions = [
            'North America', 'Europe', 'Asia-Pacific', 'Latin America',
            'Middle East', 'Africa', 'Oceania'
        ]
        
        self.tender_types = [
            'Public Procurement', 'Private Contract', 'Government RFP',
            'Hospital Tender', 'Insurance Contract', 'Research Grant'
        ]
        
        self.service_descriptions = [
            'Comprehensive medical equipment maintenance and support services',
            'Advanced diagnostic imaging systems with training and warranty',
            'Laboratory testing services including pathology and clinical chemistry',
            'Surgical instrument procurement and sterilization services',
            'Electronic health record system implementation and maintenance',
            'Patient monitoring systems with 24/7 technical support',
            'Pharmaceutical distribution and inventory management services',
            'Rehabilitation equipment and therapy services',
            'Emergency medical services and ambulance fleet management',
            'Dental care equipment and maintenance contracts'
        ]
        
        self.requirement_templates = [
            'Equipment must comply with ISO 13485 medical device standards',
            'Service provider must have minimum 5 years experience',
            'All staff must be certified in healthcare technology',
            'Response time for emergency repairs must be under 4 hours',
            'System must integrate with existing hospital infrastructure',
            'Training must be provided for all end users',
            'Warranty period minimum 2 years with parts and labor',
            'Compliance with HIPAA and local data protection regulations'
        ]
    
    def generate_tender_data(self) -> pd.DataFrame:
        """Generate complete tender dataset"""
        data = []
        
        for i in range(self.num_records):
            record = self._generate_single_tender(i)
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _generate_single_tender(self, tender_id: int) -> Dict:
        """Generate a single tender record"""
        category = random.choice(self.healthcare_categories)
        region = random.choice(self.regions)
        tender_type = random.choice(self.tender_types)
        
        # Generate pricing based on category
        base_price = self._generate_base_price(category)
        
        # Generate dates
        issue_date = datetime.now() - timedelta(days=random.randint(1, 365))
        deadline = issue_date + timedelta(days=random.randint(14, 90))
        
        # Generate text content
        description = self._generate_description(category)
        requirements = self._generate_requirements()
        specifications = self._generate_specifications(category)
        
        return {
            'tender_id': f"TND-{tender_id:06d}",
            'title': f"{category} - {tender_type}",
            'category': category,
            'region': region,
            'tender_type': tender_type,
            'issue_date': issue_date.strftime('%Y-%m-%d'),
            'deadline': deadline.strftime('%Y-%m-%d'),
            'estimated_value': base_price,
            'currency': 'USD',
            'description': description,
            'requirements': requirements,
            'technical_specifications': specifications,
            'contract_duration': random.randint(12, 60),  # months
            'evaluation_criteria': self._generate_evaluation_criteria(),
            'contact_info': self._generate_contact_info(),
            'documents_required': self._generate_required_documents(),
            'pricing_breakdown': self._generate_pricing_breakdown(base_price),
            'payment_terms': self._generate_payment_terms(),
            'warranty_terms': f"{random.randint(12, 36)} months comprehensive warranty",
            'compliance_requirements': self._generate_compliance_requirements(),
            'submission_format': random.choice(['Online Portal', 'Email', 'Physical Mail', 'Hybrid']),
            'language': random.choice(['English', 'Spanish', 'French', 'German', 'Portuguese']),
            'status': random.choice(['Open', 'Closed', 'Awarded', 'Cancelled']),
            'winner': self._generate_winner() if random.random() > 0.7 else None
        }
    
    def _generate_base_price(self, category: str) -> float:
        """Generate realistic base price based on category"""
        price_ranges = {
            'Medical Equipment': (50000, 2000000),
            'Pharmaceutical Services': (25000, 500000),
            'Laboratory Services': (30000, 800000),
            'Radiology Equipment': (100000, 5000000),
            'Surgical Instruments': (20000, 300000),
            'IT Healthcare Systems': (75000, 2500000),
            'Diagnostic Equipment': (40000, 1500000),
            'Patient Monitoring': (30000, 600000),
            'Rehabilitation Services': (15000, 200000),
            'Emergency Services': (100000, 1000000),
            'Dental Equipment': (10000, 150000),
            'Mental Health Services': (20000, 300000)
        }
        
        min_price, max_price = price_ranges.get(category, (25000, 500000))
        return round(random.uniform(min_price, max_price), 2)
    
    def _generate_description(self, category: str) -> str:
        """Generate detailed description"""
        base_desc = random.choice(self.service_descriptions)
        
        additional_details = [
            f"This tender is for {category.lower()} solutions serving healthcare facilities.",
            f"The contract includes installation, training, and ongoing support services.",
            f"All equipment must meet current healthcare industry standards and regulations.",
            f"The successful bidder will provide comprehensive documentation and user training.",
            f"Service level agreements include defined response times and performance metrics."
        ]
        
        selected_details = random.sample(additional_details, random.randint(2, 4))
        return base_desc + " " + " ".join(selected_details)
    
    def _generate_requirements(self) -> str:
        """Generate tender requirements"""
        requirements = random.sample(self.requirement_templates, random.randint(3, 6))
        return "; ".join(requirements)
    
    def _generate_specifications(self, category: str) -> str:
        """Generate technical specifications"""
        specs = [
            f"Category: {category}",
            f"Compatibility: Must integrate with existing systems",
            f"Standards: ISO 13485, FDA approved where applicable",
            f"Support: 24/7 technical support included",
            f"Training: Comprehensive user training program",
            f"Documentation: Complete technical and user documentation",
            f"Installation: Professional installation and commissioning"
        ]
        
        return "; ".join(specs)
    
    def _generate_evaluation_criteria(self) -> str:
        """Generate evaluation criteria"""
        criteria = [
            "Technical capability (40%)",
            "Cost effectiveness (30%)",
            "Experience and references (20%)",
            "Timeline and delivery (10%)"
        ]
        
        return "; ".join(criteria)
    
    def _generate_contact_info(self) -> str:
        """Generate contact information"""
        return f"procurement@healthcare-{random.randint(100, 999)}.org"
    
    def _generate_required_documents(self) -> str:
        """Generate required documents list"""
        docs = [
            "Technical proposal",
            "Cost breakdown",
            "Company profile",
            "References",
            "Certifications",
            "Insurance certificates"
        ]
        
        selected_docs = random.sample(docs, random.randint(4, 6))
        return "; ".join(selected_docs)
    
    def _generate_pricing_breakdown(self, base_price: float) -> str:
        """Generate pricing breakdown"""
        equipment_cost = base_price * 0.6
        service_cost = base_price * 0.25
        training_cost = base_price * 0.10
        warranty_cost = base_price * 0.05
        
        return f"Equipment: ${equipment_cost:,.2f}; Service: ${service_cost:,.2f}; Training: ${training_cost:,.2f}; Warranty: ${warranty_cost:,.2f}"
    
    def _generate_payment_terms(self) -> str:
        """Generate payment terms"""
        terms = [
            "30% advance, 60% on delivery, 10% after acceptance",
            "50% on order, 40% on delivery, 10% after commissioning",
            "Monthly payments over contract duration",
            "Quarterly payments with performance milestones"
        ]
        
        return random.choice(terms)
    
    def _generate_compliance_requirements(self) -> str:
        """Generate compliance requirements"""
        requirements = [
            "HIPAA compliance mandatory",
            "ISO 13485 certification required",
            "FDA approval where applicable",
            "Local healthcare regulations compliance",
            "Data security and privacy standards"
        ]
        
        return "; ".join(random.sample(requirements, random.randint(3, 5)))
    
    def _generate_winner(self) -> str:
        """Generate winner information"""
        companies = [
            "MedTech Solutions Ltd.",
            "Healthcare Innovations Inc.",
            "Global Medical Systems",
            "Advanced Healthcare Technologies",
            "Premier Medical Equipment Corp."
        ]
        
        return random.choice(companies)
    
    def save_sample_data(self, output_path: str):
        """Generate and save sample data"""
        df = self.generate_tender_data()
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} tender records and saved to {output_path}")
        
        # Print sample statistics
        print("\nSample Data Statistics:")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Regions: {df['region'].nunique()}")
        print(f"Average Tender Value: ${df['estimated_value'].mean():,.2f}")
        print(f"Price Range: ${df['estimated_value'].min():,.2f} - ${df['estimated_value'].max():,.2f}")
        
        return df


def main():
    """Generate sample tender data"""
    generator = TenderDataGenerator(num_records=2000)
    df = generator.save_sample_data("/home/umanggod/AMRA-healthcare-POC/data/tender_data.csv")
    
    print("\nFirst 3 records:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()