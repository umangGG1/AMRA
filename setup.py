#!/usr/bin/env python3
"""
Setup script for AMRA Healthcare Tender Pricing System
Initializes the complete RAG pipeline with ChromaDB and Mistral-3B
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging

# Add src to path
sys.path.append('./src')

from chromadb_manager import TenderChromaManager
from rag_pipeline import TenderRAGPipeline
from evaluation_framework import TenderRAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    try:
        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "chroma_db",
        "logs",
        "evaluation_results",
        "data/processed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def initialize_database():
    """Initialize ChromaDB with tender data"""
    try:
        logger.info("Initializing ChromaDB...")
        
        # Check if tender data exists
        data_path = "./data/healthcare_contracts_cleaned_20250718_185226_searchable.json"
        if not Path(data_path).exists():
            logger.error(f"Tender data not found at {data_path}")
            return False
        
        # Initialize ChromaDB manager
        chroma_manager = TenderChromaManager(persist_directory="./chroma_db")
        
        # Create collections
        chroma_manager.create_collections()
        
        # Load and process data
        df = chroma_manager.load_tender_data(data_path)
        df_processed = chroma_manager.preprocess_tender_data(df)
        
        # Populate collections
        chroma_manager.populate_collections(df_processed)
        
        # Get statistics
        stats = chroma_manager.get_collection_stats()
        
        logger.info("ChromaDB initialized successfully")
        logger.info(f"Collection statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def test_rag_pipeline():
    """Test the RAG pipeline with sample queries"""
    try:
        logger.info("Testing RAG pipeline...")
        
        # Initialize pipeline
        pipeline = TenderRAGPipeline(
            model_name="mistralai/Mistral-3B-Instruct-v0.2",
            chroma_persist_dir="./chroma_db"
        )
        
        # Test query
        test_query = "What is the average price for healthcare services in Bogotá?"
        
        logger.info(f"Testing query: {test_query}")
        result = pipeline.query(test_query)
        
        logger.info(f"Query successful. Response time: {result['response_time']:.2f}s")
        logger.info(f"Response preview: {result['response'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"RAG pipeline test failed: {e}")
        return False

def run_evaluation():
    """Run evaluation on the system"""
    try:
        logger.info("Running system evaluation...")
        
        # Initialize pipeline
        pipeline = TenderRAGPipeline(chroma_persist_dir="./chroma_db")
        
        # Initialize evaluator
        evaluator = TenderRAGEvaluator(pipeline)
        
        # Run evaluation (subset for setup)
        evaluation_results = evaluator.run_comprehensive_evaluation()
        
        # Save results
        evaluator.save_evaluation_results("./evaluation_results/setup_evaluation.json")
        evaluator.generate_evaluation_report("./evaluation_results/setup_evaluation.md")
        
        logger.info("Evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False

def create_env_file():
    """Create environment file with default settings"""
    env_content = """# AMRA Healthcare Tender Pricing System Configuration

# Model Configuration
MODEL_NAME=mistralai/Mistral-3B-Instruct-v0.2
EMBEDDING_MODEL=all-mpnet-base-v2
MAX_TOKENS=512
TEMPERATURE=0.1

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=tender_pricing

# Data Configuration
DATA_PATH=./data/healthcare_contracts_cleaned_20250718_185226_searchable.json
PROCESSED_DATA_PATH=./data/processed/

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    logger.info("Environment file created")

def main():
    """Main setup function"""
    logger.info("Starting AMRA Healthcare Tender Pricing System setup...")
    
    # Step 1: Create directories
    setup_directories()
    
    # Step 2: Create environment file
    create_env_file()
    
    # Step 3: Install requirements
    if not install_requirements():
        logger.error("Setup failed at requirements installation")
        return False
    
    # Step 4: Initialize database
    if not initialize_database():
        logger.error("Setup failed at database initialization")
        return False
    
    # Step 5: Test RAG pipeline
    if not test_rag_pipeline():
        logger.error("Setup failed at RAG pipeline test")
        return False
    
    # Step 6: Run evaluation
    if not run_evaluation():
        logger.warning("Evaluation failed, but setup can continue")
    
    logger.info("Setup completed successfully!")
    
    # Print next steps
    print("\n" + "="*60)
    print("AMRA Healthcare Tender Pricing System - Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the Streamlit application:")
    print("   streamlit run app.py")
    print("\n2. Or test the system directly:")
    print("   python test_chromadb.py")
    print("\n3. View evaluation results:")
    print("   cat evaluation_results/setup_evaluation.md")
    print("\n4. Check logs:")
    print("   tail -f logs/app.log")
    print("\nSystem is ready for use!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)