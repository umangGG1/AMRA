#!/usr/bin/env python3
"""
Virtual Environment Setup Script for AMRA Healthcare Tender Pricing System
Creates virtual environment and installs all dependencies
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging
import venv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("./venv")
    
    if venv_path.exists():
        logger.info("Virtual environment already exists")
        return True
    
    try:
        logger.info("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        logger.info("Virtual environment created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get path to virtual environment Python"""
    if os.name == 'nt':  # Windows
        return Path("./venv/Scripts/python.exe")
    else:  # Unix/Linux/macOS
        return Path("./venv/bin/python")

def get_venv_pip():
    """Get path to virtual environment pip"""
    if os.name == 'nt':  # Windows
        return Path("./venv/Scripts/pip.exe")
    else:  # Unix/Linux/macOS
        return Path("./venv/bin/pip")

def install_requirements_in_venv():
    """Install requirements in virtual environment"""
    try:
        pip_path = get_venv_pip()
        
        logger.info("Upgrading pip in virtual environment...")
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        
        logger.info("Installing requirements in virtual environment...")
        subprocess.check_call([str(pip_path), "install", "-r", "requirements.txt"])
        
        logger.info("Requirements installed successfully in virtual environment")
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

def initialize_database_in_venv():
    """Initialize ChromaDB in virtual environment"""
    try:
        python_path = get_venv_python()
        
        logger.info("Initializing ChromaDB in virtual environment...")
        
        # Check if tender data exists
        data_path = "./data/healthcare_contracts_cleaned_20250718_185226_searchable.json"
        if not Path(data_path).exists():
            logger.error(f"Tender data not found at {data_path}")
            return False
        
        # Run ChromaDB initialization script
        init_script = '''
import sys
sys.path.append("./src")
from chromadb_manager import TenderChromaManager

# Initialize ChromaDB manager
chroma_manager = TenderChromaManager(persist_directory="./chroma_db")

# Create collections
chroma_manager.create_collections()

# Load and process data
        df = chroma_manager.load_tender_data("./data/healthcare_contracts_cleaned_20250718_185226_searchable.json")
df_processed = chroma_manager.preprocess_tender_data(df)

# Populate collections
chroma_manager.populate_collections(df_processed)

# Get statistics
stats = chroma_manager.get_collection_stats()
print("ChromaDB initialized successfully")
print(f"Collection statistics: {stats}")
'''
        
        # Write temporary script
        with open("temp_init.py", "w") as f:
            f.write(init_script)
        
        # Run script in virtual environment
        result = subprocess.run([str(python_path), "temp_init.py"], 
                              capture_output=True, text=True)
        
        # Clean up
        os.remove("temp_init.py")
        
        if result.returncode == 0:
            logger.info("Database initialized successfully")
            print(result.stdout)
            return True
        else:
            logger.error(f"Database initialization failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def create_activation_scripts():
    """Create activation scripts for different platforms"""
    
    # Create activation script for Unix/Linux/macOS
    unix_script = '''#!/bin/bash
# AMRA Healthcare Tender Pricing System - Virtual Environment Activation

echo "🏥 AMRA Healthcare Tender Pricing System"
echo "Activating virtual environment..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:./src"

echo "✅ Virtual environment activated!"
echo ""
echo "Available commands:"
echo "  python test_chromadb.py          - Test ChromaDB setup"
echo "  python demo.py                   - Run complete demo"
echo "  streamlit run app.py             - Start web interface"
echo "  python -m src.rag_pipeline       - Test RAG pipeline"
echo "  deactivate                       - Exit virtual environment"
echo ""
'''
    
    with open("activate.sh", "w") as f:
        f.write(unix_script)
    
    # Create activation script for Windows
    windows_script = '''@echo off
REM AMRA Healthcare Tender Pricing System - Virtual Environment Activation

echo 🏥 AMRA Healthcare Tender Pricing System
echo Activating virtual environment...

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;.\\src

echo ✅ Virtual environment activated!
echo.
echo Available commands:
echo   python test_chromadb.py          - Test ChromaDB setup
echo   python demo.py                   - Run complete demo
echo   streamlit run app.py             - Start web interface
echo   python -m src.rag_pipeline       - Test RAG pipeline
echo   deactivate                       - Exit virtual environment
echo.
'''
    
    with open("activate.bat", "w") as f:
        f.write(windows_script)
    
    logger.info("Activation scripts created: activate.sh (Unix/Linux/macOS) and activate.bat (Windows)")

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

# Virtual Environment
VIRTUAL_ENV_PATH=./venv
PYTHONPATH=./src
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    logger.info("Environment file created")

def test_installation():
    """Test the installation in virtual environment"""
    try:
        python_path = get_venv_python()
        
        logger.info("Testing installation...")
        
        # Test script
        test_script = '''
import sys
sys.path.append("./src")

# Test imports
try:
    import chromadb
    import transformers
    import streamlit
    import langchain
    import pandas as pd
    import numpy as np
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test ChromaDB
try:
    from chromadb_manager import TenderChromaManager
    print("✅ ChromaDB manager imported successfully")
except Exception as e:
    print(f"❌ ChromaDB manager error: {e}")
    sys.exit(1)

# Test RAG pipeline
try:
    from rag_pipeline import TenderRAGPipeline
    print("✅ RAG pipeline imported successfully")
except Exception as e:
    print(f"❌ RAG pipeline error: {e}")
    sys.exit(1)

print("✅ All tests passed - Installation successful!")
'''
        
        # Write and run test script
        with open("temp_test.py", "w") as f:
            f.write(test_script)
        
        result = subprocess.run([str(python_path), "temp_test.py"], 
                              capture_output=True, text=True)
        
        os.remove("temp_test.py")
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting AMRA Healthcare Tender Pricing System setup with virtual environment...")
    
    # Step 1: Create virtual environment
    if not create_virtual_environment():
        logger.error("Setup failed at virtual environment creation")
        return False
    
    # Step 2: Create directories
    setup_directories()
    
    # Step 3: Create environment file
    create_env_file()
    
    # Step 4: Install requirements in virtual environment
    if not install_requirements_in_venv():
        logger.error("Setup failed at requirements installation")
        return False
    
    # Step 5: Test installation
    if not test_installation():
        logger.error("Setup failed at installation test")
        return False
    
    # Step 6: Initialize database in virtual environment
    if not initialize_database_in_venv():
        logger.error("Setup failed at database initialization")
        return False
    
    # Step 7: Create activation scripts
    create_activation_scripts()
    
    logger.info("Setup completed successfully!")
    
    # Print next steps
    print("\n" + "="*70)
    print("🏥 AMRA Healthcare Tender Pricing System - Setup Complete!")
    print("="*70)
    print("\nVirtual environment setup successful!")
    print("\nTo activate the virtual environment:")
    print("\n📧 On Unix/Linux/macOS:")
    print("   source activate.sh")
    print("   # or manually:")
    print("   source venv/bin/activate")
    print("\n🖥️ On Windows:")
    print("   activate.bat")
    print("   # or manually:")
    print("   venv\\Scripts\\activate.bat")
    print("\nOnce activated, you can:")
    print("1. Start the Streamlit application:")
    print("   streamlit run app.py")
    print("\n2. Run the complete demo:")
    print("   python demo.py")
    print("\n3. Test the system:")
    print("   python test_chromadb.py")
    print("\n4. Check evaluation results:")
    print("   cat evaluation_results/setup_evaluation.md")
    print("\n✅ System is ready for use!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)