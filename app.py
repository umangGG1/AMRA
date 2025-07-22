"""
Streamlit Interface for AMRA Healthcare Tender Pricing System
Interactive web interface for querying tender pricing data using RAG
"""

# Fix SQLite3 compatibility for Streamlit Cloud
import sqlite3
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import sys
import os
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append('./src')

from rag_pipeline import EnhancedTenderRAGPipeline
from chromadb_manager import HealthcareChromaManager
from pricing_recommendation_system import PricingRecommendationSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AMRA Healthcare Tender Pricing",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stats-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .context-doc {
        background-color: #fff9e6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ffa500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'pricing_system' not in st.session_state:
    st.session_state.pricing_system = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'auto_initialized' not in st.session_state:
    st.session_state.auto_initialized = False

@st.cache_resource
def initialize_pipeline():
    """Initialize enhanced RAG pipeline with caching"""
    try:
        # Check which API to use
        use_mistral = os.getenv('USE_MISTRAL_API', 'true').lower() == 'true'
        
        if use_mistral:
            # Check Mistral API key
            mistral_key = os.getenv('MISTRAL_API_KEY')
            if not mistral_key:
                st.error("❌ MISTRAL_API_KEY not found in environment variables!")
                st.info("💡 Please add your Mistral API key to the .env file")
                st.info("🔗 Get your API key from: https://console.mistral.ai/")
                return None
            
            # Test Mistral API connection
            try:
                import requests
                test_response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {mistral_key}"},
                    json={
                        "model": "mistral-tiny",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1
                    },
                    timeout=10
                )
                if test_response.status_code == 200:
                    st.success("✅ Connected to Mistral AI API")
                elif test_response.status_code == 401:
                    st.error("❌ Invalid Mistral API key!")
                    return None
                else:
                    st.warning(f"⚠️ Mistral API test failed: {test_response.status_code}")
            except Exception as e:
                st.warning(f"⚠️ Could not test Mistral API: {str(e)}")
        else:
            # Check HuggingFace API token
            api_token = os.getenv('HUGGINGFACE_API_TOKEN')
            if not api_token:
                st.error("❌ HUGGINGFACE_API_TOKEN not found in environment variables!")
                st.info("💡 Please add your Hugging Face API token to the .env file")
                return None
            
            # Validate token format
            if not api_token.startswith('hf_'):
                st.error("❌ Invalid token format! Hugging Face tokens should start with 'hf_'")
                st.info("💡 Please check your token format in the .env file")
                return None
        
            # Test HuggingFace API connection
            try:
                import requests
                auth_response = requests.get("https://huggingface.co/api/whoami", 
                                           headers={"Authorization": f"Bearer {api_token}"}, 
                                           timeout=10)
                
                if auth_response.status_code == 200:
                    logger.info("✅ HuggingFace API authentication successful")
                    user_info = auth_response.json()
                    st.success(f"✅ Connected to Hugging Face as: {user_info.get('name', 'Unknown')}")
                elif auth_response.status_code == 401:
                    logger.error("❌ HuggingFace API token is invalid or expired")
                    st.error("❌ Your Hugging Face API token is invalid or expired!")
                    st.info("💡 Please get a new token from https://huggingface.co/settings/tokens")
                    return None
                else:
                    logger.warning(f"⚠️ HuggingFace API authentication test failed: {auth_response.status_code}")
                    st.warning(f"⚠️ HuggingFace API authentication failed: {auth_response.status_code}")
                    
            except Exception as api_test_error:
                logger.warning(f"⚠️ Could not test HuggingFace API connection: {str(api_test_error)}")
                st.warning(f"⚠️ Could not test HuggingFace API connection: {str(api_test_error)}")
        
        if use_mistral:
            pipeline = EnhancedTenderRAGPipeline(
                model_name="mistral-small-latest",  # Mistral AI model
                chroma_persist_dir="./chroma_db",
                max_tokens=1024,
                temperature=0.1,
                context_window=4000,
                use_api=True,
                use_mistral_api=True,  # Use Mistral API
                mistral_api_key=mistral_key
            )
        else:
            pipeline = EnhancedTenderRAGPipeline(
                model_name="mistralai/Mistral-7B-Instruct-v0.1",  # HuggingFace model
                chroma_persist_dir="./chroma_db",
                max_tokens=1024,
                temperature=0.1,
                context_window=4000,
                use_api=True,
                use_mistral_api=False,  # Use HuggingFace API
                api_token=api_token if not use_mistral else None
            )
        return pipeline
    except Exception as e:
        st.error(f"Error initializing enhanced pipeline: {str(e)}")
        logger.error(f"Pipeline initialization error: {str(e)}")
        return None

@st.cache_resource
def initialize_pricing_system():
    """Initialize pricing recommendation system with caching"""
    try:
        use_mistral = os.getenv('USE_MISTRAL_API', 'true').lower() == 'true'
        
        pricing_system = PricingRecommendationSystem(
            chroma_persist_dir="./chroma_db",
            use_llm=True,  # Enable LLM integration
            use_mistral_api=use_mistral  # Use same API as RAG pipeline
        )
        # Setup system (will use existing collections if available)
        pricing_system.setup_system()
        return pricing_system
    except Exception as e:
        st.error(f"Error initializing pricing system: {str(e)}")
        return None

@st.cache_data
def load_database_stats():
    """Load database statistics with caching"""
    try:
        chroma_manager = HealthcareChromaManager("./chroma_db")
        chroma_manager.create_collections()
        return chroma_manager.get_collection_stats()
    except Exception as e:
        st.error(f"Error loading database stats: {str(e)}")
        return {}

def setup_database():
    """Setup enhanced database if not already initialized"""
    if not st.session_state.db_initialized:
        try:
            with st.spinner("Setting up enhanced database... This may take a few minutes."):
                pipeline = st.session_state.rag_pipeline
                if pipeline:
                    stats = pipeline.setup_enhanced_database("./data/healthcare_enhanced_20250718_181121_streamlined_20250718_182651.json")
                    st.session_state.db_initialized = True
                    st.success("Enhanced database initialized successfully!")
                    return stats
        except Exception as e:
            st.error(f"Error setting up enhanced database: {str(e)}")
            return None
    return load_database_stats()

def auto_initialize_systems():
    """Automatically initialize both systems when the app loads"""
    if not st.session_state.auto_initialized:
        with st.spinner("🚀 Initializing systems... Please wait"):
            # Initialize RAG pipeline
            if not st.session_state.rag_pipeline:
                st.session_state.rag_pipeline = initialize_pipeline()
            
            # Initialize pricing system  
            if not st.session_state.pricing_system:
                st.session_state.pricing_system = initialize_pricing_system()
            
            # Check if database needs setup
            if st.session_state.rag_pipeline and not st.session_state.db_initialized:
                try:
                    stats = load_database_stats()
                    # If stats show collections with documents, mark as initialized
                    if stats and any(info.get('document_count', 0) > 0 for info in stats.values()):
                        st.session_state.db_initialized = True
                    else:
                        # Setup database if needed
                        setup_database()
                except Exception as e:
                    logger.warning(f"Could not verify database status: {e}")
            
            st.session_state.auto_initialized = True
            
            # Show initialization status
            if st.session_state.rag_pipeline and st.session_state.pricing_system:
                st.success("✅ Systems initialized successfully!")
            else:
                st.warning("⚠️ Some systems failed to initialize - check your configuration")

def main():
    """Main application interface"""
    
    # Auto-initialize systems on first load
    auto_initialize_systems()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AMRA Healthcare Tender Pricing System</h1>', unsafe_allow_html=True)
    
    # Show which API is being used
    use_mistral = os.getenv('USE_MISTRAL_API', 'true').lower() == 'true'
    api_name = "Mistral AI" if use_mistral else "Hugging Face"
    st.markdown(f'<p style="text-align: center; color: #666; font-size: 1.1em;">🚀 Enhanced with {api_name} API, Structured Responses & Advanced Analytics</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Show initialization status
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.rag_pipeline:
                st.success("✅ RAG System")
            else:
                st.error("❌ RAG System")
        
        with col2:
            if st.session_state.pricing_system:
                st.success("✅ Pricing System")
            else:
                st.error("❌ Pricing System")
        
        if st.session_state.db_initialized:
            st.success("✅ Database Ready")
        else:
            st.warning("⚠️ Database Loading")
        
        st.divider()
        st.header("🔧 Manual Controls")
        
        # Initialize pipeline
        if st.button("Initialize RAG Pipeline"):
            with st.spinner("Initializing pipeline..."):
                st.session_state.rag_pipeline = initialize_pipeline()
                if st.session_state.rag_pipeline:
                    st.success("Pipeline initialized!")
                else:
                    st.error("Failed to initialize pipeline")
        
        # Initialize pricing system
        if st.button("Initialize Pricing System"):
            with st.spinner("Initializing pricing system..."):
                st.session_state.pricing_system = initialize_pricing_system()
                if st.session_state.pricing_system:
                    st.success("Pricing system initialized!")
                else:
                    st.error("Failed to initialize pricing system")
        
        # Setup database
        if st.session_state.rag_pipeline and st.button("Setup Database"):
            setup_database()
        
        st.divider()
        
        # Query filters
        st.header("🔍 Query Filters")
        
        # Collection selection
        collection_options = [
            "pricing_context",
            "service_similarity", 
            "geographic_pricing",
            "contractor_performance"
        ]
        
        selected_collections = st.multiselect(
            "Select Collections",
            collection_options,
            default=["pricing_context", "service_similarity"]
        )
        
        # Number of results
        n_results = st.slider("Number of Results", 1, 20, 10)
        
        # Price range filter
        price_ranges = ["All", "0-50K", "50K-200K", "200K-1M", "1M-5M", "5M+"]
        selected_price_range = st.selectbox("Price Range", price_ranges)
        
        # Department filter
        departments = ["All", "Bogotá", "Antioquia", "Valle", "Atlántico", "Other"]
        selected_department = st.selectbox("Department", departments)
        
        # Contract status filter
        statuses = ["All", "Active", "Cancelled", "Draft", "Completed"]
        selected_status = st.selectbox("Status", statuses)
        
        st.divider()
        
        # System stats
        if st.session_state.db_initialized:
            stats = load_database_stats()
            if stats:
                st.header("📊 Database Stats")
                for collection, info in stats.items():
                    st.metric(
                        collection.replace("_", " ").title(),
                        f"{info['document_count']:,}"
                    )
        
        # Query history
        st.header("📝 Query History")
        if st.session_state.query_history:
            for i, query in enumerate(st.session_state.query_history[-5:]):
                if st.button(f"Query {i+1}: {query[:30]}...", key=f"history_{i}"):
                    st.session_state.current_query = query
        
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 General Query", "💰 Pricing Recommendations"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Query Interface")
            
            # Query input - clean and simple
            query_text = st.text_area(
                "Enter your query:",
                value=st.session_state.current_query,
                height=120,
                placeholder="Ask about tender pricing, contracts, contractors, or services...\n\nExamples:\n• What should I charge for medical equipment maintenance in Bogotá?\n• Show me winning bid prices for healthcare consulting services\n• What's the competitive price range for pharmaceutical contracts?\n• Compare pricing for medical services across different regions"
            )
            
            # Query buttons
            if st.button("🔍 Search", type="primary", disabled=not st.session_state.rag_pipeline):
                if query_text.strip():
                    process_query(query_text, selected_collections, n_results, 
                                selected_price_range, selected_department, selected_status)
                else:
                    st.warning("Please enter a query")
            
            if st.button("🗑️ Clear", key="clear_query"):
                st.session_state.current_query = ""
                st.rerun()
            
            # Show initialization status
            if not st.session_state.rag_pipeline:
                st.warning("⚠️ Please initialize the RAG pipeline first using the sidebar button.")
                
                # API info
                use_mistral = os.getenv('USE_MISTRAL_API', 'true').lower() == 'true'
                if use_mistral:
                    st.info("💡 **Note:** Using Mistral AI API. Add `MISTRAL_API_KEY=your_key_here` to your `.env` file.")
                    st.info("🔗 Get your API key from: https://console.mistral.ai/")
                else:
                    st.info("💡 **Note:** Using Hugging Face API. Add `HUGGINGFACE_API_TOKEN=your_token_here` to your `.env` file.")
                
                if st.button("🚀 Quick Initialize RAG Pipeline", key="quick_rag_init"):
                    with st.spinner("Initializing enhanced RAG pipeline..."):
                        st.session_state.rag_pipeline = initialize_pipeline()
                        if st.session_state.rag_pipeline:
                            st.success("Enhanced RAG pipeline initialized!")
                            st.rerun()
                        else:
                            st.error("Failed to initialize enhanced RAG pipeline. Check your API token.")
            
            # Helpful tips
            st.divider()
            st.subheader("💡 Query Examples")
            
            # Example queries that work well
            example_queries = [
                "What should I charge for medical equipment maintenance in Bogotá?",
                "Show me winning bid prices for healthcare consulting services",
                "What's the competitive price range for pharmaceutical contracts?",
                "Compare pricing for medical services across different regions",
                "Which contractors are most successful and what do they charge?",
                "What are the typical costs for hospital IT services?",
                "Show me the most expensive healthcare contracts and their details"
            ]
            
            st.info("💡 **Try these example queries:**")
            for i, query in enumerate(example_queries, 1):
                if st.button(f"{i}. {query}", key=f"example_{i}"):
                    st.session_state.current_query = query
                    st.rerun()
        
        with col2:
            st.header("📈 Quick Stats")
            
            # Database overview
            stats = load_database_stats()
            if stats:
                total_docs = sum(info['document_count'] for info in stats.values())
                st.metric("Total Documents", f"{total_docs:,}")
                
                # Show collection breakdown
                st.subheader("📊 Collections")
                for collection, info in stats.items():
                    st.metric(
                        collection.replace("_", " ").title(),
                        f"{info['document_count']:,}"
                    )
                
                # Note about document counts
                st.info("💡 **Note:** Document counts may vary between systems due to different filtering and preprocessing methods.")
                
                # Query performance
                if st.session_state.rag_pipeline:
                    pipeline_stats = st.session_state.rag_pipeline.get_enhanced_stats()
                    query_stats = pipeline_stats.get('query_stats', {})
                    
                    st.metric("Total Queries", query_stats.get('total_queries', 0))
                    
                    avg_time = query_stats.get('avg_response_time', 0)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                    
                    success_rate = query_stats.get('success_rate', 0)
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                else:
                    st.metric("Total Queries", "0")
                    st.metric("Avg Response Time", "N/A")
            else:
                st.metric("Total Documents", "0")
                st.metric("Total Queries", "0")
                st.metric("Avg Response Time", "N/A")
    
    with tab2:
        st.header("💰 Pricing Recommendations")
        
        # Pricing query interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Service Description")
            
            # Service description input - clean and simple
            service_description = st.text_area(
                "Describe the healthcare service you want to price:",
                height=120,
                placeholder="Enter a detailed description of the healthcare service you need pricing for...\n\nExamples:\n• Medical equipment maintenance for hospitals in Bogotá\n• Healthcare IT consulting services for government facilities\n• Pharmaceutical supply and distribution services\n• Clinical laboratory services and analysis\n• Healthcare facility management and support"
            )
            
            # Optional filters (simplified)
            st.subheader("Optional Filters")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                target_dept = st.text_input("Target Department (optional):", 
                                          placeholder="e.g., Bogotá D.C., Antioquia, Valle del Cauca")
                target_dept = target_dept if target_dept.strip() else None
                
                price_range_filter = st.text_input("Expected Price Range (optional):", 
                                                 placeholder="e.g., 50K-200K, 1M-5M")
                price_range_filter = price_range_filter if price_range_filter.strip() else None
            
            with col_b:
                service_cat = st.text_input("Service Category (optional):", 
                                          placeholder="e.g., medical_services, professional_services")
                service_cat = service_cat if service_cat.strip() else None
                
                num_results = st.slider("Number of Similar Contracts", 5, 20, 10)
            
            # Get pricing recommendations button
            if st.button("🔍 Get Pricing Recommendations", type="primary", 
                        disabled=not st.session_state.pricing_system):
                if service_description.strip():
                    get_pricing_recommendations(service_description, target_dept, 
                                              price_range_filter, service_cat, num_results)
                else:
                    st.warning("Please enter a service description")
        
        with col2:
            st.subheader("💡 Pricing Insights")
            
            if st.session_state.pricing_system:
                st.success("✅ Pricing system ready")
                
                # Show some general statistics
                try:
                    stats = load_database_stats()
                    if stats:
                        pricing_docs = stats.get('pricing_context', {}).get('document_count', 0)
                        st.metric("Pricing References", f"{pricing_docs:,}")
                        
                        similarity_docs = stats.get('service_similarity', {}).get('document_count', 0)
                        st.metric("Service Comparisons", f"{similarity_docs:,}")
                        
                        geo_docs = stats.get('geographic_pricing', {}).get('document_count', 0)
                        st.metric("Geographic Data", f"{geo_docs:,}")
                        
                        contractor_docs = stats.get('contractor_performance', {}).get('document_count', 0)
                        st.metric("Contractor Records", f"{contractor_docs:,}")
                        
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
            else:
                st.warning("⚠️ Initialize pricing system first")
                
                if st.button("🚀 Quick Initialize", key="quick_init"):
                    with st.spinner("Initializing..."):
                        st.session_state.pricing_system = initialize_pricing_system()
                        if st.session_state.pricing_system:
                            st.success("Pricing system initialized!")
                            st.rerun()

def get_pricing_recommendations(service_description: str, target_dept: str, 
                              price_range: str, service_cat: str, num_results: int):
    """Get and display pricing recommendations"""
    try:
        with st.spinner("Analyzing pricing data..."):
            recommendations = st.session_state.pricing_system.get_pricing_recommendations(
                query_description=service_description,
                target_department=target_dept,
                price_range=price_range,
                service_category=service_cat,
                n_results=num_results
            )
            
            display_pricing_recommendations(recommendations)
            
    except Exception as e:
        st.error(f"Error getting pricing recommendations: {str(e)}")

def display_pricing_recommendations(recommendations: Dict[str, Any]):
    """Display pricing recommendations in a formatted way"""
    
    st.subheader("🎯 Pricing Recommendations")
    
    # LLM Response (if available)
    if 'llm_response' in recommendations and recommendations['llm_response']:
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Recommendation:**")
        st.write(recommendations['llm_response'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Key metrics
    pricing_analysis = recommendations.get('pricing_analysis', {})
    confidence_assessment = recommendations.get('confidence_assessment', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recommended_price = pricing_analysis.get('recommended_price')
        if recommended_price:
            st.metric("Recommended Price", f"${recommended_price:,.0f}")
        else:
            st.metric("Recommended Price", "No data")
    
    with col2:
        price_range = pricing_analysis.get('price_range', {})
        if price_range.get('min') and price_range.get('max'):
            st.metric("Price Range", f"${price_range['min']:,.0f} - ${price_range['max']:,.0f}")
        else:
            st.metric("Price Range", "No data")
    
    with col3:
        confidence = pricing_analysis.get('confidence', 'unknown')
        st.metric("Confidence Level", confidence.title())
    
    with col4:
        total_contracts = recommendations.get('total_similar_contracts', 0)
        st.metric("Similar Contracts", f"{total_contracts}")
    
    # Confidence assessment
    if confidence_assessment:
        st.subheader("📊 Confidence Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_score = confidence_assessment.get('overall_score', 0)
            st.metric("Overall Score", f"{overall_score:.1%}")
        
        with col2:
            avg_similarity = confidence_assessment.get('average_similarity', 0)
            st.metric("Avg Similarity", f"{avg_similarity:.1%}")
        
        with col3:
            data_completeness = confidence_assessment.get('data_completeness', 0)
            st.metric("Data Completeness", f"{data_completeness:.1%}")
    
    # Similar contracts
    similar_contracts = recommendations.get('similar_contracts', [])
    if similar_contracts:
        st.subheader("📄 Similar Contracts")
        
        for i, contract in enumerate(similar_contracts[:5]):  # Show top 5
            with st.expander(f"Contract {i+1} - Score: {contract['weighted_score']:.3f}"):
                st.write(f"**Collection:** {contract['collection']}")
                st.write(f"**Content:** {contract['document']}")
                
                # Show metadata if available
                metadata = contract.get('metadata', {})
                if metadata:
                    st.write("**Metadata:**")
                    if 'contract_value' in metadata:
                        st.write(f"- Value: ${metadata['contract_value']:,.0f}")
                    if 'entity_department' in metadata:
                        st.write(f"- Department: {metadata['entity_department']}")
                    if 'contract_type' in metadata:
                        st.write(f"- Type: {metadata['contract_type']}")
                    if 'service_category' in metadata:
                        st.write(f"- Category: {metadata['service_category']}")
    
    # Price distribution visualization
    if pricing_analysis.get('distribution'):
        st.subheader("📈 Price Distribution Analysis")
        
        distribution = pricing_analysis['distribution']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistical Summary:**")
            st.write(f"- Mean: ${distribution['mean']:,.0f}")
            st.write(f"- Median: ${distribution['median']:,.0f}")
            st.write(f"- Std Dev: ${distribution['std_dev']:,.0f}")
            st.write(f"- Count: {distribution['count']}")
        
        with col2:
            st.write("**Price Quartiles:**")
            st.write(f"- Q1 (25%): ${distribution['q25']:,.0f}")
            st.write(f"- Q2 (50%): ${distribution['median']:,.0f}")
            st.write(f"- Q3 (75%): ${distribution['q75']:,.0f}")
            st.write(f"- Range: ${distribution['min']:,.0f} - ${distribution['max']:,.0f}")

def process_query(query: str, collections: List[str], n_results: int, 
                 price_range: str, department: str, status: str):
    """Process user query and display results"""
    
    # Build filters
    filters = {}
    if price_range != "All":
        filters['price_range'] = price_range
    if department != "All":
        filters['entity_department'] = department
    if status != "All":
        filters['process_status'] = status
    
    # Execute query
    with st.spinner("Processing query..."):
        try:
            # Check if RAG pipeline is initialized
            if not st.session_state.rag_pipeline:
                st.error("RAG pipeline not initialized. Please initialize it first.")
                return
            
            # Check if database is set up
            if not st.session_state.db_initialized:
                st.warning("Database not fully initialized. Setting up now...")
                setup_database()
            
            result = st.session_state.rag_pipeline.enhanced_query(
                query=query,
                filters=filters if filters else None,
                n_results=n_results
            )
            
            # Add to history
            st.session_state.query_history.append(query)
            
            # Display results
            display_query_results(result)
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.error("Please make sure the RAG pipeline is initialized and the database is set up.")

def display_query_results(result: Dict[str, Any]):
    """Display enhanced query results with structured responses"""
    
    st.subheader("🎯 Enhanced Query Results")
    
    # Query info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Query Type", result['query_type'].replace('_', ' ').title())
    with col2:
        st.metric("Response Time", f"{result['response_time']:.2f}s")
    with col3:
        st.metric("Context Documents", result['num_context_docs'])
    with col4:
        if result.get('cached'):
            st.metric("Cache Hit", "✅")
        else:
            st.metric("Cache Hit", "❌")
    
    # Structured response
    structured_response = result.get('structured_response')
    if structured_response:
        # Display structured pricing information
        if hasattr(structured_response, 'primary_price') and structured_response.primary_price:
            st.subheader("💰 Pricing Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Primary Price", f"${structured_response.primary_price:,.0f}")
            with col2:
                if structured_response.price_range:
                    min_price, max_price = structured_response.price_range
                    st.metric("Price Range", f"${min_price:,.0f} - ${max_price:,.0f}")
            with col3:
                st.metric("Confidence", f"{structured_response.confidence_score:.1%}")
        
        # Display response text
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Response:**")
        st.write(structured_response.response_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display factors affecting price
        if structured_response.factors_affecting_price:
            st.subheader("📊 Factors Affecting Price")
            for factor in structured_response.factors_affecting_price:
                st.write(f"• {factor}")
        
        # Display recommendations
        if structured_response.recommendations:
            st.subheader("💡 Recommendations")
            for rec in structured_response.recommendations:
                st.write(f"• {rec}")
        
        # Display comparable contracts
        if structured_response.comparable_contracts:
            st.subheader("📄 Comparable Contracts")
            for i, contract in enumerate(structured_response.comparable_contracts[:5]):
                with st.expander(f"Contract {i+1} - ${contract.get('value', 'N/A'):,.0f}"):
                    st.write(f"**Department:** {contract.get('department', 'Unknown')}")
                    st.write(f"**Service:** {contract.get('service', 'Unknown')}")
                    st.write(f"**Value:** ${contract.get('value', 0):,.0f}")
                    if contract.get('similarity_score'):
                        st.metric("Similarity", f"{contract['similarity_score']:.3f}")
    else:
        # Fallback for non-structured responses
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.markdown("**Response:**")
        st.write(result.get('response', 'No response available'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Context documents
    if result.get('context_docs'):
        st.subheader("📄 Source Documents")
        
        for i, doc in enumerate(result['context_docs'][:5]):  # Show top 5
            with st.expander(f"Document {i+1} (Collection: {doc.get('collection', 'Unknown')})"):
                st.write(doc.get('content', 'No content available'))
                
                # Metadata
                if doc.get('metadata'):
                    st.subheader("Metadata")
                    metadata_df = pd.DataFrame([doc['metadata']])
                    st.dataframe(metadata_df, use_container_width=True)
                
                if doc.get('distance'):
                    st.metric("Relevance Score", f"{1-doc['distance']:.3f}")
    
    # Visualization for pricing queries
    if result['query_type'] in ['specific_pricing', 'comparative_analysis', 'budget_estimation'] and result.get('context_docs'):
        create_pricing_visualization(result['context_docs'])

def create_pricing_visualization(context_docs: List[Dict]):
    """Create visualizations for pricing data"""
    
    st.subheader("📊 Pricing Visualization")
    
    # Extract pricing data
    pricing_data = []
    for doc in context_docs:
        metadata = doc['metadata']
        if 'contract_value' in metadata and metadata['contract_value']:
            pricing_data.append({
                'value': metadata['contract_value'],
                'department': metadata.get('entity_department', 'Unknown'),
                'contract_type': metadata.get('contract_type', 'Unknown'),
                'collection': doc['collection']
            })
    
    if pricing_data:
        df = pd.DataFrame(pricing_data)
        
        # Price distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df, 
                x='value', 
                title='Price Distribution',
                labels={'value': 'Contract Value ($)', 'count': 'Number of Contracts'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            if len(df['department'].unique()) > 1:
                fig_box = px.box(
                    df, 
                    x='department', 
                    y='value',
                    title='Price by Department'
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Visualization requires data from multiple departments")

# Run the application
if __name__ == "__main__":
    main()