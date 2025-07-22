# AMRA Healthcare POC - Deployment Guide

## 🚀 Streamlit Community Cloud Deployment

### Prerequisites
- GitHub account with repository: `https://github.com/umangGG1/AMRA.git`
- Streamlit Community Cloud account (free)
- API keys for Mistral AI and/or HuggingFace

### Step 1: Streamlit Cloud Setup
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `umangGG1/AMRA`
5. Set main file path: `app.py`
6. Set branch: `main`

### Step 2: Environment Variables Configuration
In Streamlit Cloud app settings, add these secrets:

```toml
# .streamlit/secrets.toml format
USE_MISTRAL_API = true
MISTRAL_API_KEY = "your_mistral_api_key_here"
HUGGINGFACE_API_TOKEN = "your_huggingface_token_here"
CHROMA_PERSIST_DIR = "./chroma_db"
MAX_TOKENS = 1024
TEMPERATURE = 0.1
CONTEXT_WINDOW = 4000
LOG_LEVEL = "INFO"
```

### Step 3: Database Initialization
The ChromaDB will be initialized automatically on first run with the included data.

### Step 4: Deploy
Click "Deploy!" and wait for the build to complete.

## 🛠️ Alternative Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)
- **Pros**: Free, easy setup, automatic HTTPS
- **Cons**: Limited resources, public repository required
- **Best for**: POC demonstrations, small teams

### Option 2: Docker Container
```bash
# Build Docker image
docker build -t amra-healthcare .

# Run container
docker run -p 8501:8501 --env-file .env amra-healthcare
```

### Option 3: Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Option 4: Cloud Platforms
- **Heroku**: Add `Procfile` with `web: streamlit run app.py --server.port=$PORT`
- **AWS EC2**: Deploy with reverse proxy (nginx)
- **Google Cloud Run**: Use container deployment
- **DigitalOcean App Platform**: Connect GitHub repository

## 🔧 Environment Configuration

### Required Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `MISTRAL_API_KEY` | Mistral AI API key | `your_api_key_here` |
| `HUGGINGFACE_API_TOKEN` | HuggingFace token | `hf_token_here` |
| `USE_MISTRAL_API` | Use Mistral (true) or HF (false) | `true` |

### Optional Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_PERSIST_DIR` | Database directory | `./chroma_db` |
| `MAX_TOKENS` | LLM max tokens | `1024` |
| `TEMPERATURE` | LLM temperature | `0.1` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 🎯 Post-Deployment Checklist

### Testing Your Deployment
1. **RAG System Test**: Ask "What are typical costs for hospital IT services?"
2. **Pricing System Test**: Enter "Medical equipment maintenance in Bogotá"
3. **Performance Check**: Verify response times <3 seconds
4. **Error Handling**: Test with invalid queries

### Monitoring
- Check Streamlit Cloud logs for errors
- Monitor API usage and rate limits
- Track system performance metrics

### Security
- Never commit `.env` file to repository
- Use environment variables for all secrets
- Enable HTTPS (automatic on Streamlit Cloud)
- Consider IP whitelisting for production

## 🚨 Troubleshooting

### Common Issues

**1. Database Not Loading**
```
Solution: Check if chroma_db folder exists and contains data
```

**2. API Authentication Errors**
```
Solution: Verify API keys in secrets/environment variables
```

**3. Memory Errors on Streamlit Cloud**
```
Solution: Reduce model size or use lighter embeddings
```

**4. Slow Response Times**
```
Solution: Implement caching, optimize embeddings, or upgrade hosting
```

### Getting Help
- Check application logs in Streamlit Cloud dashboard
- Review GitHub repository issues
- Contact support team

## 📱 Access Your Deployed App

Once deployed, your app will be available at:
`https://umanggg1-amra-app-main-hash.streamlit.app/`

Share this URL for POC demonstrations!