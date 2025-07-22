# Mistral AI API Setup Guide

This guide explains how to switch from Hugging Face API to Mistral AI API for better authentication and performance.

## Prerequisites

1. **Get a Mistral AI API Key**
   - Visit [https://console.mistral.ai/](https://console.mistral.ai/)
   - Sign up or log in to your account
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key (it starts with a random string, not a specific prefix)

## Configuration Steps

1. **Create or Update .env File**
   ```bash
   cp .env.example .env
   ```

2. **Configure the Environment Variables**
   Edit your `.env` file:
   ```bash
   # Set to use Mistral AI API
   USE_MISTRAL_API=true
   
   # Add your Mistral API key
   MISTRAL_API_KEY=your_actual_mistral_api_key_here
   
   # You can leave the HuggingFace token empty if using Mistral
   # HUGGINGFACE_API_TOKEN=
   ```

3. **Available Models**
   The system will use these models in order of preference:
   - `mistral-small-latest` (recommended for most use cases)
   - `mistral-tiny` (faster, lower cost)
   - `open-mistral-7b` (open source version)

## Benefits of Using Mistral AI API

- **Better Authentication**: No token format restrictions
- **Direct API Access**: No intermediate services
- **Better Performance**: Optimized for Mistral models
- **More Reliable**: Direct connection to Mistral's infrastructure
- **Cost Effective**: Competitive pricing

## Switching Back to Hugging Face (Optional)

If you want to switch back to Hugging Face API:

1. Set `USE_MISTRAL_API=false` in your `.env` file
2. Ensure you have a valid `HUGGINGFACE_API_TOKEN` 
3. Restart the application

## Model Comparison

| Model | API | Speed | Cost | Quality |
|-------|-----|-------|------|---------|
| mistral-small-latest | Mistral AI | Fast | Low | High |
| mistral-tiny | Mistral AI | Very Fast | Very Low | Good |
| mistralai/Mistral-7B-Instruct-v0.1 | Hugging Face | Medium | Medium | High |

## Troubleshooting

### API Key Issues
- Ensure your API key is correct and active
- Check that you have sufficient credits in your Mistral account
- Verify your account has API access enabled

### Connection Issues
- Check your internet connection
- Verify the API endpoint is accessible
- Look at the application logs for specific error messages

### Model Selection
- The system automatically falls back to smaller models if needed
- Check the application status to see which model is currently active

## Cost Estimation

Mistral AI pricing (approximate):
- **mistral-tiny**: ~$0.14 per 1M input tokens, ~$0.42 per 1M output tokens
- **mistral-small**: ~$0.6 per 1M input tokens, ~$1.8 per 1M output tokens

For typical healthcare queries (500 tokens input, 200 tokens output), costs are very reasonable.

## Support

If you encounter issues:
1. Check the Streamlit app status indicators
2. Review the console logs
3. Verify your API key and credits
4. Try switching models using the fallback system