# Setup Guide for Agentic-AI Skill Extraction System

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. **Clone the repository (if you haven't already)**

2. **Create a virtual environment**

```bash
# Navigate to your project directory
cd path/to/Agentic-Ai

# Create virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
# source venv/bin/activate
```

3. **Install dependencies in the virtual environment**

```bash
pip install -r requirements.txt
```

3. **Configure your API keys**

Create or edit the file `.streamlit/secrets.toml` and add your API keys:

```toml
# Required API keys
GEMINI_API_KEY = "your_gemini_api_key_here"
ADZUNA_APP_ID = "your_adzuna_app_id_here"
ADZUNA_APP_KEY = "your_adzuna_api_key_here"

# Optional API keys
GITHUB_TOKEN = "your_github_token_here"  # For higher GitHub API rate limits
```

You can get these API keys from:
- Gemini API: https://ai.google.dev/
- Adzuna API: https://developer.adzuna.com/
- GitHub: https://github.com/settings/tokens

## Running the Application

Run the Streamlit app with:

```bash
streamlit run app.py
```

## System Architecture

The system now features:

1. **Hybrid Approach**
   - Base local taxonomy for offline operation
   - Dynamic API-based skill extraction with Gemini
   - Integration with Adzuna for job market data
   - Additional data sources (GitHub, Stack Overflow)

2. **Caching Layer**
   - In-memory caching with TTL (Time To Live)
   - File-based caching for persistence between runs

3. **Persistent Database**
   - SQLite database for storing API responses
   - Ensures deterministic outputs for identical inputs
   - Reduces API costs and improves performance

4. **Progressive Enhancement**
   - Graceful fallbacks when services are unavailable
   - Rule-based extraction as a reliable foundation
   - AI enhancements when available

## Troubleshooting

### Database Issues

If you encounter database-related errors:

```bash
# Reset the persistent database (use with caution)
rm -rf database/api_responses.db
```

### API Rate Limits

If you hit API rate limits:
- The system will automatically use cached responses
- Consider increasing the cache TTL in `tech_utils.py`
- For GitHub API, authenticate with a personal access token

## Next Steps

- Implement GitHub trending technology tracking
- Add Stack Overflow integration for developer tool popularity
- Integrate course platform APIs for learning resources
