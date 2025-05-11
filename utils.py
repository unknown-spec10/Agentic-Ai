import os
import json
import time
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from functools import wraps
from pathlib import Path
import google.generativeai as genai
import logging
from dotenv import load_dotenv, find_dotenv
import streamlit as st

# Force reload of environment variables
load_dotenv(find_dotenv(), override=True)

# Configure API key
try:
    # Try to get API key from Streamlit secrets first
    try:
        API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        # Fall back to environment variable
        API_KEY = os.getenv("GEMINI_API_KEY")
        
    # For development/demo purposes only - DO NOT USE IN PRODUCTION
    if not API_KEY:
        logging.warning("⚠️ Using demo API key. For production, set your own API key.")
        API_KEY = "DEMO_KEY_FOR_TESTING_ONLY"
    
    logging.info("API key configured")
except Exception as e:
    logging.error(f"Error configuring API key: {e}")
    API_KEY = "DEMO_KEY_FOR_TESTING_ONLY"
    logging.warning("⚠️ Using fallback demo key")


API_BASE_URL = "https://generativelanguage.googleapis.com/v1"
EMBED_MODEL = "models/embedding-001"
CONTENT_MODEL = "models/gemini-1.0-pro"
MATCH_THRESHOLD = 0.7
CACHE_DIR = Path("cache")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Create a cache directory if it doesn't exist
try:
    CACHE_DIR.mkdir(exist_ok=True)
    logging.info(f"Cache directory ensured at {CACHE_DIR}")
except Exception as e:
    logging.error(f"Failed to create cache directory: {e}")
    # Fall back to a temp directory if we can't create the cache dir
    import tempfile
    CACHE_DIR = Path(tempfile.gettempdir()) / "skill_analysis_cache"
    CACHE_DIR.mkdir(exist_ok=True)
    logging.info(f"Using fallback cache directory at {CACHE_DIR}")

def get_cached_embedding(skill: str) -> Optional[List[float]]:
    """Get embedding from cache if it exists."""
    if not skill:
        return None
        
    # Create a safe filename from the skill
    safe_filename = "".join(c for c in skill if c.isalnum()).lower()
    cache_file = CACHE_DIR / f"emb_{safe_filename}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    return cached_data
        except Exception as e:
            print(f"Error reading cache for {skill}: {e}")
    return None

def save_embedding_to_cache(skill: str, embedding: List[float]):
    """Save embedding to cache."""
    if not skill or not embedding:
        return
        
    try:
        # Create a safe filename from the skill
        safe_filename = "".join(c for c in skill if c.isalnum()).lower()
        cache_file = CACHE_DIR / f"emb_{safe_filename}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    except Exception as e:
        print(f"Error saving cache for {skill}: {e}")

def rate_limit_handler(max_retries=3, initial_delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        retries += 1
                        if retries < max_retries:
                            print(f"Rate limited. Waiting {delay} seconds before retry {retries}/{max_retries}")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            print("Max retries reached. Please try again later.")
                            return None
                    else:
                        raise e
            return None
        return wrapper
    return decorator

def get_gemini_embedding(text: str, max_retries: int = 5) -> List[float]:
    # Check if we're using a demo key and return a deterministic synthetic embedding
    if API_KEY == "DEMO_KEY_FOR_TESTING_ONLY":
        logging.warning("Using synthetic embeddings in demo mode")
        # Generate a deterministic embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).digest()
        np.random.seed(int.from_bytes(text_hash[:4], byteorder='big'))
        embedding = np.random.normal(0, 1, 128)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
        
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
    api_key = API_KEY  # Use the global API_KEY from Streamlit secrets
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "models/embedding-001",
        "content": {
            "parts": [{
                "text": text
            }]
        }
    }
    
    base_wait_time = 1
    
    for retry_count in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}?key={api_key}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            if "embedding" in result:
                return result["embedding"]["values"]
            else:
                raise ValueError("No embedding values found in response")
                
        except Exception as e:
            logging.error(f"Error getting embedding on attempt {retry_count + 1}: {str(e)}")
            if retry_count < max_retries - 1:
                time.sleep(base_wait_time * (2 ** retry_count))
                continue
            raise

def generate_gemini_content(prompt: str, max_retries: int = 5) -> str:
    """
    Generate content using the Gemini API with retry logic.
    
    Args:
        prompt (str): The input prompt for content generation
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Generated content from Gemini API
    """
    # Check if we're using a demo key and return template responses
    if API_KEY == "DEMO_KEY_FOR_TESTING_ONLY":
        logging.warning("Using template responses in demo mode")
        # Simple hard-coded responses for demo purposes
        if "skill gap analysis" in prompt.lower():
            return "Based on the analysis, you have a strong foundation in key technologies. I recommend focusing on cloud architecture and DevOps to enhance your profile."
        elif "recommend job" in prompt.lower():
            return "Based on your skills, consider roles like Senior Developer, Cloud Architect, or DevOps Engineer."
        elif "learning path" in prompt.lower():
            return "To advance your career, focus on: 1) Cloud certifications, 2) CI/CD pipelines, 3) Containerization technologies."
        else:
            return "I've analyzed your request and created a customized response. To get more detailed insights, please use a valid Gemini API key."
            
    base_url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
    api_key = API_KEY  # Use the global API_KEY from Streamlit secrets
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 1,
            "topK": 1,
            "maxOutputTokens": 2048
        }
    }
    
    base_wait_time = 1
    
    for retry_count in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}?key={api_key}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise ValueError("No valid response content found")
                
        except Exception as e:
            logging.error(f"Error on attempt {retry_count + 1}: {str(e)}")
            if retry_count < max_retries - 1:
                time.sleep(base_wait_time * (2 ** retry_count))
                continue
            raise

@rate_limit_handler(max_retries=3, initial_delay=5)
def get_industry_benchmarks(job_title: str) -> Dict[str, Any]:
    """
    Generate industry benchmarks for a given job title using AI (Gemini API).
    """
    if not job_title:
        return {}

    prompt = f"""
    Generate realistic industry benchmarks for the role of {job_title}.
    Return in JSON format:
    {{
      "technical_skills": [
        {{"skill": "Skill Name", "importance": "High/Medium/Low", "typical_level": "Beginner/Intermediate/Expert"}}
      ],
      "soft_skills": ["Skill 1", "Skill 2"],
      "certifications": ["Certification 1", "Certification 2"],
      "experience_levels": {{
        "entry": "0-2 years",
        "mid": "3-5 years",
        "senior": "6+ years"
      }}
    }}
    """

    result = generate_gemini_content(prompt)

    # Validate that AI response contains expected keys
    if "technical_skills" not in result:
        print("Error: Unexpected AI response format, returning default empty structure.")
        return {
            "technical_skills": [],
            "soft_skills": [],
            "certifications": [],
            "experience_levels": {}
        }

    return result


import requests
from typing import List, Dict, Any
from utils import generate_gemini_content  # Ensure this is correctly imported

# Replace these with your actual Adzuna credentials
ADZUNA_APP_ID = "22967e1b"
ADZUNA_APP_KEY = "5a1a65d458e18127af9240fc50e09658"

def get_country_code_and_currency(location: str) -> tuple:
    """
    Get country code and currency symbol based on location.
    
    Args:
        location (str): Country name
        
    Returns:
        tuple: (country_code, currency_symbol, currency_format)
    """
    location_map = {
        "india": ("in", "₹", "indian"),
        "united kingdom": ("gb", "£", "british"),
        "united states": ("us", "$", "us")
    }
    return location_map.get(location.lower(), ("in", "₹", "indian"))

def format_salary(amount: float, currency_format: str) -> str:
    """Format salary based on country's currency format."""
    if currency_format == "indian":
        if amount >= 100000:
            amount_lakhs = amount / 100000
            return f"₹{amount_lakhs:.2f} L"
        return f"₹{amount:,.2f}"
    elif currency_format == "british":
        return f"£{amount:,.0f}"
    else:  # US format
        return f"${amount:,.0f}"

@rate_limit_handler(max_retries=3, initial_delay=5)
def fetch_job_postings(
    job_title: str, 
    skills: List[str], 
    location: str = "india", 
    max_results: int = 10,
    category: str = None,
    salary_min: int = None,
    salary_max: int = None,
    contract_type: str = None,
    sort_by: str = None,
    sort_direction: str = None,
    distance: int = None,
    max_days_old: int = None
) -> Dict[str, Any]:
    """Fetch job postings from Adzuna based on job title, skills, and location."""
    if not job_title or not skills:
        return {"job_matches": []}

    # Get Adzuna credentials from Streamlit secrets
    app_id = st.secrets.get("ADZUNA_APP_ID")
    api_key = st.secrets.get("ADZUNA_APP_KEY")
    
    if not app_id or not api_key:
        logging.warning("Adzuna credentials not found in Streamlit secrets")
        return {"job_matches": []}

    # Get country-specific information
    country_code, currency_symbol, currency_format = get_country_code_and_currency(location)
    api_url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"

    # Construct query parameters
    params = {
        "app_id": app_id,
        "app_key": api_key,
        "what": job_title,
        "results_per_page": max_results,
        "content-type": "application/json"
    }
    
    # Add optional parameters if provided
    if location:
        params["where"] = location
    if category:
        params["category"] = category
    if salary_min:
        params["salary_min"] = salary_min
    if salary_max:
        params["salary_max"] = salary_max
    if contract_type:
        params["contract_type"] = contract_type
    if sort_by:
        params["sort_by"] = sort_by
    if sort_direction:
        params["sort_direction"] = sort_direction
    if distance:
        params["distance"] = distance
    if max_days_old:
        params["max_days_old"] = max_days_old

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        job_data = response.json()

        # Extract job postings
        job_matches = []
        for job in job_data.get("results", []):
            # Format salary information with better handling for missing values
            min_salary = job.get('salary_min')
            max_salary = job.get('salary_max')
            
            if min_salary and max_salary:
                salary_range = f"{format_salary(min_salary, currency_format)} - {format_salary(max_salary, currency_format)}"
            elif min_salary:
                salary_range = f"{format_salary(min_salary, currency_format)}+"
            elif max_salary:
                salary_range = f"Up to {format_salary(max_salary, currency_format)}"
            else:
                salary_range = "Salary not specified"
            
            job_matches.append({
                "title": job.get("title", "N/A"),
                "company": job.get("company", {}).get("display_name", "Unknown"),
                "location": job.get("location", {}).get("display_name", "Unknown"),
                "match_percentage": 85,
                "salary_range": salary_range,
                "description": job.get("description", "No description available."),
                "required_skills": skills,
                "application_link": job.get("redirect_url", "#"),
                "contract_type": job.get("contract_type", "Not specified"),
                "category": job.get("category", {}).get("label", "General"),
                "date_posted": job.get("created", "Unknown date")
            })

        return {"job_matches": job_matches}

    except requests.exceptions.RequestException as e:
        print(f"Adzuna API failure: {e}, falling back to AI-generated job matches for {job_title}")

        # Fallback: Use AI to generate job postings with enhanced parameters
        skills_str = ", ".join(skills)
        location_prompt = f" in {location}" if location else ""
        category_prompt = f" in the {category} category" if category else ""
        salary_prompt = ""
        if salary_min and salary_max:
            salary_prompt = f" with salary range {format_salary(salary_min, currency_format)}-{format_salary(salary_max, currency_format)}"
        elif salary_min:
            salary_prompt = f" with minimum salary {format_salary(salary_min, currency_format)}"
        elif salary_max:
            salary_prompt = f" with maximum salary {format_salary(salary_max, currency_format)}"
        contract_prompt = f" for {contract_type} positions" if contract_type else ""
        
        prompt = f"""
        Generate {max_results} realistic job postings for {job_title}{location_prompt}{category_prompt}{salary_prompt}{contract_prompt} based on these skills: {skills_str}.
        Return in JSON format:
        {{
          "job_matches": [
            {{
              "title": "Job Title",
              "company": "Company Name",
              "location": "{location if location else 'City, State'}",
              "match_percentage": 85,
              "salary_range": "Salary in {currency_symbol} for {location}",
              "description": "Brief job description",
              "required_skills": ["Skill 1", "Skill 2"],
              "application_link": "https://example.com/job/123",
              "contract_type": "{contract_type if contract_type else 'Full-time/Part-time/Contract'}",
              "category": "{category if category else 'General category'}",
              "date_posted": "Recent date"
            }}
          ]
        }}
        """
        result = generate_gemini_content(prompt)
        return result if "job_matches" in result else {"job_matches": []}


@rate_limit_handler(max_retries=3, initial_delay=5)
def recommend_job_roles(skills: List[str], location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """Recommend job roles based on the user's current skills and experience level"""
    if not skills:
        return {"job_recommendations": []}
    
    skills_str = ", ".join(skills)
    
    # Determine experience level
    experience_level = "senior" if experience_years >= 5 else "mid" if experience_years >= 2 else "entry"
    
    prompt = f"""
    Based on these skills: {skills_str}
    For someone with {experience_years} years of experience ({experience_level} level)
    Recommend suitable job roles in {location}.
    
    Return in JSON format:
    {{
      "job_recommendations": [
        {{ 
          "title": "Job Title", 
          "description": "Brief role description", 
          "match_percentage": 85,
          "key_skills_match": ["Skill 1", "Skill 2"],
          "additional_skills_needed": ["Skill 3", "Skill 4"],
          "experience_match": "Good/Fair/Excellent",
          "location": "City, Country",
          "salary_range": "Salary range based on location and experience"
        }}
      ]
    }}
    """
    result = generate_gemini_content(prompt)
    return result if "job_recommendations" in result else {"job_recommendations": []}
    
@rate_limit_handler(max_retries=3, initial_delay=5)
def generate_learning_recommendations(missing_skills: List[str]) -> Dict[str, Any]:
    """Generate personalized learning recommendations based on missing skills."""
    if not missing_skills:
        return {"recommendations": []}
    
    skills_str = ", ".join(missing_skills)
    prompt = f"""
    For these skills: {skills_str}
    Provide specific learning recommendations.
    
    Return in JSON format:
    {{
      "recommendations": [
        {{ 
          "skill": "Skill Name", 
          "course": "Course Title - Provider",
          "url": "https://example.com/course",
          "type": "Course/Book/Tutorial",
          "level": "Beginner/Intermediate/Advanced",
          "duration": "Estimated completion time",
          "cost": "Free/Paid/Price range"
        }}
      ]
    }}
    """
    result = generate_gemini_content(prompt)
    return result if "recommendations" in result else {"recommendations": []}

@rate_limit_handler(max_retries=3, initial_delay=5)
def analyze_market_demand(job_title: str) -> Dict[str, Any]:
    """Analyze market demand for a given job role"""
    if not job_title:
        return {"market_demand": {}}
    
    prompt = f"""
    Provide detailed market demand analysis for {job_title} in JSON format:
    {{
      "market_demand": {{
        "demand_trend": "Growing/Stable/Declining",
        "trend_description": "Brief explanation of the trend",
        "salary_range": "$X - $Y per year",
        "top_industries": ["Industry 1", "Industry 2", "Industry 3"],
        "top_locations": ["Location 1", "Location 2", "Location 3"],
        "key_skills_in_demand": ["Skill 1", "Skill 2", "Skill 3"],
        "future_outlook": "Positive/Neutral/Negative with brief explanation"
      }}
    }}
    """
    result = generate_gemini_content(prompt)
    if "market_demand" not in result:
        return {"market_demand": {}}
    return result

@rate_limit_handler(max_retries=3, initial_delay=5)
def suggest_alternative_careers(resume_skills: List[str], aspired_role: str, match_percentage: float) -> Dict[str, Any]:
    """Suggest alternative career paths based on current skills"""
    if match_percentage >= 70 or not resume_skills:
        return {"alternatives": []}
    
    skills_str = ", ".join(resume_skills)
    prompt = f"""
    For someone with these skills: {skills_str}
    Who wants to be a {aspired_role} (current match: {match_percentage}%)
    Suggest 3 alternative career paths.
    
    Return in JSON format:
    {{
      "alternatives": [
        {{ 
          "title": "Alternative Job Title",
          "fit_reason": "Why this is a good fit based on skills",
          "current_match": 85,
          "transition_difficulty": "Easy/Moderate/Difficult",
          "additional_skills_needed": ["Skill 1", "Skill 2"],
          "average_salary": "$X per year"
        }}
      ]
    }}
    """
    result = generate_gemini_content(prompt)
    return result if "alternatives" in result else {"alternatives": []}

def calculate_skill_similarity(skill1: str, skill2: str) -> float:
    """Calculate similarity between two skills using embeddings."""
    if not skill1 or not skill2:
        return 0.0
    
    try:
        embedding1 = get_gemini_embedding(skill1)
        embedding2 = get_gemini_embedding(skill2)
        
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2)
        
        # Ensure the result is between 0 and 1
        similarity = max(0.0, min(1.0, (similarity + 1) / 2))
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating similarity between '{skill1}' and '{skill2}': {e}")
        return 0.0

def generate_career_path(current_role: str, target_role: str) -> Dict[str, Any]:
    """Generate a recommended career path from current to target role"""
    if not current_role or not target_role:
        return {"career_path": []}
    
    prompt = f"""
    Create a career development path from {current_role} to {target_role}.
    
    Return in JSON format:
    {{
      "career_path": [
        {{
          "stage": 1,
          "role": "Intermediate Role",
          "skills_to_acquire": ["Skill 1", "Skill 2"],
          "estimated_time": "1-2 years",
          "recommended_actions": ["Action 1", "Action 2"]
        }}
      ],
      "total_transition_time": "Estimated total transition time",
      "potential_challenges": ["Challenge 1", "Challenge 2"]
    }}
    """
    result = generate_gemini_content(prompt)
    return result if "career_path" in result else {"career_path": []}

def analyze_interview_readiness(resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
    """Analyze interview readiness based on skill match"""
    if not resume_skills or not job_skills:
        return {"readiness": {}}
    
    resume_skills_str = ", ".join(resume_skills)
    job_skills_str = ", ".join(job_skills)
    
    prompt = f"""
    Analyze interview readiness for a candidate with these skills: {resume_skills_str}
    For a job requiring: {job_skills_str}
    
    Return in JSON format:
    {{
      "readiness": {{
        "overall_score": 75,
        "technical_readiness": 80,
        "key_strengths": ["Strength 1", "Strength 2"],
        "areas_to_prepare": ["Area 1", "Area 2"],
        "recommended_interview_prep": ["Prep 1", "Prep 2"],
        "practice_questions": ["Question 1", "Question 2"]
      }}
    }}
    """
    result = generate_gemini_content(prompt)
    return result if "readiness" in result else {"readiness": {}}

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        raise

def save_to_cache(key: str, data: dict):
    """
    Save data to cache file.
    
    Args:
        key (str): Cache key
        data (dict): Data to cache
    """
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, 'w') as f:
        json.dump(data, f)

def load_from_cache(key: str) -> dict:
    """
    Load data from cache file.
    
    Args:
        key (str): Cache key
        
    Returns:
        dict: Cached data or None if not found
    """
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None
