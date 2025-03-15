import requests
import json
import os
import numpy as np
import time
from typing import List, Dict, Any, Optional
import streamlit as st

# Constants
API_KEY = st.secrets["GEMINI_API_KEY"]
MATCH_THRESHOLD = 0.15  # Define a threshold for matching skills
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
EMBED_MODEL = "models/embedding-001"
CONTENT_MODEL = "gemini-1.5-pro-latest"

def get_api_key():
    return st.secrets.get("GEMINI_API_KEY", "")

def get_adzuna_credentials():
    return (
        st.secrets.get("ADZUNA_APP_ID", ""),
        st.secrets.get("ADZUNA_APP_KEY", "")
    )

def get_gemini_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text using Gemini API."""
    if not text.strip():
        return []
    
    api_url = f"{API_BASE_URL}/models/embedding-001:embedContent"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key":  get_api_key()
    }
    
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }

    response = None  # ✅ Initialize response to avoid UnboundLocalError

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get("embedding", {}).get("values", [])
    
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error in get_gemini_embedding: {e}")
        
        if response and response.status_code == 429:  # ✅ Ensure response exists before accessing status_code
            retry_after = int(response.headers.get('Retry-After', 1))
            print(f"Rate limited. Retrying after {retry_after} seconds")
            time.sleep(retry_after)
            return get_gemini_embedding(text)  # Retry once
        
        return []


def generate_gemini_content(prompt: str, max_retries=5) -> Dict[str, Any]:
    """Generate content using Gemini API with exponential backoff for rate limiting."""
    api_url = f"{API_BASE_URL}/models/{CONTENT_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key":  get_api_key()
    }
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    retry_count = 0
    base_wait_time = 1
    
    while retry_count <= max_retries:
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            text_content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            
            # Handle JSON parsing with better error recovery
            try:
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0].strip()
                elif "```" in text_content:
                    json_content = [block for block in text_content.split("```") if block.strip()]
                    if json_content:
                        text_content = json_content[0].strip()
                
                return json.loads(text_content)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {text_content}")
                return {}
                
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            # Handle rate limiting with exponential backoff
            if hasattr(response, 'status_code') and response.status_code == 429:
                wait_time = base_wait_time * (2 ** retry_count)
                print(f"Rate limited. Retrying after {wait_time} seconds (attempt {retry_count+1}/{max_retries+1})")
                time.sleep(wait_time)
                retry_count += 1
            else:
                print(f"Error in generate_gemini_content: {e}")
                return {}
                
    # If we've exhausted all retries
    print("Maximum retries reached. Could not complete API request.")
    return {}

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

def fetch_job_postings(
    job_title: str, 
    skills: List[str], 
    location: str = None, 
    max_results: int = 5,
    category: str = None,
    salary_min: int = None,
    salary_max: int = None,
    contract_type: str = None,
    sort_by: str = None,
    sort_direction: str = None,
    distance: int = None,
    max_days_old: int = None
) -> Dict[str, Any]:
    """
    Fetch job postings from Adzuna based on job title, skills, and location.
    Uses Gemini AI as a fallback if the Adzuna API request fails.
    
    Args:
        job_title (str): The job title to search for
        skills (List[str]): List of skills the candidate has
        location (str, optional): Location to search jobs in (city, state, or country)
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        category (str, optional): Job category filter (e.g., "it-jobs", "engineering-jobs")
        salary_min (int, optional): Minimum salary filter in local currency units
        salary_max (int, optional): Maximum salary filter in local currency units
        contract_type (str, optional): Type of employment ("permanent", "contract", "part_time", "full_time")
        sort_by (str, optional): Sort method ("date", "relevance", "salary")
        sort_direction (str, optional): Sort order ("up" or "down")
        distance (int, optional): Search radius in miles/km from the specified location
        max_days_old (int, optional): Only show jobs posted within the specified number of days
        
    Returns:
        Dict[str, Any]: Dictionary containing job matches
    """
    if not job_title or not skills:
        return {"job_matches": []}

    api_url = f"https://api.adzuna.com/v1/api/jobs/us/search/1"

     # Get Adzuna credentials
    adzuna_id, adzuna_key = get_adzuna_credentials()

    # Construct query parameters
    params = {
        "app_id": adzuna_id,
        "app_key": adzuna_key,
        "what": job_title,  # Search for jobs matching this title
        "results_per_page": max_results,
        "content-type": "application/json"
    }
    
    # Add optional parameters if provided
    if location:
        params["where"] = location
    
    # Add category filtering
    if category:
        params["category"] = category
    
    # Add salary range filtering
    if salary_min:
        params["salary_min"] = salary_min
    if salary_max:
        params["salary_max"] = salary_max
    
    # Add contract type filtering
    if contract_type:
        params["contract_type"] = contract_type
    
    # Add sort options
    if sort_by:
        params["sort_by"] = sort_by
    if sort_direction:
        params["sort_direction"] = sort_direction
        
    # Add distance radius
    if distance:
        params["distance"] = distance
        
    # Add posting recency filter
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
                salary_range = f"${min_salary:,} - ${max_salary:,}"
            elif min_salary:
                salary_range = f"${min_salary:,}+"
            elif max_salary:
                salary_range = f"Up to ${max_salary:,}"
            else:
                salary_range = "Salary not specified"
            
            job_matches.append({
                "title": job.get("title", "N/A"),
                "company": job.get("company", {}).get("display_name", "Unknown"),
                "location": job.get("location", {}).get("display_name", "Unknown"),
                "match_percentage": 85,  # Placeholder, as Adzuna does not provide match % directly
                "salary_range": salary_range,
                "description": job.get("description", "No description available."),
                "required_skills": skills,  # Adzuna does not provide skills, so we return input skills
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
            salary_prompt = f" with salary range ${salary_min:,}-${salary_max:,}"
        elif salary_min:
            salary_prompt = f" with minimum salary ${salary_min:,}"
        elif salary_max:
            salary_prompt = f" with maximum salary ${salary_max:,}"
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
              "salary_range": "$X-$Y",
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


def recommend_job_roles(skills: List[str], num_recommendations: int = 5) -> Dict[str, Any]:
    """Recommend job roles based on the user's current skills"""
    if not skills:
        return {"job_recommendations": []}
    
    skills_str = ", ".join(skills)
    prompt = f"""
    Based on these skills: {skills_str}
    Recommend {num_recommendations} suitable job roles.
    
    Return in JSON format:
    {{
      "job_recommendations": [
        {{ 
          "title": "Job Title", 
          "description": "Brief role description", 
          "match_percentage": 85,
          "key_skills_match": ["Skill 1", "Skill 2"],
          "additional_skills_needed": ["Skill 3", "Skill 4"]
        }}
      ]
    }}
    """
    result = generate_gemini_content(prompt)
    return result if "job_recommendations" in result else {"job_recommendations": []}
    
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
    """Calculate similarity between two skills using embeddings"""
    embedding1 = get_gemini_embedding(skill1)
    embedding2 = get_gemini_embedding(skill2)
    
    if not embedding1 or not embedding2:
        return 1.0  # Maximum distance (no match)
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(vec1 - vec2)
    return distance

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
