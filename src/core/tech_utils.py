"""
Dynamic technology and skills extraction utilities for the skill analysis system.
This module provides functions to dynamically extract and identify technology stacks,
common tech skills, and other technology-related information using AI and other data sources.

Implements a hybrid approach with:
1. Base local taxonomy
2. Caching layer for API responses
3. Multiple data sources (Gemini, Adzuna, GitHub, Stack Overflow)
4. Progressive enhancement with graceful fallbacks
"""

import json
import logging
import os
import re
import numpy as np
import faiss
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing utilities
from src.utils.general_utils import generate_gemini_content

# Import cache manager
from src.utils.cache_manager import get_cached, set_cached, delete_cached

# Import persistent database
from src.api.persistent_db import get_api_response, store_api_response, api_response_exists

# Configure logging
# Logging config handled in general_utils.py
logger = logging.getLogger('tech_utils')

# Constants
CACHE_TTL = 60 * 60 * 24 * 7  # 7 days in seconds
GITHUB_API_URL = "https://api.github.com"
STACKOVERFLOW_API_URL = "https://api.stackexchange.com/2.3"
COURSERA_API_URL = "https://api.coursera.org/api/courses.v1"

# Base taxonomy of technical skills (fallback for offline operation)
BASE_SKILL_TAXONOMY = {
    "Programming Languages": [
        "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", 
        "TypeScript", "PHP", "Ruby", "Swift", "Kotlin"
    ],
    "Web Frameworks": [
        "React", "Angular", "Vue.js", "Django", "Flask", "Express",
        "Spring Boot", "Laravel", "ASP.NET", "Ruby on Rails", "Next.js"
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "Jenkins", "GitHub Actions", "CircleCI", "Ansible", "Puppet"
    ],
    "Database Technologies": [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
        "Cassandra", "DynamoDB", "SQLite", "Oracle", "Neo4j"
    ],
    "AI & Machine Learning": [
        "TensorFlow", "PyTorch", "scikit-learn", "NLP", "Computer Vision",
        "Machine Learning", "Deep Learning", "Data Science", "BERT", "Transformers"
    ],
    "Mobile Development": [
        "iOS", "Android", "React Native", "Flutter", "Xamarin",
        "Swift", "Kotlin", "Objective-C", "Mobile UI"
    ],
    "Tools & Methodologies": [
        "Git", "Agile", "Scrum", "JIRA", "CI/CD", "TDD",
        "Microservices", "REST API", "GraphQL", "DevOps"
    ]
}

def fetch_github_trending_tech() -> List[str]:
    """Fetch trending technologies from GitHub API."""
    cache_key = "github_trending_tech"
    cached_data = get_cached(cache_key)
    if cached_data:
        logger.info("Using cached GitHub trending tech data")
        return cached_data
    
    try:
        # GitHub trending repositories API endpoint
        headers = {}
        github_token = st.secrets.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        
        # Get trending repositories for multiple languages
        languages = ["python", "javascript", "java", "go", "typescript"]
        trending_tech = set()
        
        for language in languages:
            url = f"{GITHUB_API_URL}/search/repositories?q=language:{language}&sort=stars&order=desc&per_page=10"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                repos = response.json().get("items", [])
                
                # Extract tech from repository topics and descriptions
                for repo in repos:
                    # Add the language itself
                    trending_tech.add(language.title())
                    
                    # Add repository topics
                    topics = repo.get("topics", [])
                    for topic in topics:
                        if len(topic) > 2 and not topic.startswith("awesome"):
                            trending_tech.add(topic.replace("-", " ").title())
                    
                    # Extract tech terms from description
                    description = repo.get("description", "")
                    if description:
                        tech_stack = rule_based_extract_tech_stack(description)
                        trending_tech.update(tech_stack)
        
        result = list(trending_tech)
        # Cache the result for 24 hours
        set_cached(cache_key, result, ttl=60*60*24)
        return result
    
    except Exception as e:
        logger.error(f"Error fetching GitHub trending tech: {e}")
        return []


def fetch_stackoverflow_popular_tech() -> List[str]:
    """Fetch popular technologies from Stack Overflow API."""
    cache_key = "stackoverflow_popular_tech"
    cached_data = get_cached(cache_key)
    if cached_data:
        logger.info("Using cached Stack Overflow popular tech data")
        return cached_data
    
    try:
        # Stack Overflow popular tags API endpoint
        url = f"{STACKOVERFLOW_API_URL}/tags?pagesize=30&order=desc&sort=popular&site=stackoverflow"
        response = requests.get(url)
        
        if response.status_code == 200:
            tags = response.json().get("items", [])
            popular_tech = []
            
            for tag in tags:
                name = tag.get("name", "")
                if name and not name.startswith(".") and len(name) > 1:
                    # Format tag names for consistency
                    if name in ["c#", "f#"]:
                        popular_tech.append(name.upper())
                    elif name == "html" or name == "css" or name == "php":
                        popular_tech.append(name.upper())
                    elif "." in name or name in ["aws", "gcp"]:
                        popular_tech.append(name.upper())
                    else:
                        popular_tech.append(name.title())
            
            # Cache the result for 3 days
            set_cached(cache_key, popular_tech, ttl=60*60*24*3)
            return popular_tech
    
    except Exception as e:
        logger.error(f"Error fetching Stack Overflow popular tech: {e}")
    
    return []


def fetch_course_platform_tech() -> List[str]:
    """Fetch popular technologies from course platforms."""
    cache_key = "course_platform_tech"
    cached_data = get_cached(cache_key)
    if cached_data:
        logger.info("Using cached course platform tech data")
        return cached_data
    
    try:
        # Combine with Gemini API to get course tech trends
        prompt = """
        What are the most in-demand technical skills according to popular online learning platforms 
        like Coursera, Udemy, and edX? Focus on programming languages, frameworks, and tools.
        
        Respond ONLY with a JSON array of skill names (20-25 skills). Each skill should be properly capitalized.
        Example: ["Python", "JavaScript", "AWS", "Docker", "React"]
        """
        
        result = generate_gemini_content(prompt)
        
        # Try to parse the result as JSON array
        if '[' in result and ']' in result:
            # Extract the JSON array part if there's additional text
            json_part = result[result.find('['):result.rfind(']')+1]
            skills = json.loads(json_part)
            
            if isinstance(skills, list) and len(skills) > 0:
                # Cache the result for 1 week
                set_cached(cache_key, skills, ttl=CACHE_TTL)
                return skills
    
    except Exception as e:
        logger.error(f"Error fetching course platform tech: {e}")
    
    return []


def extract_tech_stack(description: str) -> List[str]:
    """Extract technology stack from job description using progressive enhancement.
    
    Uses the following approach:
    1. Check cache first for similar descriptions
    2. Try AI-based extraction if available
    3. Fall back to rule-based extraction
    4. Cache results for future use
    """
    # Generate a cache key based on the first 100 chars of the description
    description_hash = hash(description[:100])
    cache_key = f"tech_stack_{description_hash}"
    
    # Check cache first
    cached_tech_stack = get_cached(cache_key)
    if cached_tech_stack:
        logger.info("Using cached tech stack extraction")
        return cached_tech_stack
    
    # Check if we can use the AI-based extraction
    from src.core.skill_analysis import GEMINI_API_KEY, model
    
    if GEMINI_API_KEY and GEMINI_API_KEY != "DEMO_KEY_FOR_TESTING_ONLY" and model:
        logger.info("Using AI-based tech stack extraction")
        try:
            tech_stack = ai_extract_tech_stack(description)
            # Cache the results
            set_cached(cache_key, tech_stack, ttl=CACHE_TTL)
            return tech_stack
        except Exception as e:
            logger.error(f"AI-based tech stack extraction failed: {e}. Falling back to rule-based extraction.")
    
    # If AI extraction fails or can't be used, fall back to the rule-based extraction
    logger.info("Using rule-based tech stack extraction")
    tech_stack = rule_based_extract_tech_stack(description)
    
    # Cache the results
    set_cached(cache_key, tech_stack, ttl=CACHE_TTL)
    return tech_stack


def ai_extract_tech_stack(description: str) -> List[str]:
    """Extract technology stack from job description using Gemini API.
    
    This function uses a persistent database to ensure deterministic outputs for the same inputs.
    """
    # Generate a short hash for the description to use as cache/DB lookup
    description_hash = str(hash(description))  # Use hash for consistency
    
    # First, check if this request has a stored response in the persistent database
    if api_response_exists('gemini_tech_stack', description_hash):
        logging.info("Using persisted tech stack extraction result")
        return get_api_response('gemini_tech_stack', description_hash)
    
    # Try rule-based extraction first as a fallback that will always be available
    rule_based_results = rule_based_extract_tech_stack(description)
    
    # If rule-based approach found a good number of technologies, use it directly
    if len(rule_based_results) >= 5:
        # Store in persistent database for future deterministic outputs
        store_api_response('gemini_tech_stack', description_hash, rule_based_results, {
            'timestamp': datetime.now().isoformat(),
            'api': 'rule_based',
            'function': 'rule_based_extract_tech_stack'
        })
        logging.info(f"Using rule-based extraction, found {len(rule_based_results)} technologies")
        return rule_based_results
    
    # If rule-based approach didn't find many technologies, try AI-based approach
    # Create the prompt with explicit instructions to return JSON format
    prompt = f"""
    Extract the technical stack mentioned in the following text. Focus on identifying specific:
    - Programming languages (like Python, Java, JavaScript, C++, etc.)
    - Frameworks (like React, Angular, Django, Spring, etc.)
    - Libraries (like NumPy, Pandas, JUnit, etc.)
    - Databases (like PostgreSQL, MongoDB, MySQL, etc.)
    - Cloud providers and services (like AWS, Azure, GCP, S3, Lambda, etc.)
    - Development tools (like Git, Docker, Kubernetes, etc.)
    - Other technical products or platforms (like Salesforce, SAP, etc.)
    
    Text:
    {description}
    
    IMPORTANT: Respond ONLY with a valid JSON array of technologies. Include only the specific technology names without explanations or additional text.
    Example response: ["Python", "React", "PostgreSQL", "AWS", "Docker"]
    
    Do NOT include any markdown formatting or text outside the JSON array.
    """
    
    try:
        result = generate_gemini_content(prompt)
        
        # Several approaches to extract a potential JSON array
        extracted_tech_stack = None
        
        # Try direct JSON parsing first if the response looks clean
        try:
            if result.strip().startswith('[') and result.strip().endswith(']'):
                extracted_tech_stack = json.loads(result.strip())
        except json.JSONDecodeError:
            pass
            
        # If direct parsing failed, try to extract JSON array from text
        if not extracted_tech_stack and '[' in result and ']' in result:
            try:
                # Find the first '[' and last ']' to extract the JSON array
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                json_part = result[start_idx:end_idx]
                
                # Clean up potential issues in the JSON string
                # Replace single quotes with double quotes for JSON compliance
                json_part = json_part.replace("'", '"')
                # Remove any newlines or extra spaces that might break JSON parsing
                json_part = re.sub(r'\s+', ' ', json_part)
                
                extracted_tech_stack = json.loads(json_part)
            except (json.JSONDecodeError, Exception):
                pass
        
        # If JSON extraction failed, try regex pattern matching for technologies
        if not extracted_tech_stack:
            # Look for items in quotes that might be technologies
            tech_pattern = r'["\']([\w\+\#\-\.]+)["\']'
            matches = re.findall(tech_pattern, result)
            if matches:
                extracted_tech_stack = matches
        
        # Process the extracted tech stack if we found something
        if extracted_tech_stack and isinstance(extracted_tech_stack, list):
            # Normalize tech stack names and remove duplicates
            normalized_stack = []
            for tech in extracted_tech_stack:
                if isinstance(tech, str) and tech.strip():
                    # Convert to string and strip whitespace
                    tech_str = tech.strip()
                    # Only add if not already in the list (case-insensitive)
                    if not any(item.lower() == tech_str.lower() for item in normalized_stack):
                        normalized_stack.append(tech_str)
            
            # Sort and deduplicate the combined results
            combined_results = sorted(list(set(normalized_stack)))
            
            if combined_results:  # Only store if we got actual results
                # Store the combined result in persistent database for future deterministic outputs
                store_api_response('gemini_tech_stack', description_hash, combined_results, {
                    'timestamp': datetime.now().isoformat(),
                    'api': 'gemini',
                    'function': 'ai_extract_tech_stack'
                })
                
                logging.info(f"Successfully extracted {len(combined_results)} technologies using AI")
                return combined_results
        
        # If AI extraction didn't find anything useful or processing failed
        logging.info("AI extraction produced no usable results, falling back to rule-based extraction")
    except Exception as e:
        logging.info(f"AI extraction error: {str(e)}, falling back to rule-based extraction")
    
    # Return and store rule-based results as fallback
    store_api_response('gemini_tech_stack', description_hash, rule_based_results, {
        'timestamp': datetime.now().isoformat(),
        'api': 'rule_based_fallback',
        'function': 'rule_based_extract_tech_stack'
    })
    
    return rule_based_results


def rule_based_extract_tech_stack(description: str) -> List[str]:
    """Extract technology stack from job description using keyword matching."""
    # Common tech keywords that might appear in job descriptions
    common_technologies = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'html', 'css', 'sass', 'jquery', 'bootstrap', 'tailwind',
        'node.js', 'django', 'flask', 'spring', 'express', 'laravel', 'rails', 'asp.net',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'serverless', 'terraform', 'cloudformation',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'cassandra',
        'git', 'github', 'gitlab', 'bitbucket', 'jenkins', 'circleci', 'travisci', 'github actions',
        'react native', 'flutter', 'android', 'ios', 'xamarin',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'jupyter'
    ]
    
    found_tech = []
    description_lower = description.lower()
    
    for tech in common_technologies:
        if tech in description_lower:
            # Capitalize tech names for consistency
            if tech == 'html' or tech == 'css' or tech == 'php':
                found_tech.append(tech.upper())
            elif tech == 'ios':
                found_tech.append('iOS')
            elif '.' in tech or tech in ['aws', 'gcp']:
                found_tech.append(tech.upper())
            else:
                found_tech.append(tech.title())
    
    return sorted(list(set(found_tech)))


def get_common_tech_skills() -> List[str]:
    """Get a list of common tech skills using multiple data sources with caching.
    
    Uses progressive enhancement with multiple data sources:
    1. Check cache first
    2. Try to collect data from multiple sources (GitHub, Stack Overflow, Course platforms)
    3. Combine with AI-generated list if API is available
    4. Fall back to base taxonomy if external sources are unavailable
    5. Cache results for future use
    """
    cache_key = "common_tech_skills"
    
    # Check cache first
    cached_skills = get_cached(cache_key)
    if cached_skills:
        logger.info("Using cached common tech skills")
        return cached_skills
        
    try:
        # Collect skills from multiple sources in parallel
        all_skills = set()
        data_sources = []
        
        # Try to fetch from external APIs in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            github_future = executor.submit(fetch_github_trending_tech)
            stackoverflow_future = executor.submit(fetch_stackoverflow_popular_tech)
            course_future = executor.submit(fetch_course_platform_tech)
            
            # Collect results
            for future in as_completed([github_future, stackoverflow_future, course_future]):
                try:
                    skills = future.result()
                    if skills:
                        all_skills.update(skills)
                        if future == github_future:
                            data_sources.append("GitHub")
                        elif future == stackoverflow_future:
                            data_sources.append("Stack Overflow")
                        elif future == course_future:
                            data_sources.append("Course Platforms")
                except Exception as e:
                    logger.error(f"Error fetching from data source: {e}")
        
        # If we have Gemini API available, enhance with AI-generated list
        from src.core.skill_analysis import GEMINI_API_KEY, model
        if GEMINI_API_KEY and GEMINI_API_KEY != "DEMO_KEY_FOR_TESTING_ONLY" and model:
            try:
                ai_skills = fetch_common_tech_skills()
                if ai_skills:
                    all_skills.update(ai_skills)
                    data_sources.append("Gemini")
            except Exception as e:
                logger.error(f"Failed to fetch AI-generated tech skills: {e}")
                
        # If we couldn't get skills from any external source, fall back to base taxonomy
        if not all_skills:
            logger.warning("No skills found from external sources, using base taxonomy")
            for category, skills in BASE_SKILL_TAXONOMY.items():
                all_skills.update(skills)
            data_sources.append("Base Taxonomy")
                
        # Convert to list and sort
        result = sorted(list(all_skills))
        
        logger.info(f"Fetched common tech skills from: {', '.join(data_sources)}")
        
        # Cache the result
        set_cached(cache_key, result, ttl=CACHE_TTL)
        return result
        
    except Exception as e:
        logger.error(f"Error getting common tech skills: {e}")
        
        # Return flattened base taxonomy as fallback
        fallback_skills = []
        for category, skills in BASE_SKILL_TAXONOMY.items():
            fallback_skills.extend(skills)
        return sorted(list(set(fallback_skills)))


def fetch_common_tech_skills() -> List[str]:
    """Fetch common tech skills using Gemini API with persistent caching.
    
    Uses the persistent database to ensure deterministic outputs for future calls.
    """
    # Check if we have a persisted response
    cache_key = "common_tech_skills_2024"
    if api_response_exists('gemini_common_skills', cache_key):
        logger.info("Using persisted common tech skills")
        return get_api_response('gemini_common_skills', cache_key)
    
    # Generate the prompt for Gemini API
    prompt = """
    Generate a list of the 100 most common technical skills in the software industry in 2023-2024.
    Include programming languages, frameworks, libraries, databases, cloud services, and tools.
    
    Respond ONLY with a valid JSON array of skill names. 
    Example response: ["Python", "JavaScript", "AWS", "Docker", "Kubernetes"]
    """
    
    try:
        result = generate_gemini_content(prompt)
        
        # Try to parse the result as JSON array
        if '[' in result and ']' in result:
            # Extract the JSON array part if there's additional text
            json_part = result[result.find('['):result.rfind(']')+1]
            skills = json.loads(json_part)
            
            if isinstance(skills, list):
                # Normalize skill names
                normalized_skills = [skill.strip() for skill in skills if skill.strip()]
                normalized_skills = sorted(list(set(normalized_skills)))
                
                # Store in persistent database for future deterministic outputs
                store_api_response('gemini_common_skills', cache_key, normalized_skills, {
                    'timestamp': datetime.now().isoformat(),
                    'api': 'gemini',
                    'function': 'fetch_common_tech_skills'
                })
                
                return normalized_skills
            else:
                logger.warning("Unexpected JSON structure in Gemini response for common skills")
        else:
            logger.warning("AI response does not contain valid JSON array for common skills")
    except Exception as e:
        logger.error(f"Error fetching common tech skills: {e}")
    
    # Return a default list of common tech skills if the API call fails
    fallback_skills = []
    for category, skills in BASE_SKILL_TAXONOMY.items():
        fallback_skills.extend(skills)
    return sorted(list(set(fallback_skills)))


def fetch_skills_for_role(role: str, location: str = "any") -> List[str]:
    """Fetch skills specific to a job role using Adzuna API or Gemini with persistent database.
    
    This function ensures deterministic outputs for the same role inputs by storing
    successful API responses in a persistent database.
    """
    # Create a cache key based on role and location
    cache_key = f"{role.lower().replace(' ', '_')}_{location.lower().replace(' ', '_')}"
    
    # First check if we have this in the persistent database
    if api_response_exists('role_skills', cache_key):
        logger.info(f"Using persisted skills for role: {role}")
        return get_api_response('role_skills', cache_key)
        
    try:
        # Try to get skills from Adzuna first
        app_id = st.secrets.get("ADZUNA_APP_ID")
        api_key = st.secrets.get("ADZUNA_APP_KEY")
        
        if app_id and api_key:
            # Attempt to get skills from Adzuna API
            skills = _fetch_skills_from_adzuna(role, location, app_id, api_key)
            if skills:
                # Store successful response in persistent database
                store_api_response('role_skills', cache_key, skills, {
                    'source': 'adzuna',
                    'timestamp': datetime.now().isoformat(),
                    'role': role,
                    'location': location
                })
                return skills
        
        # Use Gemini to generate role-specific skills
        prompt = f"""
        Generate a list of the most important technical skills for the job role of "{role}" 
        {f"in {location}" if location and location.lower() != "any" else ""}.
        
        Respond ONLY with a JSON array of skill names (15-20 skills). Each skill should be properly capitalized.
        Example: ["Python", "JavaScript", "AWS", "Docker", "React"]
        """
        
        result = generate_gemini_content(prompt)
        
        # Try to parse the result as JSON array
        if '[' in result and ']' in result:
            # Extract the JSON array part if there's additional text
            json_part = result[result.find('['):result.rfind(']')+1]
            skills = json.loads(json_part)
            
            if isinstance(skills, list) and len(skills) > 0:
                # Normalize and sort skills
                normalized_skills = [skill.strip() for skill in skills if skill.strip()]
                normalized_skills = sorted(list(set(normalized_skills)))
                
                # Store in persistent database for future deterministic outputs
                store_api_response('role_skills', cache_key, normalized_skills, {
                    'source': 'gemini',
                    'timestamp': datetime.now().isoformat(),
                    'role': role,
                    'location': location
                })
                
                return normalized_skills
    except Exception as e:
        logger.error(f"Error fetching skills for role {role}: {e}")
    
    # Fall back to common skills for the industry if specific role skills can't be determined
    industry_skills = _get_industry_skills_for_role(role)
    if industry_skills:
        return industry_skills
    
    # Ultimate fallback is to return common tech skills
    return get_common_tech_skills()[:20]  # Return top 20 common skills


def _fetch_skills_from_adzuna(role: str, location: str, app_id: str, api_key: str) -> List[str]:
    """Helper function to fetch skills from Adzuna API."""
    try:
        # Prepare the API query
        encoded_role = requests.utils.quote(role)
        base_url = "https://api.adzuna.com/v1/api/jobs"
        
        # Determine location parameter
        country = "gb"  # Default to GB
        area = ""  # No specific area by default
        
        if location.lower() != "any":
            # Map common countries (could be expanded)
            if "united states" in location.lower() or "us" == location.lower():
                country = "us"
            elif "uk" in location.lower() or "united kingdom" in location.lower():
                country = "gb"
            elif "canada" in location.lower():
                country = "ca"
            elif "australia" in location.lower():
                country = "au"
            # Add specific area if needed - this varies by country format
        
        # Build the URL
        url = f"{base_url}/{country}/search/1?app_id={app_id}&app_key={api_key}&results_per_page=10&what={encoded_role}"
        if area:
            url += f"&where={requests.utils.quote(area)}"
            
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            # Extract skills from job descriptions
            all_skills = set()
            for job in results:
                description = job.get("description", "")
                if description:
                    # Extract skills from each job description
                    job_skills = extract_tech_stack(description)
                    all_skills.update(job_skills)
            
            # Return as sorted list
            return sorted(list(all_skills))
            
    except Exception as e:
        logger.error(f"Error fetching from Adzuna API: {e}")
    
    return []


def _get_industry_skills_for_role(role: str) -> List[str]:
    """Get common skills for an industry based on the role."""
    # Map roles to industries/categories from our base taxonomy
    role_lower = role.lower()
    
    # Simple mapping of roles to skill categories
    if any(term in role_lower for term in ["frontend", "front-end", "front end", "ui", "ux", "web developer"]):
        return BASE_SKILL_TAXONOMY.get("Web Frameworks", [])
    
    elif any(term in role_lower for term in ["backend", "back-end", "back end", "api", "server"]):
        return BASE_SKILL_TAXONOMY.get("Programming Languages", [])
    
    elif any(term in role_lower for term in ["devops", "sre", "reliability", "infrastructure"]):
        return BASE_SKILL_TAXONOMY.get("Cloud & DevOps", [])
    
    elif any(term in role_lower for term in ["data", "analyst", "analytics", "bi", "intelligence"]):
        return BASE_SKILL_TAXONOMY.get("Database Technologies", [])
    
    elif any(term in role_lower for term in ["ml", "ai", "machine learning", "data science", "scientist"]):
        return BASE_SKILL_TAXONOMY.get("AI & Machine Learning", [])
    
    elif any(term in role_lower for term in ["mobile", "ios", "android", "app"]):
        return BASE_SKILL_TAXONOMY.get("Mobile Development", [])
    
    # Combine multiple categories for general roles
    elif any(term in role_lower for term in ["full stack", "fullstack", "engineer", "developer"]):
        skills = []
        for category in ["Programming Languages", "Web Frameworks", "Database Technologies"]:
            skills.extend(BASE_SKILL_TAXONOMY.get(category, []))
        return sorted(list(set(skills)))
    
    # Default: return a mix of skills from all categories
    else:
        all_skills = []
        for category, category_skills in BASE_SKILL_TAXONOMY.items():
            all_skills.extend(category_skills[:3])  # Take top 3 from each category
        return sorted(list(set(all_skills)))


def get_skill_taxonomy() -> Dict[str, Any]:
    """Get a comprehensive taxonomy of technical skills."""
    try:
        # Try to use Gemini to create a taxonomy
        prompt = """
        Generate a comprehensive taxonomy of technical skills organized by categories.
        Include the following categories:
        - Programming Languages
        - Web Frameworks
        - Backend Technologies
        - Frontend Technologies
        - Database Systems
        - Cloud Platforms
        - DevOps Tools
        - AI/ML Technologies
        - Mobile Development
        - Security Technologies
        - Data Technologies
        - Software Methodologies
        
        For each category, provide a list of the most relevant skills.
        
        Respond ONLY with a valid JSON object where each key is a category and each value is an array of skills.
        Example:
        {
          "Programming Languages": ["Python", "JavaScript", "Java"],
          "Web Frameworks": ["React", "Angular", "Vue.js"]
        }
        """
        
        result = generate_gemini_content(prompt)
        
        # Try to parse the result as JSON
        if '{' in result and '}' in result:
            # Extract the JSON part if there's additional text
            json_part = result[result.find('{'):result.rfind('}')+1]
            taxonomy = json.loads(json_part)
            
            if isinstance(taxonomy, dict) and len(taxonomy) > 0:
                return taxonomy
    except Exception as e:
        logging.error(f"Error getting skill taxonomy: {e}")
    
    # Fall back to a basic taxonomy
    return {
        "Programming Languages": ["Python", "JavaScript", "Java", "C++", "Go", "TypeScript"],
        "Web Frameworks": ["React", "Angular", "Vue.js", "Django", "Flask", "Express"],
        "Cloud Platforms": ["AWS", "Azure", "GCP", "Docker", "Kubernetes"],
        "Database Systems": ["SQL", "MongoDB", "PostgreSQL", "Redis", "Elasticsearch"],
        "DevOps Tools": ["Git", "Jenkins", "GitHub Actions", "Terraform", "Ansible"],
        "AI/ML Technologies": ["TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision"]
    }

