import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
import json
import numpy as np
import faiss
import google.generativeai as genai
import atexit
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import sys
import requests
import traceback
from google.generativeai.types import GenerateContentResponse
import argparse
import re
import PyPDF2
from datetime import datetime
import math
import time
from collections import Counter
import streamlit as st

# Import functions from utils to avoid duplication
from src.utils.general_utils import (
    extract_text_from_pdf as utils_extract_text_from_pdf,
    get_country_code_and_currency,
    format_salary,
    get_gemini_embedding,
    generate_gemini_content,
    api_call_with_retry
)

# Import dynamic tech extraction utilities
from src.core.tech_utils import (
    extract_tech_stack,
    get_common_tech_skills,
    fetch_skills_for_role,
    get_skill_taxonomy
)

# Configure logging - handled in general_utils.py
# No need to configure logging here as it's already done in general_utils.py

# Configure Gemini API - use the same API_KEY from utils module
from src.utils.general_utils import API_KEY
GEMINI_API_KEY = API_KEY
logging.info("Using API key from utils module")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize the model with the correct configuration
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]

    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except Exception as e:
    logging.error(f"Error initializing Gemini model: {str(e)}")
    model = None

# Skill aliases for better matching
skill_aliases = {
    'javascript': ['js', 'es6', 'ecmascript', 'node.js', 'nodejs', 'typescript'],
    'python': ['py', 'python3', 'django', 'flask', 'fastapi'],
    'java': ['core java', 'java se', 'spring', 'springboot'],
    'react': ['reactjs', 'react.js', 'react native'],
    'angular': ['angularjs', 'angular2+', 'ng'],
    'node.js': ['nodejs', 'express.js', 'express'],
    'aws': ['amazon web services', 'aws cloud', 'amazon cloud'],
    'docker': ['containerization', 'docker container'],
    'kubernetes': ['k8s', 'kube', 'kubectl'],
    'postgresql': ['postgres', 'psql'],
    'mongodb': ['mongo', 'nosql'],
    'ci/cd': ['cicd', 'continuous integration', 'continuous deployment'],
    'git': ['github', 'gitlab', 'version control']
}

def normalize_skill(skill: str) -> str:
    """Normalize a skill name for comparison."""
    try:
        normalized = skill.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove special characters
        normalized = re.sub(r'\s+', '', normalized)      # Remove whitespace
        return normalized
    except Exception as e:
        logging.error(f"Error normalizing skill: {str(e)}")
        return skill.lower().strip()

def find_skill_match(skill: str, skill_list: List[str]) -> Optional[str]:
    """Find a matching skill in the list."""
    try:
        normalized_skill = normalize_skill(skill)
        
        # Direct match first
        for s in skill_list:
            if normalize_skill(s) == normalized_skill:
                return s
        
        # Check aliases
        for main_skill, aliases in skill_aliases.items():
            if normalized_skill == normalize_skill(main_skill) or \
               any(normalize_skill(alias) == normalized_skill for alias in aliases):
                for s in skill_list:
                    if normalize_skill(s) == normalize_skill(main_skill) or \
                       any(normalize_skill(alias) == normalize_skill(s) for alias in aliases):
                        return s
        
        # Enhanced fuzzy matching for custom skills
        for s in skill_list:
            norm_s = normalize_skill(s)
            # Check if skills are substrings of each other
            if normalized_skill in norm_s or norm_s in normalized_skill:
                return s
            # Check for common variations
            if normalized_skill.replace('js', 'javascript') == norm_s or \
               norm_s.replace('js', 'javascript') == normalized_skill:
                return s
            # Handle .js variations
            if normalized_skill.replace('.js', 'js') == norm_s or \
               norm_s.replace('.js', 'js') == normalized_skill:
                return s
            # Handle hyphenated variations
            if normalized_skill.replace('-', '') == norm_s or \
               norm_s.replace('-', '') == normalized_skill:
                return s
        
        return None
    except Exception as e:
        logging.error(f"Error finding skill match: {str(e)}")
        return None

# Using get_gemini_embedding imported from utils.py

def get_model():
    """Returns a singleton instance of the model with retry logic."""
    global model
    try:
        if model is None:
            model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        logging.error(f"Failed to initialize model: {str(e)}")
        return None

def cleanup_resources():
    """Clean up any resources used during the analysis."""
    try:
        # Clean up any temporary files or resources here
        if os.path.exists("temp_embeddings.npy"):
            os.remove("temp_embeddings.npy")
        logging.info("Resources cleaned up successfully")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

# Register cleanup function with atexit
atexit.register(cleanup_resources)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    # Use the imported function from utils
    return utils_extract_text_from_pdf(file_path)

def extract_skill_level(text: str, skill: str, variations: List[str], overall_experience: float = 0) -> Dict[str, Any]:
    """Extract skill level and frequency information."""
    try:
        text_lower = text.lower()
        
        # For custom skills, create variations based on common patterns
        if not variations:
            skill_lower = skill.lower()
            variations = [
                skill_lower,
                skill_lower.replace(' ', ''),
                skill_lower.replace('-', ''),
                skill_lower.replace('.', ''),
                f"{skill_lower} framework",
                f"{skill_lower} tool",
                f"{skill_lower} technology"
            ]
        
        # Count mentions of the skill and its variations
        frequency = sum(text_lower.count(var.lower()) for var in variations)
        
        # Look for experience indicators
        experience_patterns = {
            'expert': [
                r'\bexpert\s+(?:in\s+|with\s+)?' + skill.lower(),
                r'\b\d{3,}\+?\s+projects?\s+(?:in|with)\s+' + skill.lower(),
                r'\b(?:senior|lead|principal)\s+' + skill.lower(),
                r'\b' + skill.lower() + r'\s+(?:expert|specialist|architect)',
                r'\badvanced\s+certification\s+(?:in|for)\s+' + skill.lower()
            ],
            'advanced': [
                r'\badvanced\s+(?:knowledge\s+of\s+)?' + skill.lower(),
                r'\b(?:extensive|strong)\s+experience\s+(?:in|with)\s+' + skill.lower(),
                r'\b\d{2}\+?\s+projects?\s+(?:in|with)\s+' + skill.lower(),
                r'\bproficient\s+(?:in|with)\s+' + skill.lower()
            ],
            'intermediate': [
                r'\bintermediate\s+(?:knowledge\s+of\s+)?' + skill.lower(),
                r'\bexperience\s+(?:in|with)\s+' + skill.lower(),
                r'\b(?:worked|familiar)\s+(?:with|on)\s+' + skill.lower(),
                r'\bused\s+' + skill.lower() + r'\s+(?:for|in)\s+projects?'
            ],
            'beginner': [
                r'\bbasic\s+(?:knowledge\s+of\s+)?' + skill.lower(),
                r'\bfamiliar\s+(?:with)?\s+' + skill.lower(),
                r'\bexposure\s+to\s+' + skill.lower(),
                r'\blearning\s+' + skill.lower()
            ]
        }
        
        # Find the highest level that matches
        level = 'beginner'
        max_confidence = 0.4  # Default confidence
        
        for level_name, patterns in experience_patterns.items():
            matches = sum(bool(re.search(pattern, text_lower)) for pattern in patterns)
            if matches > 0:
                confidence = min(matches / len(patterns) + 0.2, 1.0)  # Base confidence on match ratio
                if confidence > max_confidence:
                    level = level_name
                    max_confidence = confidence
        
        # Extract years of experience
        years_patterns = [
            rf'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience\s+(?:in|with)\s+)?{skill.lower()}',
            rf'{skill.lower()}\s+(?:experience|expertise)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
            rf'worked\s+(?:with|on)\s+{skill.lower()}\s+for\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)'
        ]
        
        years = 0
        for pattern in years_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    years = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        # If no specific years found, estimate based on level and frequency
        if years == 0:
            if level == 'expert':
                years = min(overall_experience, 5) if overall_experience > 0 else 5
            elif level == 'advanced':
                years = min(overall_experience, 3) if overall_experience > 0 else 3
            elif level == 'intermediate':
                years = min(overall_experience, 2) if overall_experience > 0 else 2
            else:
                years = min(overall_experience, 1) if overall_experience > 0 else 0.5
        
        # Look for certifications
        cert_patterns = [
            r'certified\s+.*?' + skill.lower(),
            r'certification\s+.*?' + skill.lower(),
            r'(?:professional|associate)\s+certification\s+.*?' + skill.lower()
        ]
        has_certification = any(re.search(pattern, text_lower) for pattern in cert_patterns)
        
        # Look for project leadership with more variations
        leadership_patterns = [
            r'(?:led|managed|directed)\s+(?:team|project|initiative)\s+.*?' + skill.lower(),
            r'(?:team\s+lead|project\s+manager|tech\s+lead)\s+.*?' + skill.lower(),
            r'(?:architected|designed)\s+.*?' + skill.lower() + r'\s+solution',
            r'(?:mentored|trained)\s+(?:team|others)\s+(?:in|on)\s+' + skill.lower()
        ]
        has_leadership = any(re.search(pattern, text_lower) for pattern in leadership_patterns)
        
        return {
            "level": level,
            "years": years,
            "frequency": frequency,
            "has_certification": has_certification,
            "has_leadership": has_leadership,
            "confidence": max_confidence,
            "weight": calculate_skill_weight(level, years, frequency, has_certification, has_leadership, max_confidence, overall_experience)
        }
    except Exception as e:
        logging.error(f"Error extracting skill level: {str(e)}")
        return {
            "level": "beginner",
            "years": 0,
            "frequency": 0,
            "has_certification": False,
            "has_leadership": False,
            "confidence": 0.4,
            "weight": 0.4
        }

def calculate_skill_weight(level: str, years: float, frequency: int, has_certification: bool, has_leadership: bool, confidence: float, overall_experience: float = 0) -> float:
    """Calculate a weighted score for a skill based on multiple factors."""
    try:
        level_weights = {
            'expert': 1.0,
            'advanced': 0.8,
            'intermediate': 0.6,
            'beginner': 0.4
        }
        
        weight = level_weights.get(level, 0.4)
        years_weight = 0.4 * (1 - math.exp(-years/5))
        frequency_weight = 0.2 * (1 - math.exp(-frequency/10))
        cert_weight = 0.2 if has_certification else 0
        leadership_weight = 0.2 if has_leadership else 0
        experience_boost = min(0.1 * (overall_experience / 5), 0.2)
        confidence_factor = 0.5 + (0.5 * confidence)
        
        total_weight = (weight + years_weight + frequency_weight + cert_weight + leadership_weight + experience_boost) * confidence_factor
        return round(min(max(total_weight, 0), 1), 2)
    except Exception as e:
        logging.error(f"Error calculating skill weight: {str(e)}")
        return 0.4

def extract_resume_details(text: str, overall_experience: float = 0) -> Dict[str, Any]:
    """Extract skills and other details from resume text using AI/LLM when possible."""
    
    # Check if we can use the AI-based extraction
    if GEMINI_API_KEY and GEMINI_API_KEY != "DEMO_KEY_FOR_TESTING_ONLY" and model:
        logging.info("Using AI-based skill extraction")
        try:
            # Use AI to extract skills
            return ai_extract_resume_details(text, overall_experience)
        except Exception as e:
            logging.error(f"AI-based extraction failed: {e}. Falling back to rule-based extraction.")
    
    # If AI extraction fails or can't be used, fall back to the traditional method
    logging.info("Using rule-based skill extraction")
    return rule_based_extract_resume_details(text, overall_experience)


def ai_extract_resume_details(text: str, overall_experience: float = 0) -> Dict[str, Any]:
    """Extract skills and other details from resume text using AI/LLM."""
    prompt = f"""
    Extract technical skills from the following resume text. Categorize them into:
    - Programming Languages
    - Web Frontend
    - Web Backend
    - Cloud & DevOps
    - Databases
    - AI/ML
    - Mobile Development
    - Methodologies

    For each skill, provide:
    - Skill name
    - Estimated experience level (Beginner, Intermediate, Advanced, Expert)
    - Years of experience if mentioned (should be a number or null if not specified)
    - Weight (a float between 0 and 1 calculated based on skill level and years of experience;
      Expert=0.8-1.0, Advanced=0.6-0.8, Intermediate=0.4-0.6, Beginner=0.2-0.4; 
      adjust within range based on years of experience)

    Consider the person has {overall_experience} years of overall experience when estimating skill levels.
    If years of experience are not mentioned for a specific skill, leave the years field as null.

    Resume text:
    {text}

    Respond only with a valid JSON object. Do not include explanations or formatting outside of the JSON.
    
    Return as JSON:
    {{
        "skills": ["Skill1", "Skill2", ...],
        "skill_details": {{
            "Skill1": {{
                "level": "Skill level",
                "years": years_mentioned_or_null,
                "confidence": confidence_score,
                "weight": calculated_weight
            }},
            ...
        }},
        "categorized_skills": {{
            "category_name": [{{
                "name": "Skill name",
                "details": {{
                    "level": "Skill level",
                    "years": years_mentioned_or_null,
                    "confidence": confidence_score,
                    "weight": calculated_weight
                }}
            }}],
            ...
        }}
    }}
    """
    
    try:
        result = generate_gemini_content(prompt)
        # Try to parse the result as JSON
        if '{' in result and '}' in result:
            # Extract the JSON part if there's additional text
            json_part = result[result.find('{'):result.rfind('}')+1]
            result_dict = json.loads(json_part)
            
            # Add timestamp
            result_dict["extracted_at"] = datetime.now().isoformat()
            
            # Save the result to a file for debugging and reference
            with open('resume_details.json', 'w') as f:
                json.dump(result_dict, f, indent=2)
            logging.info("Resume details successfully saved to resume_details.json")
            
            return result_dict
        else:
            raise ValueError("AI response does not contain valid JSON")
    except Exception as e:
        logging.error(f"Error in AI skill extraction: {e}")
        raise


def rule_based_extract_resume_details(text: str, overall_experience: float = 0) -> Dict[str, Any]:
    """Extract skills using traditional rule-based approach as a fallback."""
    common_skills = {
        'languages': {
            'python': ['python', 'py', 'python3', 'django', 'flask', 'fastapi', 'pytest'],
            'java': ['java', 'j2ee', 'spring', 'springboot', 'hibernate', 'maven', 'gradle'],
            'javascript': ['javascript', 'js', 'es6', 'es2015', 'ecmascript', 'vanilla js', 'jquery'],
            'typescript': ['typescript', 'ts', 'tsc', 'type script'],
            'c++': ['c++', 'cpp', 'c plus plus', 'stl', 'boost'],
            'go': ['golang', 'go', 'gorm', 'gin'],
            'rust': ['rust', 'rustlang', 'cargo'],
            'php': ['php', 'laravel', 'symfony', 'composer'],
            'ruby': ['ruby', 'rails', 'rake', 'gems'],
            'scala': ['scala', 'akka', 'play framework']
        },
        'web_frontend': {
            'react': ['react', 'reactjs', 'react.js', 'react native', 'nextjs', 'gatsby'],
            'angular': ['angular', 'angularjs', 'ng', 'angular cli', 'angular material'],
            'vue': ['vue', 'vuejs', 'vue.js', 'vuex', 'nuxt'],
            'html': ['html', 'html5', 'semantic html', 'web components'],
            'css': ['css', 'css3', 'scss', 'sass', 'less', 'styled components', 'tailwind'],
            'web_components': ['webpack', 'babel', 'vite', 'parcel', 'rollup'],
            'state_management': ['redux', 'mobx', 'recoil', 'zustand', 'context api']
        },
        'web_backend': {
            'node.js': ['node.js', 'nodejs', 'node', 'npm', 'yarn', 'pnpm'],
            'express': ['express', 'expressjs', 'express.js'],
            'django': ['django', 'django rest framework', 'drf'],
            'flask': ['flask', 'flask-restful', 'flask-sqlalchemy'],
            'fastapi': ['fastapi', 'pydantic'],
            'graphql': ['graphql', 'apollo', 'hasura', 'prisma'],
            'rest': ['rest', 'restful', 'openapi', 'swagger']
        },
        'cloud': {
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'ecs', 'eks', 'rds', 'dynamodb', 'cloudfront'],
            'azure': ['azure', 'microsoft azure', 'azure functions', 'cosmos db', 'app service'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform', 'app engine', 'cloud run', 'bigquery'],
            'docker': ['docker', 'containerization', 'dockerfile', 'docker-compose'],
            'kubernetes': ['kubernetes', 'k8s', 'helm', 'rancher', 'openshift'],
            'terraform': ['terraform', 'iac', 'infrastructure as code', 'pulumi']
        },
        'databases': {
            'mysql': ['mysql', 'mariadb', 'innodb'],
            'postgresql': ['postgresql', 'postgres', 'psql', 'postgis'],
            'mongodb': ['mongodb', 'mongo', 'mongoose', 'atlas'],
            'redis': ['redis', 'redis cluster', 'redis sentinel'],
            'elasticsearch': ['elasticsearch', 'elk', 'kibana', 'logstash'],
            'cassandra': ['cassandra', 'scylla'],
            'neo4j': ['neo4j', 'graph database'],
            'dynamodb': ['dynamodb', 'nosql']
        },
        'devops': {
            'git': ['git', 'github', 'gitlab', 'bitbucket', 'git flow'],
            'ci_cd': ['jenkins', 'gitlab ci', 'github actions', 'circle ci', 'travis ci', 'azure pipelines', 'ci/cd', 'cicd'],
            'monitoring': ['prometheus', 'grafana', 'datadog', 'new relic', 'cloudwatch'],
            'logging': ['elk stack', 'splunk', 'logstash', 'fluentd'],
            'security': ['oauth', 'jwt', 'authentication', 'authorization', 'ssl', 'tls']
        },
        'ai_ml': {
            'machine_learning': ['machine learning', 'ml', 'scikit-learn', 'sklearn'],
            'deep_learning': ['deep learning', 'neural networks', 'tensorflow', 'pytorch', 'keras'],
            'nlp': ['nlp', 'natural language processing', 'transformers', 'bert', 'gpt'],
            'computer_vision': ['computer vision', 'opencv', 'image processing'],
            'data_science': ['pandas', 'numpy', 'scipy', 'matplotlib', 'jupyter']
        },
        'mobile': {
            'ios': ['ios', 'swift', 'objective-c', 'xcode', 'cocoapods'],
            'android': ['android', 'kotlin', 'java android', 'android studio', 'gradle'],
            'cross_platform': ['react native', 'flutter', 'ionic', 'xamarin'],
            'mobile_tools': ['firebase', 'realm', 'mobile analytics']
        },
        'methodologies': {
            'agile': ['agile', 'scrum', 'kanban', 'sprint', 'jira'],
            'testing': ['unit testing', 'integration testing', 'e2e', 'jest', 'cypress', 'selenium'],
            'architecture': ['microservices', 'serverless', 'mvc', 'mvvm', 'clean architecture'],
            'design_patterns': ['design patterns', 'solid principles', 'dry', 'dependency injection']
        }
    }
    
    found_skills = []
    text_lower = text.lower()
    skill_details = {}
    
    # Look for skill variations and extract details
    for category, skill_dict in common_skills.items():
        for main_skill, variations in skill_dict.items():
            if any(variation in text_lower for variation in variations):
                skill_name = main_skill.title().replace('_', ' ')
                found_skills.append(skill_name)
                skill_details[skill_name] = extract_skill_level(text, main_skill, variations, overall_experience)
    
    # Special case handling
    special_cases = [
        ('CI/CD', ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment', 'continuous delivery']),
        ('Cloud', ['cloud', 'saas', 'paas', 'iaas', 'cloud native', 'cloud computing']),
        ('System Design', ['system design', 'distributed systems', 'scalability', 'high availability'])
    ]
    
    for skill_name, variations in special_cases:
        if any(term in text_lower for term in variations):
            found_skills.append(skill_name)
            skill_details[skill_name] = extract_skill_level(text, skill_name.lower(), variations, overall_experience)
    
    # Save extracted details with categories and skill details
    categorized_skills = {}
    for category, skill_dict in common_skills.items():
        category_skills = []
        for main_skill, variations in skill_dict.items():
            if any(variation in text_lower for variation in variations):
                skill_name = main_skill.title().replace('_', ' ')
                category_skills.append({
                    "name": skill_name,
                    "details": skill_details[skill_name]
                })
        if category_skills:
            categorized_skills[category] = sorted(category_skills, key=lambda x: (-x["details"]["weight"], x["name"]))
    
    resume_details = {
        "skills": sorted(list(set(found_skills))),
        "skill_details": skill_details,
        "categorized_skills": categorized_skills,
        "extracted_at": datetime.now().isoformat()
    }
    
    with open('resume_details.json', 'w') as f:
        json.dump(resume_details, f, indent=2)
    logging.info("Resume details successfully saved to resume_details.json")
    
    return resume_details

def extract_job_requirements(job_description: str) -> List[str]:
    """Extract required skills from job description using Gemini API."""
    # Check if we can use the AI-based extraction
    if GEMINI_API_KEY and GEMINI_API_KEY != "DEMO_KEY_FOR_TESTING_ONLY" and model:
        logging.info("Using AI-based job requirements extraction")
        try:
            return ai_extract_job_requirements(job_description)
        except Exception as e:
            logging.error(f"AI-based job requirements extraction failed: {e}. Falling back to rule-based extraction.")
    
    # If AI extraction fails or can't be used, fall back to traditional text analysis
    logging.info("Using rule-based job requirements extraction")
    return rule_based_extract_job_requirements(job_description)


def ai_extract_job_requirements(job_description: str) -> List[str]:
    """Extract required skills from job description using Gemini."""
    # First use rule-based extraction as a reliable baseline
    common_technologies = [
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'Go', 'Rust', 'PHP', 'Ruby', 'C#',
        'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring', 'Express',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD', 'DevOps',
        'Git', 'GitHub', 'GitLab', 'Bitbucket', 'Jenkins', 'Travis CI', 'CircleCI',
        'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Oracle', 'Redis', 'Elasticsearch',
        'Linux', 'Windows', 'MacOS', 'Unix', 'Bash', 'PowerShell', 'Shell',
        'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy', 'Machine Learning',
        'AI', 'Data Science', 'Big Data', 'Data Mining', 'Deep Learning', 'NLP', 'Computer Vision',
        'REST', 'GraphQL', 'API', 'Microservices', 'RPC', 'gRPC', 'WebSockets',
        'HTML', 'CSS', 'SASS', 'LESS', 'Bootstrap', 'Tailwind CSS', 'Material UI',
        'Agile', 'Scrum', 'Kanban', 'Waterfall', 'Jira', 'Trello', 'Confluence', 'Notion'
    ]
    
    found_skills = []
    job_description_lower = job_description.lower()
    
    # Look for skills with context indicators of being required
    for skill in common_technologies:
        skill_lower = skill.lower()
        # Check if the skill is directly mentioned in the job description
        if skill_lower in job_description_lower:
            found_skills.append(skill)
            continue
            
        # Check for skills with common prefixes/suffixes
        if f"{skill_lower} experience" in job_description_lower or \
           f"experience in {skill_lower}" in job_description_lower or \
           f"knowledge of {skill_lower}" in job_description_lower or \
           f"understanding of {skill_lower}" in job_description_lower or \
           f"{skill_lower} skills" in job_description_lower:
            found_skills.append(skill)
    
    # Look for skills in requirements section
    requirements_section = re.search(r'requirements?:?(.*?)(?:\n\n|$)', job_description_lower, re.DOTALL)
    if requirements_section:
        section_text = requirements_section.group(1)
        # Extract bullet points or numbered lists
        skills_list = re.findall(r'(?:•|-|\d+\.)\s*([^•\n]+)', section_text)
        for skill_text in skills_list:
            for skill in common_technologies:
                if skill.lower() in skill_text.lower():
                    found_skills.append(skill)
    
    # If we already found skills via rule-based approach, return those
    if found_skills:
        logging.info(f"Extracted {len(found_skills)} skills via rule-based approach")
        return sorted(list(set(found_skills)))
        
    # If rule-based extraction found nothing, try AI-based approach with careful error handling
    prompt = f"""
    Extract the technical skills required in the following job description. Focus on:
    - Programming languages
    - Frameworks and libraries
    - Tools and platforms
    - Technologies and methodologies
    
    Job Description:
    {job_description}
    
    Respond ONLY with a valid JSON object containing the required skills. Do not include any explanations.
    
    Response format:
    {"required_skills": ["Skill1", "Skill2", ...]}
    """
    
    try:
        result = generate_gemini_content(prompt)
        
        # Try to parse the result as JSON
        if '{' in result and '}' in result:
            try:
                # Extract the JSON part if there's additional text
                json_part = result[result.find('{'):result.rfind('}')+1]
                result_dict = json.loads(json_part)
                
                if 'required_skills' in result_dict and isinstance(result_dict['required_skills'], list):
                    ai_skills = result_dict['required_skills']
                    if ai_skills:
                        logging.info(f"Extracted {len(ai_skills)} skills via AI")
                        return sorted(list(set(ai_skills)))
                else:
                    # Try to extract any list from the result as a fallback
                    logging.info("AI response has JSON but missing required_skills field, trying alternate parsing")
                    for key, value in result_dict.items():
                        if isinstance(value, list) and len(value) > 0:
                            logging.info(f"Found alternative skills list under key '{key}'")
                            return sorted(list(set(value)))
            except json.JSONDecodeError:
                logging.info("Could not parse JSON from AI response, falling back to defaults")
        else:
            logging.info("AI response does not contain JSON structure, falling back to defaults")
    except Exception as e:
        logging.info(f"Error in AI skills extraction: {str(e)}")
    
    # Default fallback skills for a generic tech position if nothing else worked
    default_skills = [
        "Python", "JavaScript", "SQL", "Git", "REST API", "HTML", "CSS", "Docker"
    ]
    logging.info(f"Using default skills set for job requirements: {len(default_skills)}")
    return default_skills

def recommend_job_roles(skills: List[str], location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """Generate job role recommendations based on skills and experience using Adzuna API."""
    logging.info(f"Generating job recommendations for {experience_years} years of experience")
    
    try:
        # Get Adzuna credentials from our utility module or Streamlit secrets
        from src.utils.general_utils import ADZUNA_APP_ID, ADZUNA_APP_KEY
        
        app_id = st.secrets.get("ADZUNA_APP_ID", ADZUNA_APP_ID)
        api_key = st.secrets.get("ADZUNA_APP_KEY", ADZUNA_APP_KEY)
        
        # These credentials should now be available from our module
        if not app_id or not api_key:
            logging.info("Using default Adzuna credentials")
            app_id = ADZUNA_APP_ID
            api_key = ADZUNA_APP_KEY
        
        # Determine experience level and title variations
        experience_level = "entry" if experience_years < 2 else "mid" if experience_years < 5 else "senior"
        title_variations = {
            "entry": {
                "prefixes": ["Junior", "Associate", "Entry-Level", "Graduate"],
                "roles": ["Developer", "Engineer", "Analyst", "Consultant"]
            },
            "mid": {
                "prefixes": ["", "Senior", "Lead", "Mid-Level"],
                "roles": ["Developer", "Engineer", "Architect", "Consultant", "Team Lead"]
            },
            "senior": {
                "prefixes": ["Senior", "Lead", "Principal", "Staff", "Expert"],
                "roles": ["Developer", "Engineer", "Architect", "Manager", "Tech Lead", "Consultant"]
            }
        }
        
        base_url = "https://api.adzuna.com/v1/api"
        country_code = "in" if location.lower() == "india" else "gb"
        currency = "₹" if country_code == "in" else "£"
        
        all_jobs = []
        seen_jobs = set()
        
        # Group skills by category for better matching
        skill_categories = {
            'languages': set(skill.lower() for skill in skills if skill.lower() in 
                ['python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust']),
            'frontend': set(skill.lower() for skill in skills if skill.lower() in 
                ['react', 'angular', 'vue', 'html', 'css']),
            'backend': set(skill.lower() for skill in skills if skill.lower() in 
                ['node.js', 'django', 'flask', 'spring']),
            'cloud': set(skill.lower() for skill in skills if skill.lower() in 
                ['aws', 'azure', 'gcp', 'docker', 'kubernetes']),
            'database': set(skill.lower() for skill in skills if skill.lower() in 
                ['sql', 'mongodb', 'postgresql', 'redis'])
        }
        
        # Generate search queries based on skill combinations
        search_queries = []
        
        # Add role-based queries
        for prefix in title_variations[experience_level]["prefixes"]:
            for role in title_variations[experience_level]["roles"]:
                if skill_categories['languages']:
                    for lang in skill_categories['languages']:
                        search_queries.append(f"{prefix} {lang} {role}".strip())
                if skill_categories['frontend']:
                    search_queries.append(f"{prefix} Frontend {role}".strip())
                if skill_categories['backend']:
                    search_queries.append(f"{prefix} Backend {role}".strip())
                if skill_categories['cloud']:
                    search_queries.append(f"{prefix} Cloud {role}".strip())
        
        # Add skill combination queries
        primary_skills = list(skill_categories['languages'])[:2]  # Use top 2 programming languages
        for skill in primary_skills:
            if skill_categories['frontend']:
                search_queries.extend([f"{skill} frontend", f"frontend {skill}"])
            if skill_categories['backend']:
                search_queries.extend([f"{skill} backend", f"backend {skill}"])
            if skill_categories['cloud']:
                search_queries.extend([f"{skill} cloud", f"cloud {skill}"])
        
        # Search for jobs using generated queries
        for query in set(search_queries):  # Remove duplicates
            try:
                params = {
                    "app_id": app_id,
                    "app_key": api_key,
                    "what": query,
                    "where": location,
                    "max_days_old": 60,
                    "sort_by": "date",
                    "full_time": 1,
                    "results_per_page": 50
                }
                
                data = api_call_with_retry(f"{base_url}/jobs/{country_code}/search/1", params)
                if data and 'results' in data:
                    for job in data['results']:
                        job_id = job.get('id')
                        if job_id and job_id not in seen_jobs:
                            all_jobs.append(job)
                            seen_jobs.add(job_id)
            except Exception as e:
                logging.error(f"Error searching with query '{query}': {str(e)}")

        if not all_jobs:
            logging.warning("No jobs found in API response")
            return get_default_recommendations(location, experience_years)
        
        # Process and score jobs
        processed_jobs = {}
        for job in all_jobs:
            title = job.get('title', '').strip()
            description = job.get('description', '').lower()
            salary = job.get('salary_min', 0)
            
            if title and description:
                if title not in processed_jobs:
                    # Calculate skill match score
                    skill_matches = {
                        category: sum(1 for skill in skills if skill.lower() in description) / len(skills)
                        for category, skills in skill_categories.items() if skills
                    }
                    
                    # Calculate category coverage
                    categories_matched = sum(1 for score in skill_matches.values() if score > 0)
                    category_coverage = categories_matched / len(skill_categories) if skill_categories else 0
                    
                    # Calculate experience match
                    exp_keywords = {
                        'entry': ['junior', 'entry', 'graduate', 'trainee'],
                        'mid': ['intermediate', 'mid', 'experienced'],
                        'senior': ['senior', 'lead', 'principal', 'architect']
                    }
                    
                    exp_match = 0.5  # Default match
                    for level, keywords in exp_keywords.items():
                        if any(keyword in title.lower() for keyword in keywords):
                            if level == experience_level:
                                exp_match = 1.0
                            elif abs(list(exp_keywords.keys()).index(level) - 
                                   list(exp_keywords.keys()).index(experience_level)) == 1:
                                exp_match = 0.7
                            break
                    
                    # Calculate tech stack match
                    tech_stack = extract_tech_stack(description)
                    tech_match = len(set(tech_stack) & set(s.lower() for s in skills)) / len(skills) if skills else 0
                    
                    # Calculate final match score
                    skill_score = sum(skill_matches.values()) / len(skill_matches) if skill_matches else 0
                    match_score = (
                        skill_score * 0.4 +  # 40% weight to skill matches
                        category_coverage * 0.2 +  # 20% weight to category coverage
                        exp_match * 0.25 +  # 25% weight to experience match
                        tech_match * 0.15  # 15% weight to tech stack match
                    ) * 100
                    
                    # Extract required skills from description
                    required_skills = []
                    # Get common tech skills using the imported function
                    tech_skills = get_common_tech_skills()
                    for skill in tech_skills:
                        if skill.lower() in description:
                            required_skills.append(skill)
                    
                    # Generate learning recommendations
                    missing_skills = [skill for skill in required_skills if skill not in skills]
                    learning_recommendations = []
                    if missing_skills:
                        for skill in missing_skills[:3]:  # Top 3 missing skills
                            learning_recommendations.append({
                                "skill": skill,
                                "resources": get_learning_resources(skill, experience_level)
                            })
                    
                    processed_jobs[title] = {
                        'title': title,
                        'match_score': round(match_score, 1),
                        'required_skills': required_skills,
                        'missing_skills': missing_skills,
                        'avg_salary': f"{currency}{int(salary):,}" if salary > 0 else "Not specified",
                        'salary_range': f"{currency}{int(salary):,}" if salary > 0 else "Not specified",  # Adding both naming conventions for compatibility
                        'company': job.get('company', {}).get('display_name', 'Unknown'),
                        'location': job.get('location', {}).get('display_name', location),
                        'description': description[:300] + "...",
                        'experience_match': "Excellent" if exp_match > 0.9 else "Good" if exp_match > 0.7 else "Fair",
                        'application_link': job.get('redirect_url', '#'),
                        'tech_stack': tech_stack,
                        'learning_recommendations': learning_recommendations,
                        'category_matches': {k: round(v * 100, 1) for k, v in skill_matches.items()},
                        'remote_work': any(term in description for term in [
                            'remote', 'work from home', 'wfh', 'virtual', 'telecommute'
                        ])
                    }
        
        # Sort jobs by match score and limit to top matches
        recommended_roles = sorted(
            processed_jobs.values(),
            key=lambda x: x['match_score'],
            reverse=True
        )[:10]  # Increased from 5 to 10
        
        # Generate learning paths based on job market analysis
        learning_paths = generate_learning_paths(skills, experience_years)
        
        # Add market insights
        market_insights = {
            "total_matches": len(processed_jobs),
            "avg_match_score": round(sum(job['match_score'] for job in processed_jobs.values()) / len(processed_jobs), 1) if processed_jobs else 0,
            "remote_opportunities": sum(1 for job in processed_jobs.values() if job['remote_work']),
            "skill_demand": dict(Counter(skill for job in processed_jobs.values() for skill in job['required_skills']).most_common(5))
        }
        
        return {
            "recommended_roles": recommended_roles,
            "learning_paths": learning_paths,
            "market_insights": market_insights
        }
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        logging.error(traceback.format_exc())
        return get_default_recommendations(location, experience_years)

def get_learning_resources(skill: str, experience_level: str) -> List[Dict[str, str]]:
    """Generate learning resource recommendations for a skill based on experience level."""
    resources = []
    
    # Basic resource templates
    platforms = {
        'entry': ['Codecademy', 'freeCodeCamp', 'w3schools'],
        'mid': ['Udemy', 'Coursera', 'PluralSight'],
        'senior': ['Coursera', 'edX', 'O\'Reilly']
    }
    
    # Add platform-specific resources
    for platform in platforms[experience_level]:
        resources.append({
            "platform": platform,
            "type": "Course",
            "difficulty": experience_level.title(),
            "url": f"https://www.{platform.lower()}.com/search?q={skill.lower()}"
        })
    
    # Add documentation and community resources
    if skill.lower() in ['python', 'javascript', 'java', 'react', 'angular']:
        resources.append({
            "platform": "Official Docs",
            "type": "Documentation",
            "difficulty": "All Levels",
            "url": f"https://docs.{skill.lower()}.org"
        })
    
    return resources

def generate_learning_paths(current_skills: List[str], experience_years: float) -> List[Dict[str, Any]]:
    """Generate personalized learning paths based on skills and experience."""
    paths = []
    
    # Define skill paths based on experience level
    if experience_years < 2:
        paths.append({
            "path": "Foundation Building",
            "duration": "6 months",
            "courses": [
                "Programming Fundamentals",
                "Data Structures and Algorithms",
                "Version Control with Git"
            ],
            "certification": "Software Development Fundamentals"
        })
    elif experience_years < 5:
        paths.append({
            "path": "Advanced Development",
            "duration": "4 months",
            "courses": [
                "System Design Principles",
                "Advanced Programming Patterns",
                "CI/CD and DevOps Practices"
            ],
            "certification": "Advanced Software Engineering"
        })
    else:
        paths.append({
            "path": "Technical Leadership",
            "duration": "3 months",
            "courses": [
                "Software Architecture",
                "Team Leadership",
                "Project Management"
            ],
            "certification": "Technical Leadership"
        })
    
    # Add specialized paths based on missing common skills
    # Get common tech skills using the imported function
    tech_skills = get_common_tech_skills()
    missing_skills = set(tech_skills) - set(current_skills)
    if "AWS" in missing_skills or "Azure" in missing_skills:
        paths.append({
            "path": "Cloud Engineering",
            "duration": "4 months",
            "courses": [
                "Cloud Architecture Fundamentals",
                "AWS/Azure Services",
                "Cloud Security"
            ],
            "certification": "Cloud Engineer Certificate"
        })
    
    if "React" in missing_skills or "Angular" in missing_skills:
        paths.append({
            "path": "Frontend Development",
            "duration": "3 months",
            "courses": [
                "Modern JavaScript",
                "React/Angular Fundamentals",
                "Web Performance Optimization"
            ],
            "certification": "Frontend Developer Certificate"
        })
    
    return paths

def get_default_market_demand(location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """
    Get default market demand information when API data is not available
    
    Args:
        location: Target job market location
        experience_years: Years of experience
        
    Returns:
        Dictionary with market demand information
    """
    logging.info(f"Getting default market demand for {location} with {experience_years} years of experience")
    
    try:
        # Use the analyze_market_demand function from utils if available
        from src.utils.general_utils import analyze_market_demand
        
        # Determine job title based on experience
        if experience_years < 1:
            job_title = "Junior Software Developer"
        elif experience_years < 3:
            job_title = "Software Developer"
        elif experience_years < 5:
            job_title = "Senior Software Developer"
        else:
            job_title = "Lead Software Developer"
            
        # Try to get real market data
        try:
            return analyze_market_demand(job_title)
        except Exception as e:
            logging.warning(f"Could not get market demand from API: {e}")
            
        # Fallback to default data
        return {
            "trend": "Growing demand for software professionals with AI and cloud skills",
            "top_skills": [
                "Python", "JavaScript", "Cloud Computing", "Machine Learning", 
                "DevOps", "Data Analysis", "Artificial Intelligence"
            ],
            "top_industries": [
                "Information Technology", "FinTech", "Healthcare Tech",
                "E-commerce", "Cybersecurity & InfoSec"
            ],
            "top_locations": [
                "Bangalore", "Mumbai", "Delhi NCR", "Hyderabad", "Pune"
            ] if location.lower() == "india" else
            [
                "London", "Manchester", "Birmingham", "Edinburgh", "Cambridge"
            ] if location.lower() == "united kingdom" else
            [
                "San Francisco", "New York", "Seattle", "Austin", "Boston"
            ]
        }
    except Exception as e:
        logging.error(f"Error getting market demand: {e}")
        return {
            "trend": "Growing demand for tech professionals",
            "top_skills": ["Python", "JavaScript", "Cloud"],
            "top_industries": ["Tech", "Finance", "Healthcare"],
            "top_locations": ["Major Tech Hubs"]
        }

def get_default_recommendations(location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """Return enhanced default job recommendations based on location and experience."""
    is_india = location.lower() == "india"
    currency = "₹" if is_india else "£"
    
    # Adjust titles and salaries based on experience
    if experience_years >= 5:
        titles = ["Senior Software Engineer", "Lead Developer", "Technical Architect"]
        salary_range = f"{currency}18,00,000" if is_india else f"{currency}75,000"
    elif experience_years >= 3:
        titles = ["Software Engineer", "Full Stack Developer", "DevOps Engineer"]
        salary_range = f"{currency}12,00,000" if is_india else f"{currency}55,000"
    else:
        titles = ["Junior Developer", "Associate Engineer", "Software Developer"]
        salary_range = f"{currency}6,00,000" if is_india else f"{currency}35,000"
    
    recommended_roles = [
        {
            "title": titles[0],
            "match_score": 90,
            "required_skills": ["Python", "JavaScript", "SQL"],
            "avg_salary": salary_range,
            "salary_range": salary_range,  # Adding both naming conventions for compatibility
            "company": "Tech Corp",
            "location": "Bangalore" if is_india else "London",
            "description": "Exciting opportunity for a skilled developer...",
            "experience_match": "Excellent"
        },
        {
            "title": titles[1],
            "match_score": 85,
            "required_skills": ["React", "Node.js", "MongoDB"],
            "avg_salary": salary_range,
            "salary_range": salary_range,  # Adding both naming conventions for compatibility
            "company": "Innovation Labs",
            "location": "Mumbai" if is_india else "Manchester",
            "description": "Join our dynamic team of developers...",
            "experience_match": "Good"
        }
    ]
    
    learning_paths = generate_learning_paths([], experience_years)
    
    return {
        "recommended_roles": recommended_roles,
        "learning_paths": learning_paths
    }

def analyze_skill_gaps(resume_details: Dict[str, Any], job_skills: List[str]) -> Dict[str, Any]:
    """
    Analyze skill gaps between resume skills and job requirements.
    
    Args:
        resume_details: Dictionary containing resume details including skills
        job_skills: List of skills required for the job
        
    Returns:
        Dictionary with matching skills, missing skills, and match percentage
    """
    logging.info("Performing detailed skill gap analysis...")
    
    try:
        # Extract resume skills
        resume_skills = resume_details.get("skills", [])
        skill_details = resume_details.get("skill_details", {})
        
        if not resume_skills or not job_skills:
            return {
                "matching_skills": [],
                "missing_skills": job_skills,
                "match_percentage": 0
            }
        
        # Find matching and missing skills
        matching_skills = []
        missing_skills = []
        
        for job_skill in job_skills:
            # Check if skill or an equivalent is in resume skills
            match = find_skill_match(job_skill, resume_skills)
            
            if match:
                matching_skills.append({
                    "name": job_skill,
                    "match": match,
                    "details": skill_details.get(match, {})
                })
            else:
                missing_skills.append(job_skill)
        
        # Calculate match percentage
        if job_skills:
            match_percentage = round((len(matching_skills) / len(job_skills)) * 100)
        else:
            match_percentage = 0
        
        # Boost score based on skill proficiency levels
        proficiency_bonus = 0
        for skill_match in matching_skills:
            details = skill_match.get("details", {})
            level = str(details.get("level", ""))
            
            if level and level.lower() in ["expert", "advanced"]:
                proficiency_bonus += 2
            elif level and level.lower() == "intermediate":
                proficiency_bonus += 1
        
        # Apply proficiency bonus (up to 15%)
        if matching_skills:
            proficiency_adjustment = min(15, proficiency_bonus * 3)
            match_percentage = min(100, match_percentage + proficiency_adjustment)
        
        return {
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "match_percentage": match_percentage
        }
        
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        return {
            "matching_skills": [],
            "missing_skills": job_skills,
            "match_percentage": 0,
            "error": str(e)
        }

def run_skill_gap_analysis(resume_input: str, job_description: str, location: str = "india", overall_experience: float = 0, resume_details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run the complete skill gap analysis."""
    logging.info("Starting skill gap analysis...")
    
    try:
        # Handle resume input
        if os.path.exists(resume_input) and (resume_input.lower().endswith('.pdf') or resume_input.lower().endswith('.txt')):
            logging.info(f"Reading resume from file: {resume_input}")
            if resume_input.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(resume_input)
            else:
                with open(resume_input, 'r', encoding='utf-8') as file:
                    resume_text = file.read()
        else:
            logging.info("Using provided resume text directly")
            resume_text = resume_input
        
        if not resume_text:
            raise ValueError("No resume text could be extracted or provided")
        
        # Use provided resume_details if available, otherwise extract from text
        if resume_details is None:
            logging.info("Extracting resume details...")
            resume_details = extract_resume_details(resume_text, overall_experience)
        else:
            logging.info("Using provided resume details...")
        
        resume_skills = resume_details.get("skills", [])
        
        # Extract job requirements
        logging.info("Extracting job requirements...")
        job_skills = extract_job_requirements(job_description)
        logging.info(f"Found job skills: {job_skills}")
        
        # Extract job title
        logging.info("Extracting job title...")
        job_title = job_description.split("position")[0].strip()
        logging.info(f"Job Title: {job_title}")
        
        # Analyze skill gaps with experience consideration
        logging.info("Analyzing skill gaps...")
        gap_analysis = analyze_skill_gaps(resume_details, job_skills)
        
        # Adjust match percentage based on overall experience
        if overall_experience > 0:
            experience_factor = min(overall_experience / 5, 1)  # Cap at 5 years
            gap_analysis["match_percentage"] = min(
                100,
                gap_analysis["match_percentage"] * (1 + experience_factor * 0.2)  # Up to 20% boost
            )
        
        # Get market demand data
        logging.info("Analyzing market demand...")
        market_demand = get_default_market_demand(location, overall_experience)
        
        # Find job matches
        logging.info("Finding job matches...")
        recommended_roles = recommend_job_roles(resume_skills, location, overall_experience)
        
        # Compile results with experience information
        analysis_results = {
            "job_title": job_title,
            "overall_experience": overall_experience,
            "skill_match": gap_analysis["match_percentage"],
            "missing_skills": gap_analysis["missing_skills"],
            "matching_skills": gap_analysis["matching_skills"],
            "market_demand": market_demand,
            "recommendations": recommended_roles,
            "experience_level": "Expert" if overall_experience >= 5 else
                              "Advanced" if overall_experience >= 3 else
                              "Intermediate" if overall_experience >= 1 else
                              "Beginner"
        }
        
        # Save results
        with open('skill_gap_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        logging.info("Skill gap analysis completed and results saved to skill_gap_analysis.json")
        
        return analysis_results
        
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        logging.info("Resources cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze skill gaps from resume")
    parser.add_argument("resume_path", help="Path to the resume PDF file")
    parser.add_argument("job_description", help="Job description to compare against")
    parser.add_argument("--experience", type=float, default=0, help="Years of experience")
    parser.add_argument("--location", default="india", help="Target job market location")
    
    args = parser.parse_args()
    run_skill_gap_analysis(args.resume_path, args.job_description, args.location, args.experience)
