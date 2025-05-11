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
from utils import (
    extract_text_from_pdf as utils_extract_text_from_pdf,
    get_country_code_and_currency,
    format_salary,
    get_gemini_embedding,
    generate_gemini_content
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure Gemini API - use the same API_KEY from utils.py
from utils import API_KEY
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
    """Extract skills and other details from resume text."""
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
    """Extract required skills from job description."""
    common_skills = {
        'languages': {
            'python': ['python', 'py', 'django', 'flask', 'fastapi'],
            'java': ['java', 'j2ee', 'spring', 'springboot', 'hibernate'],
            'javascript': ['javascript', 'js', 'es6', 'ecmascript', 'typescript', 'node.js'],
            'typescript': ['typescript', 'ts', 'angular', 'next.js'],
            'c++': ['c++', 'cpp', 'stl', 'boost'],
            'go': ['golang', 'go'],
            'rust': ['rust', 'cargo']
        },
        'web': {
            'react': ['react', 'reactjs', 'react.js', 'react native', 'redux'],
            'angular': ['angular', 'angularjs', 'ng', 'angular material'],
            'vue': ['vue', 'vuejs', 'vue.js', 'vuex', 'nuxt'],
            'node.js': ['node.js', 'nodejs', 'express', 'nestjs'],
            'django': ['django', 'drf', 'django rest framework'],
            'flask': ['flask', 'flask-restful']
        },
        'cloud': {
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'azure': ['azure', 'microsoft azure', 'azure functions', 'azure devops'],
            'gcp': ['gcp', 'google cloud', 'app engine', 'cloud run'],
            'docker': ['docker', 'containerization', 'docker-compose'],
            'kubernetes': ['kubernetes', 'k8s', 'helm', 'openshift'],
            'terraform': ['terraform', 'iac', 'infrastructure as code']
        },
        'databases': {
            'mysql': ['mysql', 'mariadb', 'sql'],
            'postgresql': ['postgresql', 'postgres', 'psql'],
            'mongodb': ['mongodb', 'mongo', 'mongoose'],
            'redis': ['redis', 'redis cluster'],
            'elasticsearch': ['elasticsearch', 'elk', 'kibana'],
            'dynamodb': ['dynamodb', 'aws dynamodb']
        },
        'tools': {
            'git': ['git', 'github', 'gitlab', 'bitbucket'],
            'ci_cd': ['jenkins', 'gitlab ci', 'github actions', 'circle ci', 'ci/cd', 'continuous integration'],
            'monitoring': ['prometheus', 'grafana', 'datadog', 'new relic'],
            'jira': ['jira', 'confluence', 'agile'],
            'testing': ['unit testing', 'jest', 'pytest', 'selenium', 'cypress']
        },
        'ai_ml': {
            'machine_learning': ['machine learning', 'ml', 'scikit-learn', 'sklearn'],
            'deep_learning': ['deep learning', 'neural networks', 'tensorflow', 'pytorch'],
            'nlp': ['nlp', 'natural language processing', 'transformers'],
            'data_science': ['data science', 'pandas', 'numpy', 'jupyter']
        }
    }
    
    found_skills = []
    job_description_lower = job_description.lower()
    
    # Look for skill variations with context
    for category, skill_dict in common_skills.items():
        for main_skill, variations in skill_dict.items():
            # Check for skill mentions with surrounding context
            for variation in variations:
                # Look for required/mandatory/essential skills
                required_patterns = [
                    f"required.*?{variation}",
                    f"must have.*?{variation}",
                    f"essential.*?{variation}",
                    f"mandatory.*?{variation}",
                    f"proficient in.*?{variation}",
                    f"experience with.*?{variation}",
                    f"knowledge of.*?{variation}",
                    f"expertise in.*?{variation}",
                    f"skills:.*?{variation}",
                    f"requirements:.*?{variation}"
                ]
                
                if any(re.search(pattern, job_description_lower) for pattern in required_patterns):
                    found_skills.append(main_skill.title())
                    break  # Break after finding first match for this skill
    
    # Special case handling for common combinations
    if any(term in job_description_lower for term in ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment']):
        found_skills.append('CI/CD')
    if any(term in job_description_lower for term in ['cloud', 'saas', 'paas', 'iaas', 'cloud native']):
        found_skills.append('Cloud')
    if any(term in job_description_lower for term in ['system design', 'distributed systems', 'scalability']):
        found_skills.append('System Design')
    
    # Add skills mentioned in requirements section
    requirements_section = re.search(r'requirements?:?(.*?)(?:\n\n|$)', job_description_lower, re.DOTALL)
    if requirements_section:
        section_text = requirements_section.group(1)
        # Look for bullet points or numbered lists
        skills_list = re.findall(r'(?:•|-|\d+\.)\s*([^•\n]+)', section_text)
        for skill_text in skills_list:
            for category, skill_dict in common_skills.items():
                for main_skill, variations in skill_dict.items():
                    if any(variation in skill_text.lower() for variation in variations):
                        found_skills.append(main_skill.title())
    
    return sorted(list(set(found_skills)))

def analyze_skill_gaps(resume_details: Union[Dict[str, Any], List[str]], job_skills: List[str]) -> Dict[str, Any]:
    """
    Analyze gaps between resume skills and job requirements.
    Args:
        resume_details: Dictionary containing resume details or list of skills
        job_skills: List of required skills from job description
    Returns:
        Dictionary containing matching and missing skills with analysis
    """
    logging.info("Starting skill gap analysis")
    
    try:
        # Convert job skills to lowercase for comparison
        job_skills = [skill.lower() for skill in job_skills]
        
        # Extract skills from resume
        if isinstance(resume_details, dict):
            resume_skills = resume_details.get('skills', [])
            skill_details = resume_details.get('skill_details', {})
            if isinstance(resume_skills, str):
                resume_skills = [resume_skills]
        else:
            resume_skills = resume_details
            skill_details = {}
        
        resume_skills = [skill.lower() for skill in resume_skills]
        
        # Get market demanded skills for the role
        market_skills = [
            "python", "javascript", "java", "sql", "aws",
            "react", "angular", "node.js", "docker", "kubernetes",
            "devops", "ci/cd", "agile", "cloud", "system design"
        ]
        
        # Find matching and missing skills
        matching_skills = []
        missing_skills = []
        
        # Combined required skills (job requirements + market demands)
        all_required_skills = list(set(job_skills + [skill.lower() for skill in market_skills]))
        
        for required_skill in all_required_skills:
            skill_found = False
            
            # Check for exact matches first
            if required_skill in resume_skills:
                # Get the original case version of the skill
                original_skill = next((s for s in resume_details.get('skills', []) if s.lower() == required_skill), required_skill.title())
                skill_info = {
                    'name': original_skill,
                    'details': skill_details.get(original_skill, {
                        'level': 'intermediate',
                        'years': 0,
                        'has_certification': False,
                        'has_leadership': False
                    }),
                    'source': 'job' if required_skill in job_skills else 'market'
                }
                matching_skills.append(skill_info)
                skill_found = True
            
            # Check for variations and synonyms if not found
            if not skill_found:
                # Special case for CI/CD
                if required_skill in ['ci/cd', 'ci-cd', 'cicd']:
                    ci_cd_terms = ['jenkins', 'gitlab ci', 'github actions', 'travis ci', 'circleci', 'azure pipelines']
                    if any(term in resume_skills for term in ci_cd_terms):
                        matching_skills.append({
                            'name': 'CI/CD',
                            'details': {'level': 'intermediate', 'years': 0},
                            'source': 'job' if required_skill in job_skills else 'market'
                        })
                        skill_found = True
                
                # Special case for Cloud
                if required_skill == 'cloud':
                    cloud_terms = ['aws', 'azure', 'gcp', 'google cloud', 'cloud computing']
                    if any(term in resume_skills for term in cloud_terms):
                        matching_skills.append({
                            'name': 'Cloud',
                            'details': {'level': 'intermediate', 'years': 0},
                            'source': 'job' if required_skill in job_skills else 'market'
                        })
                        skill_found = True
                
                # Special case for System Design
                if required_skill in ['system design', 'systems design']:
                    design_terms = ['distributed systems', 'scalable systems', 'microservices', 'system architecture']
                    if any(term in resume_skills for term in design_terms):
                        matching_skills.append({
                            'name': 'System Design',
                            'details': {'level': 'intermediate', 'years': 0},
                            'source': 'job' if required_skill in job_skills else 'market'
                        })
                        skill_found = True
            
            if not skill_found:
                missing_skills.append({
                    'name': required_skill.title(),
                    'source': 'job' if required_skill in job_skills else 'market'
                })
        
        # Calculate match percentage (based on job requirements only)
        total_required = len(job_skills)
        total_matching = len([skill for skill in matching_skills if skill['source'] == 'job'])
        match_percentage = (total_matching / total_required * 100) if total_required > 0 else 0
        
        result = {
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "match_percentage": round(match_percentage, 2),
            "total_required_skills": total_required,
            "total_matching_skills": total_matching,
            "analysis_summary": f"Found {total_matching} out of {total_required} required skills ({round(match_percentage, 2)}% match)"
        }
        
        logging.info(f"Skill gap analysis completed: {result['analysis_summary']}")
        return result
        
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {
            "matching_skills": [],
            "missing_skills": job_skills,
            "match_percentage": 0,
            "total_required_skills": len(job_skills),
            "total_matching_skills": 0,
            "analysis_summary": "Error occurred during skill gap analysis"
        }

def api_call_with_retry(url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
    """Make API call with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            logging.warning(f"API request failed with status {response.status_code} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.RequestException as e:
            logging.error(f"Request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None

def normalize_country(location: str) -> str:
    """Normalize country name for consistent lookup."""
    location = location.lower().strip()
    if 'united states' in location or 'usa' in location:
        return 'united states'
    elif 'united kingdom' in location or 'uk' in location:
        return 'united kingdom'
    elif 'india' in location:
        return 'india'
    return location

def get_country_code_and_currency(location: str) -> Tuple[str, str, str]:
    """Get country code and currency information based on location."""
    location = location.lower()
    normalized_country = normalize_country(location)
    
    country_info = {
        'india': ('in', '₹', '{:,.0f}'),
        'united states': ('us', '$', '${:,.0f}'),
        'united kingdom': ('gb', '£', '£{:,.0f}'),
        'canada': ('ca', 'C$', 'C${:,.0f}'),
        'australia': ('au', 'A$', 'A${:,.0f}'),
        'germany': ('de', '€', '€{:,.0f}'),
        'france': ('fr', '€', '€{:,.0f}'),
        'netherlands': ('nl', '€', '€{:,.0f}'),
        'singapore': ('sg', 'S$', 'S${:,.0f}'),
        'japan': ('jp', '¥', '¥{:,.0f}')
    }
    
    return country_info.get(normalized_country, ('in', '₹', '{:,.0f}'))

def format_salary(amount: float, currency_format: str) -> str:
    """Format salary according to currency format."""
    try:
        return currency_format.format(amount)
    except Exception as e:
        logging.error(f"Error formatting salary: {str(e)}")
        return str(amount)

def get_default_locations(country: str) -> List[str]:
    """Get default locations based on country."""
    locations = {
        'india': ['Bangalore', 'Mumbai', 'Hyderabad', 'Delhi NCR', 'Pune'],
        'united states': ['San Francisco', 'New York', 'Seattle', 'Austin', 'Boston'],
        'united kingdom': ['London', 'Manchester', 'Birmingham', 'Edinburgh', 'Bristol']
    }
    return locations.get(country, locations['india'])

def get_default_market_demand(location: str, experience_years: float) -> Dict[str, Any]:
    """Get default market demand data when API fails."""
    normalized_country = normalize_country(location)
    _, currency_symbol, _ = get_country_code_and_currency(normalized_country)
    
    # Base salary ranges by country (annual)
    base_ranges = {
        'india': (400000, 2500000),
        'united states': (60000, 150000),
        'united kingdom': (35000, 95000),
        'canada': (60000, 130000),
        'australia': (70000, 140000),
        'germany': (45000, 90000),
        'france': (35000, 80000),
        'netherlands': (40000, 85000),
        'singapore': (48000, 120000),
        'japan': (4000000, 12000000)  # JPY
    }
    
    base_min, base_max = base_ranges.get(normalized_country, base_ranges['india'])
    
    # Adjust salary based on experience
    if experience_years >= 5:
        min_salary = base_min * 1.5
        max_salary = base_max * 1.8
    elif experience_years >= 3:
        min_salary = base_min * 1.2
        max_salary = base_max * 1.4
    else:
        min_salary = base_min
        max_salary = base_max * 1.2
    
    return {
        "trend": "Stable",
        "trend_description": "Using default market data",
        "salary_range": f"{currency_symbol}{int(min_salary):,} - {currency_symbol}{int(max_salary):,}",
        "top_industries": ["Information Technology", "Software Development", "Computer Software", "Internet", "Technology Consulting"],
        "top_locations": get_default_locations(normalized_country),
        "top_demanded_skills": ["Python", "JavaScript", "Java", "SQL", "Cloud Computing"],
        "top_tech_stack": ["React", "Node.js", "AWS", "Docker", "Kubernetes"],
        "future_outlook": "Stable",
        "total_jobs": 50,
        "remote_percentage": "30%",
        "experience_level": "Entry" if experience_years < 2 else "Mid" if experience_years < 5 else "Senior",
        "market_insights": {
            "competition_level": "Medium",
            "best_time_to_apply": "Consider upskilling while applying",
            "salary_insights": f"Estimated salary range for {experience_years} years experience in {location}",
            "remote_work_availability": "Medium",
            "market_maturity": "Growing",
            "tech_stack_diversity": 15
        }
    }

# Common tech skills for market analysis
common_tech_skills = [
    "Python", "JavaScript", "Java", "C++", "Go",
    "React", "Angular", "Vue.js", "Node.js",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "SQL", "MongoDB", "PostgreSQL",
    "Machine Learning", "AI", "Data Science",
    "DevOps", "CI/CD", "Git",
    "Agile", "Scrum", "Project Management"
]

def extract_tech_stack(description: str) -> List[str]:
    """Extract technology stack from job description."""
    tech_keywords = {
        'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust'],
        'frontend': ['react', 'angular', 'vue', 'html', 'css'],
        'backend': ['node.js', 'django', 'flask', 'spring'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
        'database': ['sql', 'mongodb', 'postgresql', 'redis']
    }
    
    found_tech = []
    description_lower = description.lower()
    
    for category, techs in tech_keywords.items():
        for tech in techs:
            if tech in description_lower:
                found_tech.append(tech)
    
    return sorted(list(set(found_tech)))

def recommend_job_roles(skills: List[str], location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """Generate job role recommendations based on skills and experience using Adzuna API."""
    logging.info(f"Generating job recommendations for {experience_years} years of experience")
    
    try:
        app_id = os.getenv('ADZUNA_APP_ID')
        api_key = os.getenv('ADZUNA_API_KEY')
        
        if not app_id or not api_key:
            logging.warning("Adzuna credentials not found, using fallback data")
            return get_default_recommendations(location, experience_years)
        
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
                    for skill in common_tech_skills:
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
    missing_skills = set(common_tech_skills) - set(current_skills)
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
