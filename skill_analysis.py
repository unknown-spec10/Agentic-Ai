import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
import json
import numpy as np
import faiss
import google.generativeai as genai
import atexit
from typing import List, Dict, Any, Optional
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure Gemini API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY secret is not set")

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

def get_gemini_embedding(text: str) -> List[float]:
    """Generate embedding using a consistent hashing approach."""
    try:
        # Generate a more stable embedding using a consistent hashing approach
        # Use a fixed dimension of 128 for embeddings
        embedding_dim = 128
        embedding = np.zeros(embedding_dim)
        
        # Normalize the text
        text = text.lower().strip()
        words = text.split()
        
        # Generate embedding values using word hashing
        for i, word in enumerate(words):
            # Use multiple hash functions to create a more distributed embedding
            h1 = hash(word) % embedding_dim
            h2 = hash(word + '_alt') % embedding_dim
            h3 = hash(f'prefix_{word}') % embedding_dim
            
            # Add weighted values to different positions
            embedding[h1] += 1.0
            embedding[h2] += 0.5
            embedding[h3] += 0.25
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        logging.error(traceback.format_exc())
        # Return a normalized default embedding
        default = np.zeros(128)
        default[0] = 1.0
        return default.tolist()

def get_model():
    """Returns a singleton instance of the model with retry logic."""
    global model
    if model is None:
        try:
            model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
    return model

def cleanup_resources():
    """Clean up any resources used during the analysis."""
    try:
        # Clean up any temporary files or resources here
        if os.path.exists("temp_embeddings.npy"):
            os.remove("temp_embeddings.npy")
        logging.info("Resources cleaned up successfully")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        logging.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_skill_level(text: str, skill: str, variations: List[str], overall_experience: float = 0) -> Dict[str, Any]:
    """Extract skill level and frequency information."""
    text_lower = text.lower()
    
    # Count mentions of the skill and its variations
    frequency = sum(text_lower.count(var.lower()) for var in variations)
    
    # Look for experience indicators
    experience_patterns = {
        'expert': [
            r'\bexpert\s+(?:in\s+|with\s+)?' + skill.lower(),
            r'\b\d{3,}\+?\s+projects?\s+(?:in|with)\s+' + skill.lower(),
            r'\b(?:senior|lead|principal)\s+' + skill.lower(),
            r'\b' + skill.lower() + r'\s+(?:expert|specialist|architect)',
            r'\badvanced\s+certification\s+(?:in|for)\s+' + skill.lower(),
            r'\bmentored|trained\s+(?:teams?|others)\s+(?:in|on)\s+' + skill.lower(),
            r'\bdesigned\s+(?:and|&)\s+implemented\s+.*?' + skill.lower(),
            r'\barchitected\s+.*?' + skill.lower(),
            r'\b(?:key|primary)\s+contributor\s+.*?' + skill.lower(),
            r'\b(?:extensive|comprehensive)\s+experience\s+.*?' + skill.lower()
        ],
        'advanced': [
            r'\badvanced\s+(?:knowledge\s+of\s+)?' + skill.lower(),
            r'\b(?:extensive|strong)\s+experience\s+(?:in|with)\s+' + skill.lower(),
            r'\b\d{2}\+?\s+projects?\s+(?:in|with)\s+' + skill.lower(),
            r'\b(?:developed|implemented|architected)\s+(?:\w+\s+)*' + skill.lower(),
            r'\bproficient\s+(?:in|with)\s+' + skill.lower(),
            r'\bsignificant\s+experience\s+(?:in|with)\s+' + skill.lower(),
            r'\bsuccessfully\s+delivered\s+.*?' + skill.lower(),
            r'\boptimized\s+.*?' + skill.lower() + r'\s+performance',
            r'\btroubleshooting\s+.*?' + skill.lower() + r'\s+issues',
            r'\b(?:led|managed)\s+.*?' + skill.lower() + r'\s+development'
        ],
        'intermediate': [
            r'\bintermediate\s+(?:knowledge\s+of\s+)?' + skill.lower(),
            r'\bexperience\s+(?:in|with)\s+' + skill.lower(),
            r'\b(?:worked|familiar)\s+(?:with|on)\s+' + skill.lower(),
            r'\bcontributed\s+to\s+.*?' + skill.lower(),
            r'\bparticipated\s+in\s+.*?' + skill.lower(),
            r'\bused\s+' + skill.lower() + r'\s+(?:for|in)\s+projects?',
            r'\bknowledge\s+of\s+' + skill.lower(),
            r'\bunderstanding\s+of\s+' + skill.lower(),
            r'\bpractical\s+experience\s+(?:in|with)\s+' + skill.lower(),
            r'\b(?:implemented|developed)\s+.*?' + skill.lower()
        ],
        'beginner': [
            r'\bbasic\s+(?:knowledge\s+of\s+)?' + skill.lower(),
            r'\bfamiliarity\s+with\s+' + skill.lower(),
            r'\bexposure\s+to\s+' + skill.lower(),
            r'\blearning\s+' + skill.lower(),
            r'\bintroduction\s+to\s+' + skill.lower(),
            r'\bfundamentals\s+of\s+' + skill.lower(),
            r'\bstarted\s+(?:learning|working\s+with)\s+' + skill.lower(),
            r'\bentry[- ]level\s+.*?' + skill.lower(),
            r'\bcompleted\s+courses?\s+(?:in|on)\s+' + skill.lower(),
            r'\bbeginning\s+to\s+use\s+.*?' + skill.lower()
        ]
    }
    
    # Look for years of experience with more variations
    year_patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|work(?:ing)?)\s+(?:with|in|using)\s+' + skill.lower(),
        r'(?:with|in|using)\s+' + skill.lower() + r'\s+(?:for\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)',
        r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+' + skill.lower() + r'\s+experience',
        r'(?:since|from)\s+(\d{4})\s+.*?' + skill.lower()  # Match year mentions
    ]
    
    years = 0
    for pattern in year_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                if len(match.group(1)) == 4:  # Year format (e.g., 2020)
                    years = datetime.now().year - int(match.group(1))
                else:
                    years = float(match.group(1))
                break
            except ValueError:
                continue
    
    # If no specific years found for the skill, use a portion of overall experience
    if years == 0 and overall_experience > 0:
        years = overall_experience * 0.8  # Assume 80% of overall experience applies to core skills
    
    # Determine level based on patterns and years
    level = 'beginner'
    max_confidence = 0
    
    for potential_level, patterns in experience_patterns.items():
        # Count how many patterns match
        matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
        confidence = matches / len(patterns)
        
        if confidence > max_confidence:
            level = potential_level
            max_confidence = confidence
    
    # Adjust level based on years of experience and frequency
    if years > 0:
        if years >= 5 or (years >= 3 and frequency >= 5):
            level = 'expert'
        elif years >= 3 or (years >= 2 and frequency >= 4):
            level = 'advanced'
        elif years >= 1 or frequency >= 3:
            level = 'intermediate'
    
    # Look for certifications with more variations
    cert_patterns = [
        r'(?:certification|certified|certificate)\s+(?:in|for)\s+' + skill.lower(),
        r'(?:' + skill.lower() + r')\s+(?:certification|certified|certificate)',
        r'(?:aws|microsoft|google|oracle)\s+certified\s+.*?' + skill.lower(),
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

def calculate_skill_weight(level: str, years: float, frequency: int, has_certification: bool, has_leadership: bool, confidence: float, overall_experience: float = 0) -> float:
    """Calculate a weighted score for a skill based on multiple factors."""
    # Base weights for each level
    level_weights = {
        'expert': 1.0,
        'advanced': 0.8,
        'intermediate': 0.6,
        'beginner': 0.4
    }
    
    # Start with base weight from level
    weight = level_weights.get(level, 0.4)
    
    # Add weight for years of experience (logarithmic scale to prevent overweighting)
    # Max contribution of 0.4 from years, reaching 95% of max at 10 years
    years_weight = 0.4 * (1 - math.exp(-years/5))
    
    # Add weight for frequency of mention (diminishing returns)
    # Max contribution of 0.2 from frequency
    frequency_weight = 0.2 * (1 - math.exp(-frequency/10))
    
    # Add weight for certifications and leadership
    cert_weight = 0.2 if has_certification else 0
    leadership_weight = 0.2 if has_leadership else 0
    
    # Factor in overall experience (small boost for experienced professionals)
    experience_boost = min(0.1 * (overall_experience / 5), 0.2)  # Max 20% boost for 10+ years
    
    # Factor in confidence of level detection
    confidence_factor = 0.5 + (0.5 * confidence)  # Range: 0.5 to 1.0
    
    # Combine weights with confidence factor and experience boost
    total_weight = (weight + years_weight + frequency_weight + cert_weight + leadership_weight + experience_boost) * confidence_factor
    
    # Ensure weight stays in [0, 1] range
    return round(min(max(total_weight, 0), 1), 2)

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

def analyze_skill_gaps(resume_details: Dict[str, Any], job_skills: List[str], job_title: str) -> Dict[str, Any]:
    """Analyze gaps between resume skills and job requirements."""
    logging.info("Starting skill gap analysis...")
    
    # Handle both dictionary and list inputs
    if isinstance(resume_details, list):
        # Convert list to dictionary format
        resume_details = {
            "skills": resume_details,
            "skill_details": {skill: {
                "level": "beginner",
                "years": 0,
                "frequency": 1,
                "has_certification": False,
                "has_leadership": False,
                "confidence": 0,
                "weight": 0.2
            } for skill in resume_details}
        }
    
    # Convert job skills to lowercase for comparison
    job_skills_lower = set(s.lower() for s in job_skills)
    resume_skills_lower = {s.lower(): details for s, details in resume_details.get("skill_details", {}).items()}
    
    # Find matching and missing skills
    matching_skills = []
    missing_skills = []
    
    for job_skill in job_skills:
        job_skill_lower = job_skill.lower()
        if job_skill_lower in resume_skills_lower:
            skill_info = {
                "name": job_skill,
                "details": resume_skills_lower[job_skill_lower]
            }
            matching_skills.append(skill_info)
        else:
            missing_skills.append(job_skill)
    
    # Calculate weighted match percentage
    if not job_skills:
        match_percentage = 100.0
    else:
        total_weight = len(job_skills)  # Each required skill has base weight of 1
        matched_weight = sum(skill["details"]["weight"] for skill in matching_skills)
        match_percentage = (matched_weight / total_weight) * 100
    
    # Sort matching skills by weight
    matching_skills.sort(key=lambda x: (-x["details"]["weight"], x["name"]))
    
    logging.info(f"Skill gap analysis completed. Match percentage: {match_percentage}%")
    
    return {
        "job_title": job_title,
        "match_percentage": round(match_percentage, 2),
        "matching_skills": matching_skills,
        "missing_skills": sorted(missing_skills)
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
    """
    Format salary based on country's currency format.
    
    Args:
        amount (float): Salary amount
        currency_format (str): Currency format type
        
    Returns:
        str: Formatted salary string
    """
    if currency_format == "indian":
        # Convert to Indian format (lakhs)
        if amount >= 100000:
            amount_lakhs = amount / 100000
            return f"₹{amount_lakhs:.2f} L"
        return f"₹{amount:,.2f}"
    elif currency_format == "british":
        return f"£{amount:,.0f}"
    else:  # US format
        return f"${amount:,.0f}"

def analyze_market_demand(job_title: str, location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """Analyze market demand for the job title using Adzuna API with experience-based insights."""
    logging.info(f"Analyzing market demand for {job_title} in {location} with {experience_years} years of experience")
    
    try:
        app_id = os.getenv('ADZUNA_APP_ID')
        api_key = os.getenv('ADZUNA_API_KEY')
        
        if not app_id or not api_key:
            logging.warning("Adzuna credentials not found, using fallback data")
            return get_default_market_demand(location, experience_years)
        
        # Get country-specific information
        country_code, currency_symbol, currency_format = get_country_code_and_currency(location)
        
        # Get market statistics from Adzuna
        base_url = "https://api.adzuna.com/v1/api"
        
        # Generate search variations based on job title and experience
        experience_level = "entry" if experience_years < 2 else "mid" if experience_years < 5 else "senior"
        search_variations = [
            f"{experience_level} {job_title}",
            f"{job_title}",
            f"{experience_level} {job_title.replace('Developer', 'Engineer')}",
            f"{experience_level} {job_title.split()[0]} Developer",
            f"{experience_level} {job_title.split()[0]} Engineer"
        ]
        
        all_jobs = []
        seen_jobs = set()  # Track unique job IDs
        
        # Search with multiple variations
        for search_term in search_variations:
            params = {
                "app_id": app_id,
                "app_key": api_key,
                "what": search_term.strip(),
                "where": location,
                "max_days_old": 60,
                "sort_by": "date",
                "full_time": 1,
                "results_per_page": 100
            }
            
            data = api_call_with_retry(f"{base_url}/jobs/{country_code}/search/1", params)
            if data and 'results' in data:
                # Add only unique jobs
                for job in data['results']:
                    job_id = job.get('id')
                    if job_id and job_id not in seen_jobs:
                        all_jobs.append(job)
                        seen_jobs.add(job_id)
        
        if not all_jobs:
            logging.warning("No jobs found in API response")
            return get_default_market_demand(location, experience_years)
        
        # Calculate salary ranges with outlier detection
        salaries = [job.get('salary_min', 0) for job in all_jobs if job.get('salary_min', 0) > 0]
        if salaries:
            # Remove outliers using IQR method
            q1 = np.percentile(salaries, 25)
            q3 = np.percentile(salaries, 75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            filtered_salaries = [s for s in salaries if lower_bound <= s <= upper_bound]
            
            if filtered_salaries:
                min_salary = min(filtered_salaries)
                max_salary = max(filtered_salaries)
                avg_salary = sum(filtered_salaries) / len(filtered_salaries)
                
                # Adjust salary range based on experience
                if experience_years >= 5:
                    min_salary = avg_salary * 1.2
                    max_salary = max_salary * 1.3
                elif experience_years >= 3:
                    min_salary = avg_salary * 0.9
                    max_salary = max_salary * 1.1
                
                salary_range = f"{format_salary(min_salary, currency_format)} - {format_salary(max_salary, currency_format)}"
            else:
                salary_range = get_default_market_demand(location, experience_years)["salary_range"]
        else:
            salary_range = get_default_market_demand(location, experience_years)["salary_range"]
        
        # Extract industries and locations with improved categorization
        industries = {}
        locations = {}
        skills_demand = {}
        remote_count = 0
        tech_stack_mentions = {}
        
        for job in all_jobs:
            # Count industries with subcategories
            category = job.get('category', {}).get('label', 'Technology')
            subcategory = job.get('category', {}).get('tag', '')
            industry_key = f"{category} - {subcategory}" if subcategory else category
            industries[industry_key] = industries.get(industry_key, 0) + 1
            
            # Enhanced location tracking
            job_location = job.get('location', {}).get('display_name', '')
            if job_location:
                locations[job_location] = locations.get(job_location, 0) + 1
            
            # Improved remote work detection
            description = job.get('description', '').lower()
            if any(term in description for term in [
                'remote', 'work from home', 'wfh', 'virtual', 'telecommute',
                'anywhere', 'flexible location', 'remote-first', 'fully remote'
            ]):
                remote_count += 1
            
            # Enhanced skill detection
            for skill in common_tech_skills:
                if skill.lower() in description:
                    skills_demand[skill] = skills_demand.get(skill, 0) + 1
            
            # Track technology stack mentions
            tech_stack = extract_tech_stack(description)
            for tech in tech_stack:
                tech_stack_mentions[tech] = tech_stack_mentions.get(tech, 0) + 1
        
        # Calculate trends and insights
        current_count = len(all_jobs)
        trend = "Growing" if current_count > 50 else "Stable" if current_count > 20 else "Limited"
        remote_percentage = int((remote_count / current_count) * 100) if current_count > 0 else 0
        
        # Sort and get top results
        top_industries = sorted(industries.items(), key=lambda x: x[1], reverse=True)[:5]
        top_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]
        top_skills = sorted(skills_demand.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tech_stack = sorted(tech_stack_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "trend": trend,
            "trend_description": f"Found {current_count} active job postings in the last 60 days",
            "salary_range": salary_range,
            "top_industries": [ind[0] for ind in top_industries],
            "top_locations": [loc[0] for loc in top_locations],
            "top_demanded_skills": [skill[0] for skill in top_skills],
            "top_tech_stack": [tech[0] for tech in top_tech_stack],
            "future_outlook": "Positive" if trend == "Growing" else "Stable" if trend == "Stable" else "Competitive",
            "total_jobs": current_count,
            "remote_percentage": f"{remote_percentage}%",
            "experience_level": experience_level.title(),
            "market_insights": {
                "competition_level": "High" if current_count > 100 else "Medium" if current_count > 50 else "Low",
                "best_time_to_apply": "Current market is actively hiring" if trend == "Growing" else "Consider upskilling first",
                "salary_insights": f"Salary range for {experience_years} years experience in {location}",
                "remote_work_availability": "High" if remote_percentage > 50 else "Medium" if remote_percentage > 25 else "Low",
                "market_maturity": "Mature" if current_count > 200 else "Growing" if current_count > 50 else "Emerging",
                "tech_stack_diversity": len(tech_stack_mentions)
            }
        }
        
    except Exception as e:
        logging.error(f"Error in market demand analysis: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return get_default_market_demand(location, experience_years)

def extract_tech_stack(description: str) -> List[str]:
    """Extract technology stack mentions from job description."""
    tech_patterns = {
        'frontend': ['react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css'],
        'backend': ['node', 'python', 'java', 'go', 'ruby', 'php', 'scala', 'c#'],
        'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch'],
        'cloud': ['aws', 'azure', 'gcp', 'cloud'],
        'devops': ['docker', 'kubernetes', 'jenkins', 'gitlab', 'github actions'],
        'testing': ['jest', 'cypress', 'selenium', 'pytest', 'junit'],
        'tools': ['git', 'jira', 'confluence', 'slack', 'teams']
    }
    
    found_tech = []
    description_lower = description.lower()
    
    for category, technologies in tech_patterns.items():
        for tech in technologies:
            if tech in description_lower:
                found_tech.append(tech)
    
    return found_tech

def normalize_country(country: str) -> str:
    """Normalize country name to standard format."""
    country_map = {
        # India variations
        "india": "india",
        "in": "india",
        "ind": "india",
        # UK variations
        "uk": "uk",
        "united kingdom": "uk",
        "britain": "uk",
        "gb": "uk",
        # US variations
        "us": "us",
        "usa": "us",
        "united states": "us",
        "america": "us"
    }
    return country_map.get(country.lower().strip(), "india")  # Default to India if unknown

def get_default_locations(country: str = "india") -> List[str]:
    """
    Returns a list of major IT hubs based on the country.
    
    Args:
        country (str): Country name (india, uk, or us)
        
    Returns:
        List[str]: List of major IT hub locations
    """
    normalized_country = normalize_country(country)
    
    locations = {
        "india": [
            "Bangalore", "Mumbai", "Delhi NCR", "Hyderabad", "Pune",
            "Chennai", "Kolkata", "Ahmedabad", "Chandigarh", "Kochi",
            "Noida", "Gurgaon", "Coimbatore", "Indore", "Thiruvananthapuram"
        ],
        "uk": [
            "London", "Manchester", "Birmingham", "Edinburgh", "Bristol",
            "Glasgow", "Leeds", "Cambridge", "Oxford", "Reading",
            "Cardiff", "Belfast", "Newcastle", "Liverpool", "Brighton"
        ],
        "us": [
            "San Francisco Bay Area", "Seattle", "New York City", "Boston", "Austin",
            "Los Angeles", "Denver", "Chicago", "Washington DC", "Portland",
            "San Diego", "Atlanta", "Raleigh-Durham", "Miami", "Salt Lake City"
        ]
    }
    return locations.get(normalized_country, locations["india"])

def get_default_market_demand(location: str = "india", experience_years: float = 0) -> Dict[str, Any]:
    """Return enhanced default market demand data based on location and experience."""
    country_code, currency_symbol, currency_format = get_country_code_and_currency(location)
    
    # Experience-based salary ranges
    if currency_format == "indian":
        if experience_years >= 5:
            salary_range = f"{currency_symbol}15.00 L - {currency_symbol}35.00 L"
        elif experience_years >= 3:
            salary_range = f"{currency_symbol}8.00 L - {currency_symbol}18.00 L"
        elif experience_years >= 1:
            salary_range = f"{currency_symbol}5.00 L - {currency_symbol}10.00 L"
        else:
            salary_range = f"{currency_symbol}3.00 L - {currency_symbol}6.00 L"
    elif currency_format == "british":
        if experience_years >= 5:
            salary_range = f"{currency_symbol}65,000 - {currency_symbol}120,000"
        elif experience_years >= 3:
            salary_range = f"{currency_symbol}45,000 - {currency_symbol}80,000"
        elif experience_years >= 1:
            salary_range = f"{currency_symbol}30,000 - {currency_symbol}50,000"
        else:
            salary_range = f"{currency_symbol}25,000 - {currency_symbol}35,000"
    else:  # US format
        if experience_years >= 5:
            salary_range = f"{currency_symbol}120,000 - {currency_symbol}200,000"
        elif experience_years >= 3:
            salary_range = f"{currency_symbol}90,000 - {currency_symbol}150,000"
        elif experience_years >= 1:
            salary_range = f"{currency_symbol}70,000 - {currency_symbol}100,000"
        else:
            salary_range = f"{currency_symbol}50,000 - {currency_symbol}80,000"
    
    experience_level = "Senior" if experience_years >= 5 else "Mid-Level" if experience_years >= 2 else "Entry-Level"
    
    return {
        "trend": "Stable",
        "trend_description": f"Market outlook for {experience_level} positions",
        "salary_range": salary_range,
        "top_industries": [
            "Enterprise Software Development",
            "Cloud Services & DevOps",
            "FinTech & Banking Technology",
            "AI/ML & Data Engineering",
            "Cybersecurity & InfoSec"
        ],
        "top_locations": get_default_locations(location),
        "top_demanded_skills": [
            "Python",
            "JavaScript",
            "Cloud (AWS/Azure)",
            "Data Analysis",
            "Machine Learning"
        ],
        "future_outlook": "Positive",
        "total_jobs": 1000,
        "remote_percentage": "40%",
        "experience_level": experience_level,
        "market_insights": {
            "competition_level": "Medium",
            "best_time_to_apply": "Market is generally stable for experienced professionals" if experience_years >= 3 else "Entry level positions are competitive",
            "salary_insights": f"Typical salary range for {experience_years} years experience in {location}",
            "remote_work_availability": "Medium to High"
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
                continue
        
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

def run_skill_gap_analysis(resume_input: str, job_description: str, location: str = "india", overall_experience: float = 0) -> Dict[str, Any]:
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
        
        # Extract details with overall experience
        logging.info("Extracting resume details...")
        resume_details = extract_resume_details(resume_text, overall_experience)
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
        gap_analysis = analyze_skill_gaps(resume_details, job_skills, job_title)
        
        # Adjust match percentage based on overall experience
        if overall_experience > 0:
            experience_factor = min(overall_experience / 5, 1)  # Cap at 5 years
            gap_analysis["match_percentage"] = min(
                100,
                gap_analysis["match_percentage"] * (1 + experience_factor * 0.2)  # Up to 20% boost
            )
        
        # Get market demand data
        logging.info("Analyzing market demand...")
        market_demand = analyze_market_demand(job_title, location, overall_experience)
        
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
