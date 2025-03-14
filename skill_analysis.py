import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
import fitz  # PyMuPDF
import json
import numpy as np
import faiss
import google.generativeai as genai
import atexit
# Import from utils file
from utils import get_gemini_embedding, analyze_market_demand, get_industry_benchmarks, fetch_job_postings,recommend_job_roles, MATCH_THRESHOLD

# Configure Gemini API with your API key
genai.configure(api_key=" AIzaSyAtJIlLfcSHpNSKflzuE7jp_egbzkZB6yI") #1st api key
# Create a global model object that can be reused
gemini_pro_model = genai.GenerativeModel("gemini-1.5-pro-latest")

def get_model():
    """Returns a singleton instance of the model"""
    global gemini_pro_model
    if gemini_pro_model is None:
        gemini_pro_model = genai.GenerativeModel("gemini-1.5-pro-latest")
    return gemini_pro_model

# Register cleanup function to execute on program exit
def cleanup_resources():
    """Clean up any resources before exiting"""
    global gemini_pro_model
    gemini_pro_model = None
    # Force garbage collection
    import gc
    gc.collect()
    print("Resources cleaned up.")

# Register the cleanup function
atexit.register(cleanup_resources)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a resume PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_resume_details(text):
    """Extracts skills, job experience, and education from resume text using Gemini."""
    if not text:
        return {"skills": [], "experience": [], "education": []}
    
    model = get_model()

    prompt = f"""
    Extract the following details from the resume text and return them in JSON format:
    {{
      "skills": ["Skill1", "Skill2", "Skill3"],
      "experience": [
        {{"position": "Job Title", "company": "Company Name", "duration": "Start Date - End Date"}}
      ],
      "education": [
        {{"degree": "Degree Name", "institution": "University Name", "year": "Year of Graduation"}}
      ]
    }}

    Resume Text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Debug - print the raw response to see what we're working with
        print("Raw response from Gemini:")
        print(response_text)
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1]
        if "```" in response_text:
            response_text = response_text.split("```")[0]
        
        response_text = response_text.strip()
        
        # Try to parse as JSON
        try:
            extracted_data = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {json_err}")
            # Fallback to empty structure
            extracted_data = {}

        # Ensure expected keys are present, else return defaults
        return {
            "skills": extracted_data.get("skills", []),
            "experience": extracted_data.get("experience", []),
            "education": extracted_data.get("education", [])
        }
        
    except Exception as e:
        print(f"Error extracting resume details: {e}")
        return {"skills": [], "experience": [], "education": []}

def save_resume_to_json(resume_details, output_file="resume_details.json"):
    """
    Saves the extracted resume details to a JSON file.
    """
    try:
        # Write the dictionary to a JSON file with proper formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resume_details, f, indent=4, ensure_ascii=False)
        
        print(f"Resume details successfully saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error saving resume details to JSON: {e}")
        return False

def get_skill_embeddings(skills):
    """Get embeddings for a list of skills"""
    embeddings = []
    dimension = None
    
    for skill in skills:
        embedding = get_gemini_embedding(skill)
        if embedding:
            embeddings.append(embedding)
            # Set dimension based on first successful embedding
            if dimension is None:
                dimension = len(embedding)
        else:
            print(f"Failed to get embedding for skill: {skill}")
            # Skip this skill instead of adding a zero vector
    
    if not embeddings:
        return np.array([]).astype('float32')
    
    return np.array(embeddings).astype('float32')

def build_skill_index(skills):
    """Build a FAISS index from a list of skills"""
    # Get embeddings for all skills
    embeddings = get_skill_embeddings(skills)
    
    if len(embeddings) == 0:
        print("No embeddings to index")
        return None
    
    # Get dimension of embeddings
    dimension = embeddings.shape[1]
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    print(f"Added {len(skills)} skill embeddings to FAISS index")
    
    return index

def extract_job_requirements(job_description):
    """Extract required skills from a job description using Gemini."""
    if not job_description:
        return []
        
    model = get_model()
    
    prompt = f"""
    Extract a list of required skills or technologies from this job description.
    Return them as a JSON array of strings with only the skill names:
    ["Skill1", "Skill2", "Skill3"]
    
    Job Description:
    {job_description}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response of code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1]
        if "```" in response_text:
            response_text = response_text.split("```")[0]
        
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            required_skills = json.loads(response_text)
            if not isinstance(required_skills, list):
                print("Unexpected response format - not a list")
                required_skills = []
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            required_skills = []

        return required_skills
        
    except Exception as e:
        print(f"Error extracting job requirements: {e}")
        return []

# Update in skill_analysis.py
def analyze_skill_gaps(resume_skills, job_skills, job_title=None):
    """
    Analyze gaps between resume skills and job requirements using FAISS.
    Also compares skills against industry benchmarks if job_title is provided.
    """
    if not resume_skills or not job_skills:
        return {
            "missing_skills": job_skills,
            "matched_skills": [],
            "match_percentage": 0,
            "industry_benchmark_comparison": None
        }
    
    # Build index for resume skills
    resume_index = build_skill_index(resume_skills)
    
    if resume_index is None:
        return {
            "missing_skills": job_skills,
            "matched_skills": [],
            "match_percentage": 0,
            "industry_benchmark_comparison": None
        }
    
    # For each job skill, find if there's a similar resume skill
    matched_skills = []
    missing_skills = []
    resume_skill_mapping = {i: skill for i, skill in enumerate(resume_skills)}
    
    for job_skill in job_skills:
        # Get embedding for job skill
        job_embedding = get_gemini_embedding(job_skill)
        
        if not job_embedding:
            print(f"Failed to get embedding for job skill: {job_skill}")
            missing_skills.append(job_skill)
            continue
        
        # Convert to the right format for FAISS
        job_embedding = np.array([job_embedding]).astype('float32')
        
        # Search for similar skills in resume
        distances, indices = resume_index.search(job_embedding, 1)  # Get closest match
        
        # Check if there's a good match (using shared threshold from utils)
        if distances[0][0] < MATCH_THRESHOLD:  # Lower distance = better match
            matched_skills.append({
                "job_skill": job_skill,
                "resume_skill": resume_skill_mapping.get(indices[0][0], "Unknown skill"),
                "distance": float(distances[0][0])
            })
        else:
            missing_skills.append(job_skill)
    
    # Calculate match percentage
    match_percentage = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0
    
    # Add industry benchmark comparison if job_title is provided
    industry_benchmark_comparison = None
    if job_title:
        # Fetch industry benchmarks
        benchmarks = get_industry_benchmarks(job_title)
        
        # Extract all benchmark technical skills
        benchmark_skills = []
        if "technical_skills" in benchmarks:
            for skill_item in benchmarks["technical_skills"]:
                benchmark_skills.append(skill_item["skill"])
        
        # Build index for benchmark skills
        benchmark_index = build_skill_index(benchmark_skills)
        benchmark_matches = []
        missing_benchmark_skills = []
        
        if benchmark_index is not None:
            for resume_skill in resume_skills:
                # Get embedding for resume skill
                resume_embedding = get_gemini_embedding(resume_skill)
                
                if not resume_embedding:
                    continue
                
                # Convert to the right format for FAISS
                resume_embedding = np.array([resume_embedding]).astype('float32')
                
                # Search for similar skills in benchmarks
                distances, indices = benchmark_index.search(resume_embedding, 1)
                
                # Check if there's a good match
                if distances[0][0] < MATCH_THRESHOLD:
                    benchmark_skill = benchmark_skills[indices[0][0]]
                    benchmark_matches.append({
                        "benchmark_skill": benchmark_skill,
                        "resume_skill": resume_skill,
                        "distance": float(distances[0][0])
                    })
            
            # Identify missing benchmark skills
            matched_benchmark_skill_names = [match["benchmark_skill"] for match in benchmark_matches]
            missing_benchmark_skills = [skill for skill in benchmark_skills if skill not in matched_benchmark_skill_names]
        
        # Calculate benchmark match percentage
        benchmark_match_percentage = (len(benchmark_matches) / len(benchmark_skills) * 100) if benchmark_skills else 0
        
        industry_benchmark_comparison = {
            "benchmark_skills": benchmark_skills,
            "matched_benchmark_skills": benchmark_matches,
            "missing_benchmark_skills": missing_benchmark_skills,
            "benchmark_match_percentage": benchmark_match_percentage,
            "soft_skills": benchmarks.get("soft_skills", []),
            "certifications": benchmarks.get("certifications", []),
            "experience_levels": benchmarks.get("experience_levels", {})
        }
    
    # Include benchmark comparison in the result
    return {
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "match_percentage": match_percentage,
        "industry_benchmark_comparison": industry_benchmark_comparison
    }



# Add to imports at the top of skill_analysis.py
from utils import analyze_market_demand

# Update the run_skill_gap_analysis function to include job title extraction
# Update in skill_analysis.py
def run_skill_gap_analysis(resume_path, job_description_text):
    """
    Complete workflow for skill gap analysis, including job role recommendations based on resume skills.
    """
    try:
        # Extract text from resume
        resume_text = extract_text_from_pdf(resume_path)
        
        # Extract details from resume
        resume_details = extract_resume_details(resume_text)
        
        # Save resume details to JSON
        save_resume_to_json(resume_details)
        
        # Extract required skills from job description
        job_skills = extract_job_requirements(job_description_text)
        
        # Extract job title from job description
        job_title = extract_job_title(job_description_text)
        
        # Analyze skill gaps (now includes industry benchmark comparison)
        gap_analysis = analyze_skill_gaps(resume_details["skills"], job_skills, job_title)
        
        # Get market demand analysis for the job title
        market_analysis = analyze_market_demand(job_title)
        
        # Fetch automated job matches based on resume skills
        job_matches = fetch_job_postings(job_title, resume_details["skills"], max_results=5)
        
        # Recommend job roles based on resume skills
        # (Assumes recommend_job_roles() is defined in utils and imported)
        from utils import recommend_job_roles
        recommended_roles = recommend_job_roles(resume_details["skills"])
        
        # Combine results
        analysis_results = {
            "job_title": job_title,
            "resume_skills": resume_details["skills"],
            "job_required_skills": job_skills,
            "skill_gap_analysis": gap_analysis,
            "market_demand": market_analysis["market_demand"],
            "job_matches": job_matches["job_matches"],
            "learning_recommendations": {"recommendations": []},
            "job_recommendations": recommended_roles  # New key for recommended job roles
        }
        
        # Save complete analysis to JSON
        with open("skill_gap_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)
        
        print("Skill gap analysis complete. Results saved to skill_gap_analysis.json")
        
        return analysis_results
        
    except Exception as e:
        print(f"Error in skill gap analysis: {e}")
        return {
            "job_title": "",
            "resume_skills": [],
            "job_required_skills": [],
            "skill_gap_analysis": {"missing_skills": [], "matched_skills": [], "match_percentage": 0},
            "market_demand": {},
            "job_matches": [],
            "learning_recommendations": {"recommendations": []},
            "job_recommendations": {"job_recommendations": []}
        }

# Add this new function to extract job title
def extract_job_title(job_description):
    """Extract job title from job description using Gemini."""
    if not job_description:
        return "Unknown Job Title"
        
    model = get_model()
    
    prompt = f"""
    Extract the specific job title from this job description. 
    Return only the exact title without any explanation or additional text.
    
    Job Description:
    {job_description}
    """

    try:
        response = model.generate_content(prompt)
        job_title = response.text.strip()
        return job_title
        
    except Exception as e:
        print(f"Error extracting job title: {e}")
        return "Unknown Job Title"
    

# Main execution block
if __name__ == "__main__":
    try:
        # Example job description
        job_description = """
        Job Description: Senior Python Developer
        We are looking for a Senior Python Developer with expertise in machine learning, 
        data analysis, and web development. The ideal candidate should have experience with 
        frameworks like Flask or Django, database management systems, and cloud platforms 
        like AWS or Azure. Knowledge of Docker, CI/CD pipelines, and agile methodologies is required.
        """

        # Run the analysis
        analysis = run_skill_gap_analysis("deep podder.pdf", job_description)

        # Print a summary
        print(f"\nJob Title: {analysis['job_title']}")
        print(f"\nSkill Match: {analysis['skill_gap_analysis']['match_percentage']:.1f}%")
        
        print("\nMissing Skills:")
        for skill in analysis['skill_gap_analysis']['missing_skills']:
            print(f"- {skill}")

        print("\nMatched Skills:")
        for match in analysis['skill_gap_analysis']['matched_skills']:
            print(f"- {match['job_skill']} (matched with: {match['resume_skill']})")

        # Print market demand information
        if analysis['market_demand']:
            market = analysis['market_demand']
            print("\nMarket Demand Analysis:")
            print(f"- Current Trend: {market.get('demand_trend', 'Unknown')}")
            print(f"- Trend Description: {market.get('trend_description', 'No data available')}")
            print(f"- Salary Range: {market.get('salary_range', 'Unknown')}")
            
            if 'top_industries' in market and market['top_industries']:
                print("- Top Industries Hiring:")
                for industry in market['top_industries'][:3]:
                    print(f"  * {industry}")
                    
            if 'top_locations' in market and market['top_locations']:
                print("- Top Locations:")
                for location in market['top_locations'][:3]:
                    print(f"  * {location}")
            
            print(f"- Future Outlook: {market.get('future_outlook', 'No data available')}")
    
    finally:
        # Final cleanup
        cleanup_resources()
        print("Script execution complete.")