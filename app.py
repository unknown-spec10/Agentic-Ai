import streamlit as st
import json
import tempfile
import os
from utils import fetch_job_postings, generate_learning_recommendations, recommend_job_roles, generate_gemini_content, get_country_code_and_currency, format_salary, analyze_market_demand
from skill_analysis import (
    extract_text_from_pdf,
    extract_resume_details,
    extract_job_requirements,
    run_skill_gap_analysis,
    analyze_skill_gaps
)
from update_analysis import update_skill_analysis

# Streamlit UI
st.set_page_config(page_title="Skill Gap Analysis & Job Search", layout="wide")

st.title("🚀 Skill Gap Analysis & Job Search")
st.subheader("Upload your resume and enter your aspired role to analyze skill gaps and find jobs!")

# Initialize session state for analysis results if not exists
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# User Inputs
col1, col2, col3 = st.columns(3)
with col1:
    job_title = st.text_input("🔹 Aspired Role", placeholder="Enter aspired job role (e.g., Data Scientist)")
with col2:
    location = st.selectbox("📍 Location", 
                        ["India", "United Kingdom", "United States"],
                        index=0,
                        help="Select your target job market")
with col3:
    experience_level = st.selectbox(
        "👨‍💼 Experience Level",
        ["Fresher", "1 year", "2 years", "3 years", "4 years", "5+ years"],
        help="Select your overall work experience"
    )

# Convert experience level to years for calculations
experience_years = {
    "Fresher": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5+ years": 5
}

resume = st.file_uploader("📝 Upload Resume (PDF)", type=["pdf"])

if resume and job_title:
    with st.spinner("Processing resume and analyzing skills..."):
        try:
            # Get years of experience from the selection
            experience_years_value = experience_years[experience_level]
            
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(resume.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Extract text from PDF using the temporary file
                resume_text = extract_text_from_pdf(temp_path)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                if not resume_text:
                    st.error("Could not extract text from the PDF. Please ensure it's a valid, text-based PDF file.")
                    st.stop()
                
                # Extract resume details with experience
                resume_details = extract_resume_details(resume_text, experience_years_value)
                resume_skills = resume_details.get("skills", [])
                st.success(f"✅ Extracted {len(resume_skills)} skills from resume.")

                # Display experience level
                st.info(f"👨‍💼 Experience Level: {experience_level} ({experience_years_value} year{'s' if experience_years_value != 1 else ''})")

                # Interactive Skills Editor
                st.subheader("🛠️ Edit Skills")
                st.write("Review, add, or remove skills before proceeding with the analysis.")
                
                # Create two columns for the skills editor
                edit_col1, edit_col2 = st.columns([2, 1])
                
                with edit_col1:
                    # Initialize session state for skills if not exists
                    if 'edited_skills' not in st.session_state:
                        st.session_state.edited_skills = resume_skills
                    
                    # Convert skills list to a multiline string for editing
                    current_skills = '\n'.join(st.session_state.edited_skills)
                    new_skills = st.text_area(
                        "Current Skills (one per line)",
                        value=current_skills,
                        height=200,
                        help="Edit existing skills or add new ones. Put each skill on a new line."
                    )
                    # Update session state with manually edited skills
                    st.session_state.edited_skills = [s.strip() for s in new_skills.split('\n') if s.strip()]
                
                with edit_col2:
                    st.write("##### Common Skills to Add")
                    # Add buttons for common skills
                    common_skills = {
                        "Languages": ["Python", "Java", "JavaScript", "TypeScript", "C++"],
                        "Web": ["React", "Angular", "Node.js", "HTML/CSS", "Django"],
                        "Cloud": ["AWS", "Azure", "Docker", "Kubernetes", "GCP"],
                        "Data": ["SQL", "MongoDB", "PostgreSQL", "Redis", "Elasticsearch"],
                        "Tools": ["Git", "Jenkins", "Jira", "Docker", "Linux"]
                    }
                    
                    # Create an expander for each category
                    for category, skills in common_skills.items():
                        with st.expander(f"➕ {category}"):
                            # Create columns for better button layout
                            cols = st.columns(2)
                            for idx, skill in enumerate(skills):
                                col = cols[idx % 2]
                                with col:
                                    if st.button(f"Add {skill}", key=f"btn_{category}_{skill}"):
                                        if skill not in st.session_state.edited_skills:
                                            st.session_state.edited_skills.append(skill)
                                            st.rerun()

                # Show current skill count
                st.info(f"📝 Current skill count: {len(st.session_state.edited_skills)}")

                # Show current skills as tags
                st.write("##### Current Skills:")
                skill_cols = st.columns(4)
                for idx, skill in enumerate(sorted(st.session_state.edited_skills)):
                    col = skill_cols[idx % 4]
                    with col:
                        if st.button(f"❌ {skill}", key=f"remove_{skill}"):
                            st.session_state.edited_skills.remove(skill)
                            st.rerun()

                if st.button("▶️ Continue with Analysis", type="primary"):
                    # Run skill gap analysis with updated skills
                    st.session_state.analysis_result = run_skill_gap_analysis(
                        resume_text,
                        f"{job_title} position requiring expertise in various technologies and frameworks.",
                        location.lower(),
                        experience_years_value,
                        {"skills": st.session_state.edited_skills}  # Pass the updated skills
                    )

                # Only display results if analysis has been run
                if st.session_state.analysis_result:
                    # Display results in organized sections
                    st.subheader("📊 Skill Gap Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Match Percentage", f"{st.session_state.analysis_result['skill_match']:.1f}%")
                    with col2:
                        st.metric("Experience Level", st.session_state.analysis_result['experience_level'])
                    with col3:
                        st.metric("Market Trend", st.session_state.analysis_result['market_demand']['trend'])

                    # Display Market Insights
                    st.subheader("📈 Market Insights")
                    market_col1, market_col2 = st.columns(2)
                    with market_col1:
                        st.write("### 🏢 Market Overview")
                        st.write(f"**Current Trend:** {st.session_state.analysis_result['market_demand']['trend']}")
                        st.write(f"**Salary Range:** {st.session_state.analysis_result['market_demand']['salary_range']}")
                        st.write(f"**Experience Level:** {st.session_state.analysis_result['experience_level']}")
                        st.write(f"**Remote Work:** {st.session_state.analysis_result['market_demand'].get('remote_percentage', 'N/A')} of positions")
                        
                        st.write("### 💼 Industry Insights")
                        industry_descriptions = {
                            "Enterprise Software Development": "Leading tech companies building enterprise applications and solutions",
                            "Cloud Services & DevOps": "Companies focused on cloud infrastructure, deployment, and automation",
                            "FinTech & Banking Technology": "Financial institutions and startups developing digital banking solutions",
                            "AI/ML & Data Engineering": "Organizations specializing in artificial intelligence and data platforms",
                            "Cybersecurity & InfoSec": "Companies focused on security solutions and infrastructure protection"
                        }
                        for industry in st.session_state.analysis_result['market_demand']['top_industries'][:5]:
                            st.write(f"- **{industry}**")
                            if industry in industry_descriptions:
                                st.write(f"  {industry_descriptions[industry]}")
                            
                    with market_col2:
                        st.write("### 📍 Top Locations")
                        for loc in st.session_state.analysis_result['market_demand']['top_locations'][:5]:
                            st.write(f"- {loc}")
                        
                        st.write("### 🎯 Most Demanded Skills")
                        for skill in st.session_state.analysis_result['market_demand'].get('top_demanded_skills', [])[:5]:
                            st.write(f"- {skill}")

                    # Display Skill Analysis
                    st.subheader("🔍 Skill Analysis")
                    skill_col1, skill_col2 = st.columns(2)
                    with skill_col1:
                        st.write("### ✅ Matching Skills")
                        for skill in st.session_state.analysis_result.get('matching_skills', []):
                            details = skill['details']
                            st.write(f"**{skill['name']}**")
                            st.write(f"- Level: {details['level'].title()}")
                            st.write(f"- Experience: {details['years']:.1f} years")
                            if details.get('has_certification'):
                                st.write("- ✓ Certified")
                            if details.get('has_leadership'):
                                st.write("- ✓ Leadership Experience")

                    with skill_col2:
                        st.write("### ❌ Missing Skills")
                        for skill in st.session_state.analysis_result.get('missing_skills', []):
                            st.write(f"- {skill}")

                    # Job Recommendations
                    st.subheader("💼 Job Recommendations")
                    recommendations = st.session_state.analysis_result.get('recommendations', {}).get('recommended_roles', [])
                    if recommendations:
                        for job in recommendations:
                            with st.expander(f"🔹 {job['title']} at {job.get('company', 'Unknown')}"):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.write(f"**Location:** {job['location']}")
                                    st.write(f"**Salary:** {job['avg_salary']}")
                                    st.write(f"**Match Score:** {job['match_score']}%")
                                    st.write(f"**Experience Match:** {job['experience_match']}")
                                with col2:
                                    st.write("**Required Skills:**")
                                    for skill in job['required_skills']:
                                        st.write(f"- {skill}")
                                st.write("**Description:**")
                                st.write(job.get('description', 'No description available'))
                                if st.button("Apply Now", key=f"apply_{job['title']}"):
                                    st.markdown(f"[Apply on Job Portal]({job.get('application_link', '#')})")

                    # Learning Recommendations
                    if st.session_state.analysis_result.get('missing_skills'):
                        st.subheader("📚 Learning Recommendations")
                        learning_paths = st.session_state.analysis_result.get('recommendations', {}).get('learning_paths', [])
                        if learning_paths:
                            for path in learning_paths:
                                with st.expander(f"📘 {path['path']} ({path['duration']})"):
                                    st.write("**Courses:**")
                                    for course in path['courses']:
                                        st.write(f"- {course}")
                                    st.write(f"**Certification:** {path['certification']}")
            finally:
                # Ensure temporary file is cleaned up even if an error occurs
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or contact support if the issue persists.")

# Sidebar information and cache control
st.sidebar.info("💡 Upload your resume and enter an aspired role to analyze skill gaps, get job recommendations, and find job listings!")

# Check if we're using a demo key
from utils import API_KEY
if API_KEY == "DEMO_KEY_FOR_TESTING_ONLY":
    st.sidebar.warning("⚠️ **Demo Mode**: Currently running with a demo API key. Some features may be limited.")
    st.sidebar.info("To use all features, provide your Gemini API key in environment variables or Streamlit secrets.")

if st.sidebar.button("♻️ Clear Cache"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Created and maintained by Deep Podder</p>
            <a href="https://github.com/unknown-spec10" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" style="margin-right: 10px;">
            </a>
            <a href="https://www.linkedin.com/in/deeppodder2005" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
