import streamlit as st
import json
import streamlit as st
import json
import os
from utils import fetch_job_postings, generate_learning_recommendations, recommend_job_roles
from skill_analysis import (
    extract_text_from_pdf,
    extract_resume_details,
    analyze_skill_gaps,
    extract_job_requirements
)
from update_analysis import update_skill_analysis

# Streamlit UI
st.set_page_config(page_title="Skill Gap Analysis & Job Search", layout="wide")

st.title("🚀 Skill Gap Analysis & Job Search")
st.subheader("Upload your resume and enter your aspired role to analyze skill gaps and find jobs!")

@st.cache_data
def cached_extract_resume_details(resume):
    resume_text = extract_text_from_pdf(resume)
    st.sidebar.write("Extracted text sample (first 500 chars):")
    st.sidebar.write(resume_text[:500] if resume_text else "No text extracted")
    return extract_resume_details(resume_text)
    # In your app.py file, after extracting details:
    extracted_details = cached_extract_resume_details(resume)
    resume_skills = extracted_details.get("skills", [])
    if not resume_skills:
        # Fallback: Use some default skills for testing
        resume_skills = ["Python", "JavaScript", "Data Analysis"]
        st.info("No skills detected automatically. Using test skills for demonstration.")
    st.success(f"✅ Extracted {len(resume_skills)} skills from resume.")

@st.cache_data
def cached_fetch_job_postings(role, skills):
    return fetch_job_postings(role, skills)

@st.cache_data
def cached_analyze_skill_gaps(resume_skills, job_skills, job_title):
    return analyze_skill_gaps(resume_skills, job_skills, job_title)

# User Inputs
job_title = st.text_input("🔹 Aspired Role", placeholder="Enter aspired job role (e.g., Data Scientist)")
resume = st.file_uploader("📝 Upload Resume (PDF)", type=["pdf"])

if resume and job_title:
    with st.spinner("Processing resume and analyzing skills..."):
        extracted_details = cached_extract_resume_details(resume)
        resume_skills = extracted_details.get("skills", [])
        st.success(f"✅ Extracted {len(resume_skills)} skills from resume.")

        # Recommend Job Roles
        st.subheader("🔮 Recommended Job Roles")
        recommended_roles = recommend_job_roles(resume_skills).get("job_recommendations", [])
        if recommended_roles:
            selected_role = st.selectbox("📌 Select a recommended job role", recommended_roles)
        else:
            st.warning("⚠ No job roles found based on resume skills.")
            selected_role = None

        # Refresh Recommendations button
        if st.button("🔄 Refresh Recommendations"):
            st.experimental_rerun()

        # Extract job-related skills
        st.info("Fetching required job skills for analysis...")
        job_skills = extract_job_requirements(job_title)
        if not job_skills:
            st.warning("⚠ No job skills found for the given role. Using an empty comparison.")

        # Perform Skill Gap Analysis
        st.subheader("📊 Skill Gap Analysis")
        skill_gap_analysis = cached_analyze_skill_gaps(resume_skills, job_skills, job_title)
        st.write(f"**Match Percentage:** {skill_gap_analysis['match_percentage']:.2f}%")

        # Display Matched Skills
        if skill_gap_analysis["matched_skills"]:
            st.write("### ✅ Matched Skills")
            for match in skill_gap_analysis["matched_skills"]:
                st.write(f"- {match['job_skill']} (matched with: {match['resume_skill']})")
        else:
            st.write("🚫 No matched skills found.")

        # Display Missing Skills
        if skill_gap_analysis["missing_skills"]:
            st.write("### ❌ Missing Skills")
            for skill in skill_gap_analysis["missing_skills"]:
                st.write(f"- {skill}")
        else:
            st.write("🚫 No missing skills found.")

        # Generate Learning Recommendations
        recommendations = generate_learning_recommendations(skill_gap_analysis["missing_skills"])
        if recommendations.get("recommendations"):
            st.write("### 📌 Top Learning Recommendations")
            for rec in recommendations["recommendations"][:3]:
                st.write(f"- {rec['skill']}: {rec['course']}")

        # Fetch Job Postings for Aspired Role
        st.subheader("💼 Job Opportunities for Aspired Role")
        with st.spinner("Fetching job postings for aspired role..."):
            job_results = cached_fetch_job_postings(job_title, resume_skills)
        if job_results and job_results.get("job_matches"):
            st.success(f"✅ Found {len(job_results['job_matches'])} job listings!")
            for job in job_results["job_matches"]:
                with st.expander(f"🔹 {job['title']} at {job['company']}"):
                    st.write(f"**Location:** {job['location']}")
                    st.write(f"**Salary Range:** {job['salary_range']}")
                    st.write(f"**Description:** {job['description'][:200]}...")
                    st.markdown(f"[Apply Here]({job['application_link']})", unsafe_allow_html=True)
        else:
            st.warning("⚠ No job postings found for aspired role.")

        # Fetch Job Postings for Recommended Role
        if selected_role:
            selected_role_title = selected_role.get("title", "") if isinstance(selected_role, dict) else selected_role
            st.subheader(f"💼 Job Opportunities for {selected_role_title}")
            with st.spinner(f"Fetching job postings for {selected_role_title}..."):
                recommended_job_results = cached_fetch_job_postings(selected_role_title, resume_skills)
            if recommended_job_results and recommended_job_results.get("job_matches"):
                st.success(f"✅ Found {len(recommended_job_results['job_matches'])} job listings for {selected_role_title}!")
                for job in recommended_job_results["job_matches"]:
                    with st.expander(f"🔹 {job['title']} at {job['company']}"):
                        st.write(f"**Location:** {job['location']}")
                        st.write(f"**Salary Range:** {job['salary_range']}")
                        st.write(f"**Description:** {job['description'][:200]}...")
                        st.markdown(f"[Apply Here]({job['application_link']})", unsafe_allow_html=True)
            else:
                st.warning(f"⚠ No job postings found for {selected_role_title}.")

    # Update Analysis Section
    st.subheader("🔄 Refine Analysis")
    if st.button("Update Analysis"):
        with st.spinner("Refining your analysis..."):
            try:
                update_skill_analysis()  # This updates the analysis and writes to updated_skill_gap_analysis.json
                with open("updated_skill_gap_analysis.json", "r", encoding="utf-8") as f:
                    updated_analysis = json.load(f)
                st.success("✅ Analysis updated successfully!")
                st.write("### Updated Skill Gap Analysis")
                st.write(f"**Updated Match Percentage:** {updated_analysis['skill_gap_analysis']['match_percentage']:.2f}%")
                if updated_analysis["skill_gap_analysis"]["matched_skills"]:
                    st.write("#### Updated Matched Skills")
                    for match in updated_analysis["skill_gap_analysis"]["matched_skills"]:
                        st.write(f"- {match['job_skill']} (matched with: {match['resume_skill']})")
                else:
                    st.write("No updated matched skills found.")
                if updated_analysis["skill_gap_analysis"]["missing_skills"]:
                    st.write("#### Updated Missing Skills")
                    for skill in updated_analysis["skill_gap_analysis"]["missing_skills"]:
                        st.write(f"- {skill}")
                else:
                    st.write("No updated missing skills found.")
                if "job_role_recommendations" in updated_analysis:
                    st.write("### Updated Recommended Job Roles")
                    for role in updated_analysis["job_role_recommendations"].get("job_recommendations", []):
                        st.write(f"- {role.get('title', 'N/A')}: {role.get('description', '')}")
                if "learning_recommendations" in updated_analysis:
                    st.write("### Updated Learning Recommendations")
                    for rec in updated_analysis["learning_recommendations"].get("recommendations", []):
                        st.write(f"- {rec.get('skill', 'N/A')}: {rec.get('course', '')}")
            except Exception as e:
                st.error(f"Error updating analysis: {e}")

st.sidebar.info("💡 Upload your resume and enter an aspired role to analyze skill gaps, get job recommendations, and find job listings!")
if st.sidebar.button("♻️ Clear Cache"):
    st.cache_data.clear()
    st.experimental_rerun()
