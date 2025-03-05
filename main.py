import streamlit as st
import pdfplumber
import json
import pytesseract
import pdf2image
import re
import requests

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file, handling NoneType errors."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text.strip() if text else None

def extract_text_with_ocr(pdf_file):
    """Uses OCR to extract text from a scanned PDF file."""
    try:
        pdf_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(pdf_file.read())
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text.strip() if text else None
    except Exception as e:
        st.error(f"OCR error: {e}")
        return None

def generate_json_from_text_fallback(text):
    """
    Uses regex to extract basic details from resume text.
    This fallback extracts email, phone, and assumes the first non-empty line is the name.
    Skills, experience, and education are left empty for you to improve.
    """
    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    phone_pattern = r"(\+?\d[\d\s\-\(\)]{7,}\d)"
    
    email_match = re.search(email_pattern, text)
    phone_match = re.search(phone_pattern, text)
    
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""
    
    # Use the first non-empty line as the name
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    name = lines[0] if lines else ""
    
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": [],
        "experience": [],
        "education": []
    }

def search_job_links(skills):
    """Generates job search links based on extracted skills."""
    job_links = []
    platforms = {
        "Indeed": "https://www.indeed.com/jobs?q={}",
        "LinkedIn": "https://www.linkedin.com/jobs/search?keywords={}",
        "Glassdoor": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword={}"
    }
    for skill in skills:
        for platform, url in platforms.items():
            search_url = url.format(skill.replace(" ", "+"))
            job_links.append((platform, search_url))
    return job_links

def analyze_skill_gap(skills):
    """Compares extracted skills with required skills for job roles using an API."""
    # If you have access to an external API for skill gap analysis, you can integrate it here.
    # For demonstration purposes, we return empty lists.
    return [], []

def suggest_learning_resources(missing_skills):
    """Suggests online courses based on missing skills."""
    course_links = []
    for skill in missing_skills:
        course_links.append(f"https://www.udemy.com/courses/search/?q={skill}")
        course_links.append(f"https://www.coursera.org/search?query={skill}")
    return course_links

st.title("📄 Resume IT")
st.write("Upload a resume (PDF) and extract structured information, analyze skill gaps, and find jobs!")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("File uploaded successfully! Extracting text...")
    uploaded_file.seek(0)
    pdf_text = extract_text_from_pdf(uploaded_file) or extract_text_with_ocr(uploaded_file)
    if pdf_text:
        st.subheader("Extracted Resume Text")
        st.text_area("Resume Content", pdf_text, height=300)
        with st.spinner("Generating structured data..."):
            # Using the regex-based fallback instead of a paid LLM service.
            structured_data = generate_json_from_text_fallback(pdf_text)
        if structured_data:
            st.subheader("📌 Extracted Information")
            st.json(structured_data)
            skills = structured_data.get("skills", [])
            if skills:
                missing_skills, suggested_roles = analyze_skill_gap(skills)
                st.subheader("🔍 Skill Gap Analysis")
                st.write("**Missing Skills:**", missing_skills)
                st.write("**Suggested Roles:**", suggested_roles)
                learning_links = suggest_learning_resources(missing_skills)
                st.subheader("🎓 Suggested Learning Resources")
                for link in learning_links:
                    st.markdown(f"[🔗 Course]({link})")
                job_links = search_job_links(skills)
                st.subheader("💼 Job Opportunities")
                for platform, link in job_links:
                    st.markdown(f"[🔗 {platform} Job Search]({link})")
            json_output = json.dumps(structured_data, indent=4)
            st.download_button("📥 Download JSON", data=json_output, file_name="resume_data.json", mime="application/json")
        else:
            st.error("Failed to generate structured JSON.")
    else:
        st.error("Failed to extract text from the PDF file.")
