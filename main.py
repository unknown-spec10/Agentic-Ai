import streamlit as st
import pdfplumber
import json
import pytesseract
import pdf2image
import requests
from langchain_community.llms.ollama import Ollama
# Removed duplicate import of Ollama

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
        # Reset file pointer
        pdf_file.seek(0)
        images = pdf2image.convert_from_bytes(pdf_file.read())
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text.strip() if text else None
    except Exception as e:
        st.error(f"OCR error: {e}")
        return None

def generate_json_from_text(text, model="glm4:latest"):
    """Uses LLM to structure resume text into JSON format."""
    llm = Ollama(model=model)
    prompt = f"""
    Extract key details from the following resume text:
    {text}
    Provide the output in valid JSON format with fields:
    - name
    - email
    - phone
    - skills (list)
    - experience (list of job roles)
    - education (list of degrees)
    """
    try:
        response = llm.invoke(prompt).strip()
        structured_data = json.loads(response)
        return structured_data
    except json.JSONDecodeError:
        st.error(f"LLM response is not valid JSON:\n{response}")
        return None

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
    # Hardcoded API endpoint and API key for demonstration purposes.
    api_url = "https://api.groq.com/match-jobs"
    api_key = "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"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"skills": skills}
    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("missing_skills", []), response_data.get("suggested_roles", [])
    except Exception as e:
        st.error(f"Error fetching skill gap analysis: {e}")
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
            structured_data = generate_json_from_text(pdf_text)
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
