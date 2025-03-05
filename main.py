import streamlit as st
import pdfplumber
import json
import pytesseract
import pdf2image
from langchain_community.llms.ollama import Ollama
from langchain_community.llms import Ollama
import tiktoken

# Function to extract text from PDF
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

# Function to extract text from scanned PDFs using OCR
def extract_text_with_ocr(pdf_file):
    """Uses OCR to extract text from a scanned PDF file."""
    try:
        images = pdf2image.convert_from_bytes(pdf_file.read())
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text.strip() if text else None
    except Exception as e:
        st.error(f"OCR error: {e}")
        return None

# Function to truncate text to fit within model token limits
def truncate_text(text, model="glm4:latest"):
    """Ensures text does not exceed model's token limit."""
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        return enc.decode(tokens[:3000])  # Truncate properly
    except Exception:
        return text[:3000]  # Fallback if encoding fails

# Function to generate structured JSON using LLM
def generate_json_from_text(text, model="glm4:latest"):
    """Uses LLM to structure resume text into JSON format."""
    llm = Ollama(model=model)
    
    truncated_text = truncate_text(text, model)

    prompt = f"""
    Extract key details from the following resume text:
    
    {truncated_text}

    Provide the output in valid JSON format with fields:
    - name
    - email
    - phone
    -`skills`: A flat list of strings (e.g., `["Python", "Machine Learning", "Java"]`)
    -`experience`: A list of job roles (e.g., `["Software Engineer", "Data Scientist"]`)
    -`education`: A list of dictionaries, each containing `degree` and `institution`
    -`certifications`: A flat list of strings (e.g., `["AWS Certified", "Google Cloud"]`)
    Ensure the JSON is correctly formatted, without extra text or explanations.
    """

    try:
        response = llm.invoke(prompt).strip()
        structured_data = json.loads(response)  # Convert response to JSON
        return structured_data
    except json.JSONDecodeError:
        st.error(f"LLM response is not valid JSON:\n{response}")  # Show output for debugging
        return None
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

# Function to search job links based on skills
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

# Streamlit UI
st.title("📄 Resume IT")
st.write("Upload a resume (PDF) and extract structured information, including job opportunities based on your skills.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("File uploaded successfully! Extracting text...")

    # Extract text using PDF or OCR
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if not pdf_text:
        st.warning("No text extracted from the PDF. Attempting OCR...")
        uploaded_file.seek(0)  # Reset file pointer for OCR
        pdf_text = extract_text_with_ocr(uploaded_file)

    if not pdf_text:
        st.error("No text extracted. Ensure it's not a scanned document or try another file.")
    else:
        # Show extracted text
        st.subheader("Extracted Resume Text")
        st.text_area("Resume Content", pdf_text, height=300)

        # Process with LLM
        with st.spinner("Generating structured data..."):
            structured_data = generate_json_from_text(pdf_text)

        if structured_data:
            st.subheader("📌 Extracted Information")
            st.json(structured_data)
            
            # Search job links based on skills
            if "skills" in structured_data:
                st.subheader("💼 Job Opportunities")
                job_links = search_job_links(structured_data["skills"])
                for platform, link in job_links:
                    st.markdown(f"[🔗 {platform} Job Search]({link})")

            # Download JSON option
            json_output = json.dumps(structured_data, indent=4)
            st.download_button(
                label="📥 Download JSON",
                data=json_output,
                file_name="resume_data.json",
                mime="application/json"
            )
        else:
            st.error("Failed to generate structured JSON. Please check the LLM response.")
