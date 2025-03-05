import streamlit as st
import pdfplumber
import json
import pytesseract
import pdf2image
import requests
import openai

# Set your OpenAI API key from Streamlit secrets or hardcode it for testing.
# For production, please use st.secrets to avoid exposing sensitive keys.
openai.api_key = st.secrets.get("OPENAI_API_KEY", "your-openai-api-key")

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

def generate_json_from_text(text, model="gpt-3.5-turbo"):
    """Uses OpenAI's ChatCompletion API to structure resume text into JSON format."""
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
Ensure the JSON is correctly formatted without any extra text.
"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        output = response["choices"][0]["message"]["content"].strip()
        structured_data = json.loads(output)
        return structured_data
    except json.JSONDecodeError:
        st.error(f"OpenAI response is not valid JSON:\n{output}")
        return None
