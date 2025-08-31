import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import google.generativeai as genai
import json
import os
import re
import subprocess
from PIL import Image, ImageEnhance, ImageFilter

#load pytesseract model
import pytesseract

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    st.warning("Downloading spaCy model 'en_core_web_lg'...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

# Initialize SkillExtractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# Gemini API
api = 'api_key'
genai.configure(api_key=api)

# Preprocess text
def preprocess_text(text):
    clean_text = re.sub(r'[^A-Za-z0-9.,;:()\- ]+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip().lower()

# Gemini skills
def get_skills_from_gemini(job_description):
    prompt = f"Extract only the skills from the following job description:\n{job_description}"
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-002")
        response = model.generate_content(prompt)
        skills = response.text.strip().split("\n")
        return [s.strip("*•- ").strip() for s in skills if s.strip()]
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return []

# SkillNer skills
def extract_skills(job_description):
    extracted_skills = skill_extractor.annotate(job_description)
    return [match['doc_node_value'] for match in extracted_skills['results'].get('soft_matches', [])]

# OCR from image (enhanced)
def extract_text_from_image(image_file):
    image = Image.open(image_file).convert('L')  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)  # Sharpen image
    image = ImageEnhance.Contrast(image).enhance(2)  # Increase contrast
    text = pytesseract.image_to_string(image)
    return text

# Streamlit UI
st.title("📄 AI-Powered Job Post Skill Extractor")

option = st.radio("Choose input method:", ("Text", "Image"))

if option == "Text":
    job_post = st.text_area("Enter the Jop Posting here :")
elif option == "Image":
    uploaded_image = st.file_uploader("Upload job post image (PNG/JPG):", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        extracted_text = extract_text_from_image(uploaded_image)
        st.text_area("Extracted Text from Image:", extracted_text, height=200)
        job_post = extracted_text
    else:
        job_post = ""

if st.button("Extract Skills"):
    if job_post.strip():
        cleaned_text = preprocess_text(job_post)
        gemini_skills = get_skills_from_gemini(cleaned_text)
        skillner_skills = extract_skills(cleaned_text)
        final_skills = sorted(set(gemini_skills + skillner_skills))

        st.subheader("✅ Extracted Skills:")
        st.json({"extracted_skills": final_skills})
    else:
        st.warning("Please enter or upload a job description.")
