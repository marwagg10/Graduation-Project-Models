import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nltk.download('stopwords')

# Load SBERT model
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Load English NER model
nlp = spacy.load("en_core_web_sm")


# Google Gemini API Key
api = 'Api_key'
if api:
    genai.configure(api_key=api)
else:
    st.error("API Key not found.")


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip() if text.strip() else "No readable text found in the PDF."

# Skill extractor (General, No Hardcoded Words)
def extract_dynamic_skills(text):
    stop_words = set(stopwords.words('english'))

    # Run spaCy NER
    doc = nlp(text)
    entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "TECH", "LANGUAGE"]]

    # TF-IDF keywords
    words = [w.lower() for w in re.findall(r'\b\w{3,}\b', text) if w.lower() not in stop_words]
    text_cleaned = ' '.join(words)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20)
    vectorizer.fit_transform([text_cleaned])
    tfidf_keywords = list(vectorizer.get_feature_names_out())

    # Combine and filter short words only
    skills = set(entities + tfidf_keywords)
    skills = [s for s in skills if len(s) > 2]

    return sorted(set(skills))

# Clean and validate JSON output
def safe_json_parse(text):
    cleaned = re.sub(r'```json|```', '', text).strip()
    cleaned = re.sub(r'“|”', '"', cleaned)
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "error": "JSON parsing error",
            "raw_content": text
        }

# Gemini evaluation
def evaluate_resume(resume_text, job_description):
    prompt = f"""
    Evaluate this resume against the job description.
    Return only valid JSON in this format:
    {{
        "match_percentage": int,
        "missing_skills": [str],
        "strengths": [str],
        "improvements": [str]
    }}

    Job Description:
    {job_description}

    Resume:
    {resume_text}
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-002")
        response = model.generate_content(prompt)
        if hasattr(response, "candidates") and response.candidates:
            raw = response.candidates[0].content.parts[0].text
            return safe_json_parse(raw)
        return {"error": "No response from Gemini"}
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.set_page_config(page_title="ATS Resume Evaluation", page_icon="📄")
st.title("📄🔍 ATS Resume Evaluation System")
st.write("Upload your resume (PDF) and paste the job description to get a detailed AI-powered evaluation.")

job_description = st.text_area("Job Description")
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Evaluate Resume"):
    if uploaded_file and job_description:
        with st.spinner("Processing..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            jd_skills = extract_dynamic_skills(job_description)
            resume_skills = extract_dynamic_skills(resume_text)
            evaluation = evaluate_resume(resume_text, job_description)

        st.subheader("Extracted Resume Text")
        st.write(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

        st.subheader("AI Evaluation")
        if "error" in evaluation:
            st.subheader("Raw Output")
            st.code(evaluation.get("raw_content", ""), language="json")
        else:
            st.json(evaluation)



        st.subheader("Skill Match using Sentence-BERT")
        st.write("**Extracted Job Skills:**", jd_skills)
        st.write("**Extracted Resume Skills:**", resume_skills)


    else:
        st.error("Please upload a resume and enter the job description.")
