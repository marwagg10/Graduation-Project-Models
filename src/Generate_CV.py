import streamlit as st
import json
import re
from io import BytesIO
from datetime import date
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.colors import HexColor

from transformers import pipeline
import google.generativeai as genai

# Configure Gemini API
api_key = 'api_key'
if api_key:
    genai.configure(api_key=api_key)

# Load pre-trained summarization model
@st.cache_resource
def load_pretrained_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_pretrained_model()

# Generate merged professional summary
def generate_combined_summary(data, tone):
    raw_text = (
        f"{data['firstName']} {data['lastName']} is a {data['experience']} "
        f"{data['jobTitle']} skilled in {', '.join(data['skills'])}. "
        f"Graduated from {data['education']} and currently works at {data['company']}. "
        f"Fluent in {data['language']}."
    )
    try:
        summary = summarizer(raw_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        summary = f"[Pre-trained Error] {str(e)}"

    prompt = f"""
    Rewrite the following professional CV summary as a single, natural paragraph.
    Maintain a {tone.lower()} tone. Keep it concise, fluent, and polished.

    Input summary:
    \"\"\"{summary}\"\"\"
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-002")
        response = model.generate_content(prompt)
        final_summary = response.text.strip()
    except Exception as e:
        final_summary = f"[Gemini Error] {str(e)}"

    return final_summary

# Streamlit UI
st.title("📄 Create Your Professional CV")

col1, col2 = st.columns(2)

with col1:
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    nationality = st.text_input("Nationality")
    phone = st.text_input("Phone Number")
    dob = st.date_input("Date of Birth", min_value=date(2000, 1, 1), max_value=date.today())
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.text_area("Education", height=100)
    job_title = st.text_input("Job Title")
    company = st.text_input("Current Company")
    hire_date = st.date_input("Hire Date", min_value=date(2000, 1, 1), max_value=date.today())

with col2:
    skills_input = st.text_area("Skills (comma-separated)", height=100)
    experience = st.selectbox("Experience Level", ["Entry", "Mid-level", "Senior"])
    language = st.text_area("Language", height=68)
    github = st.text_input("GitHub Profile")
    linkedin = st.text_input("LinkedIn Profile")
    email = st.text_input("E-mail")

# Summary tone selection
summary_tone = st.selectbox(
    "Select the writing tone for your professional summary:",
    ["Formal", "Creative", "Concise", "Friendly"]
)

# Color picker for PDF
theme_color = st.color_picker("Pick Theme Color for PDF", "#2E86C1")

# Generate CV
if st.button("Generate CV"):
    if not first_name or not last_name or not email:
        st.warning("⚠️ Please fill in at least First Name, Last Name, and E-mail before generating your CV.")
    else:
        # Clean and split skills using both English and Arabic commas
        skills = [skill.strip() for skill in re.split(r",|،", skills_input) if skill.strip()]

        cv_data = {
            "firstName": first_name,
            "lastName": last_name,
            "nationality": nationality,
            "phone": phone,
            "dob": str(dob),
            "gender": gender,
            "education": education,
            "skills": skills,
            "experience": experience,
            "language": language,
            "jobTitle": job_title,
            "company": company,
            "hireDate": str(hire_date),
            "github": github,
            "email": email,
            "linkedin": linkedin
        }

        final_summary = generate_combined_summary(cv_data, summary_tone)

        st.subheader("Professional Summary")
        edited_summary = st.text_area("You can edit the summary before saving:", final_summary, height=250)

        with st.expander("📋 View CV Data (JSON)"):
            st.json(cv_data)

        def generate_pdf(cv_data, summary, theme_color):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            primary_color = HexColor(theme_color)

            c.setFillColor(primary_color)
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(300, 770, "Curriculum Vitae")
            c.setFillColor("black")
            y = 740

            def draw_text(label, value, y_pos):
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(primary_color)
                c.drawString(100, y_pos, label + ":")
                c.setFillColor("black")
                c.setFont("Helvetica", 12)
                wrapped = simpleSplit(value, "Helvetica", 12, 400)
                for line in wrapped:
                    c.drawString(250, y_pos, line)
                    y_pos -= 15
                return y_pos - 10

            y = draw_text("Name", f"{cv_data['firstName']} {cv_data['lastName']}", y)
            y = draw_text("Nationality", cv_data['nationality'], y)
            y = draw_text("Phone", cv_data['phone'], y)
            y = draw_text("Date of Birth", cv_data['dob'], y)
            y = draw_text("Gender", cv_data['gender'], y)
            y = draw_text("Education", cv_data['education'], y)
            y = draw_text("Skills", ", ".join(cv_data['skills']), y)
            y = draw_text("Experience", cv_data['experience'], y)
            y = draw_text("Language", cv_data['language'], y)
            y = draw_text("Job Title", cv_data['jobTitle'], y)
            y = draw_text("Current Company", cv_data['company'], y)
            y = draw_text("Hire Date", cv_data['hireDate'], y)
            y = draw_text("GitHub", cv_data['github'], y)
            y = draw_text("E-mail", cv_data['email'], y)
            y = draw_text("LinkedIn", cv_data['linkedin'], y)

            y -= 20
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(primary_color)
            c.drawString(100, y, "Professional Summary:")
            y -= 20
            c.setFillColor("black")
            c.setFont("Helvetica", 12)
            for line in simpleSplit(summary, "Helvetica", 12, 400):
                c.drawString(100, y, line)
                y -= 15

            c.save()
            buffer.seek(0)
            return buffer

        pdf_buffer = generate_pdf(cv_data, edited_summary, theme_color)
        st.download_button("📥 Download CV as PDF", data=pdf_buffer,
                           file_name=f"{first_name}_{last_name}_CV.pdf", mime="application/pdf")
