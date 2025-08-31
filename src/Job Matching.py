%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import altair as alt
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download("punkt_tab")

download_nltk_resources()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('/content/Data_project (1).csv')

df = load_data()

# Load the pre-trained model
model = load('/content/best_rf_model (1).pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def preprocess(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]
    return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

# Handling job requirements
df["Processed_Requirments"] = df["Requirments"].fillna("").apply(preprocess)

# Converting texts to a vector using TF-IDF
vectorizer = TfidfVectorizer()
job_tfidf = vectorizer.fit_transform(df["Processed_Requirments"])

# Initialize LabelEncoders
le_job_title = LabelEncoder()
le_company = LabelEncoder()
le_location = LabelEncoder()
le_country = LabelEncoder()
le_employment = LabelEncoder()
le_experience = LabelEncoder()

# Fit encoders on the data (assuming column names match; adjust if needed)
df['Job Title Encoded'] = le_job_title.fit_transform(df['Job Title'].fillna('Unknown'))
df['Company Name Encoded'] = le_company.fit_transform(df['Company Name: '].fillna('Unknown'))  # Adjusted column name without colon/space
df['Location Encoded'] = le_location.fit_transform(df['Location'].fillna('Unknown'))
df['Country Encoded'] = le_country.fit_transform(df['Country'].fillna('Unknown'))
df['Employment Type Encoded'] = le_employment.fit_transform(df['Employment Type'].fillna('Unknown'))
df['Experience Needed Encoded'] = le_experience.fit_transform(df['Experience Needed'].fillna('Unknown'))

# App Title
st.title("🔍 Job Matching Using ML & Similarity")
st.markdown("Enter your skills and discover matching jobs using AI.")

# User input
raw_skills = st.text_area("Enter your skills (free text):", height=150)

if st.button("Check Matching Jobs"):
    if raw_skills.strip() == "":
        st.warning("Please enter your skills first.")
    else:
        user_input_processed = preprocess(raw_skills)
        user_tfidf = vectorizer.transform([user_input_processed])

        # Cosine similarity
        cosine_scores = cosine_similarity(user_tfidf, job_tfidf)[0]
        df["Cosine_Similarity"] = cosine_scores
        df["Match_Percentage"] = (cosine_scores * 100).round(2)

        # Creating features for the model (using encoded categorical features + Cosine_Similarity)
        features = pd.DataFrame({
            "Job Title Encoded": df['Job Title Encoded'],
            "Company Name Encoded": df['Company Name Encoded'],
            "Location Encoded": df['Location Encoded'],
            "Country Encoded": df['Country Encoded'],
            "Employment Type Encoded": df['Employment Type Encoded'],
            "Experience Needed Encoded": df['Experience Needed Encoded'],
            "Cosine_Similarity": cosine_scores
        })

        # Verify the number of features
        if model.n_features_in_ != features.shape[1]:
            st.error(f"Model expects {model.n_features_in_} features, but provided {features.shape[1]} features.")
        else:
            # Predict the matching percentage through the model
            df["Predicted_Match"] = model.predict(features).round(2)

            # Get top 5 matches
            top_matches = df.sort_values(by="Predicted_Match", ascending=False).head(5)

            if top_matches["Predicted_Match"].max() == 0:
                st.warning("No jobs matched your input skills.")
            else:
                st.success("Top 5 matching jobs predicted!")

                st.subheader("Matching Results (JSON Format):")
                st.json(top_matches[["Job Title", "Predicted_Match"]].rename(columns={"Predicted_Match": "Match Percentage"}).to_dict(orient='records'))

                # Chart
                chart = alt.Chart(top_matches).mark_bar().encode(
                    x=alt.X('Predicted_Match:Q', title="Predicted Match %"),
                    y=alt.Y('Job Title:N', sort='-x'),
                    color=alt.Color('Predicted_Match:Q', scale=alt.Scale(scheme='greens'))
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)