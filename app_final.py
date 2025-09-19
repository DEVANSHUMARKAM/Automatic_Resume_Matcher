from flask import Flask, render_template, request, redirect, url_for, flash
import os
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from werkzeug.utils import secure_filename
import PyPDF2
import xml.etree.ElementTree as ET
# NEW: Import libraries for the web crawler
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
UPLOAD_FOLDER = 'resumes_sample' 
STRUCTURED_FOLDER = 'structured_resumes'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STRUCTURED_FOLDER'] = STRUCTURED_FOLDER
app.secret_key = 'a_very_secure_secret_key'

# --- Global variables ---
vectorizer = TfidfVectorizer()
tfidf_matrix = None
all_files = []
file_index_map = {}
structured_data_cache = {}

# --- Helper Functions (remain the same) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    return " ".join(stemmed_tokens)

def index_all_resumes():
    global vectorizer, tfidf_matrix, all_files, file_index_map, structured_data_cache
    txt_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.txt')]
    xml_files = [os.path.join(app.config['STRUCTURED_FOLDER'], f) for f in os.listdir(app.config['STRUCTURED_FOLDER']) if f.endswith('.xml')]
    all_files = txt_files + xml_files
    if not all_files:
        tfidf_matrix = None
        return
    print(f"Indexing {len(all_files)} total resumes...")
    cleaned_resumes, structured_data_cache = [], {}
    file_index_map = {filepath: i for i, filepath in enumerate(all_files)}
    for filepath in xml_files:
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            skills = [skill.text.lower() for skill in root.findall('./skills/skill')]
            exp_years = int(root.find('./experience/years').text)
            structured_data_cache[filepath] = {'skills': skills, 'experience_years': exp_years}
        except Exception as e:
            print(f"Could not parse {filepath}: {e}")
    for filepath in all_files:
        content = ""
        if filepath.endswith('.xml'):
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                content = root.find('full_text').text or ""
            except Exception:
                content = ""
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        cleaned_resumes.append(preprocess_text(content))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_resumes)
    print("All resumes indexed successfully.")

def perform_search(query_vector, matrix, files):
    cosine_similarities = cosine_similarity(query_vector, matrix)
    scores = cosine_similarities.flatten()
    ranked_results = sorted(zip(scores, files), reverse=True)
    return [(score, os.path.basename(filepath)) for score, filepath in ranked_results]

# --- Web Page Routes ---

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/search', methods=['GET', 'POST'])
def search_page():
    initial_results, refined_results, job_description = [], [], ""
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        feedback_resume_name = request.form.get('feedback_resume', '')
        if job_description and tfidf_matrix is not None:
            cleaned_jd = preprocess_text(job_description)
            jd_vector = vectorizer.transform([cleaned_jd])
            feedback_resume_full_path = next((f for f in all_files if os.path.basename(f) == feedback_resume_name), None)
            if feedback_resume_full_path and feedback_resume_full_path in file_index_map:
                relevant_resume_index = file_index_map[feedback_resume_full_path]
                relevant_vector = tfidf_matrix[relevant_resume_index]
                new_jd_vector = 1.0 * jd_vector + 0.75 * relevant_vector
                refined_results = perform_search(new_jd_vector, tfidf_matrix, all_files)
                initial_results = perform_search(jd_vector, tfidf_matrix, all_files)
            else:
                initial_results = perform_search(jd_vector, tfidf_matrix, all_files)
    return render_template('search.html', initial_results=initial_results, refined_results=refined_results, job_description=job_description)

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        files_uploaded_count = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                    txt_filename = filename.rsplit('.', 1)[0] + '.txt'
                    txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
                    with open(txt_filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
                    os.remove(filepath)
                files_uploaded_count += 1
        if files_uploaded_count > 0:
            flash(f'{files_uploaded_count} file(s) processed! Re-indexing...')
            index_all_resumes()
            flash('Re-indexing complete.')
        else:
            flash('No valid files were selected.')
        return redirect(url_for('upload_page'))
    return render_template('upload.html')

@app.route('/add-structured', methods=['GET', 'POST'])
def add_structured_page():
    if request.method == 'POST':
        name = request.form.get('name')
        exp_years = request.form.get('experience_years')
        skills = request.form.get('skills')
        full_text = request.form.get('full_text')
        resume_root = ET.Element('resume')
        personal_info = ET.SubElement(resume_root, 'personal_info')
        ET.SubElement(personal_info, 'name').text = name
        skills_root = ET.SubElement(resume_root, 'skills')
        for skill in skills.split(','):
            ET.SubElement(skills_root, 'skill').text = skill.strip()
        experience = ET.SubElement(resume_root, 'experience')
        ET.SubElement(experience, 'years').text = exp_years
        ET.SubElement(resume_root, 'full_text').text = full_text
        tree = ET.ElementTree(resume_root)
        ET.indent(tree, space="\t", level=0)
        safe_name = secure_filename(name).replace(' ', '_')
        filename = os.path.join(app.config['STRUCTURED_FOLDER'], f"{safe_name}_resume.xml")
        tree.write(filename, encoding='unicode')
        flash(f'Successfully created structured profile for {name}. Re-indexing...')
        index_all_resumes()
        flash('Re-indexing complete.')
        return redirect(url_for('add_structured_page'))
    return render_template('add_structured_resume.html')

# --- NEW: Route for the Web Crawler ---
@app.route('/crawler', methods=['GET', 'POST'])
def crawler_page():
    jobs = []
    if request.method == 'POST':
        url = request.form.get('url')
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a VERY SIMPLE scraping rule. It might need to be adjusted
            # for different websites. It looks for <h2> tags for titles and
            # the next sibling element for the description.
            for title_tag in soup.find_all('h2'):
                title = title_tag.get_text(strip=True)
                description_tag = title_tag.find_next_sibling()
                description = ""
                if description_tag:
                    description = description_tag.get_text(strip=True, separator='\n')
                
                if title and description:
                    jobs.append({'title': title, 'description': description})
            
            if not jobs:
                flash("Could not find any jobs with the current scraping rule. The website structure might be different.", "error")
            else:
                flash(f"Successfully scraped {len(jobs)} jobs!", "success")

        except requests.exceptions.RequestException as e:
            flash(f"Error fetching the URL: {e}", "error")
    
    return render_template('crawler.html', jobs=jobs)

# --- Main execution block ---
if __name__ == '__main__':
    for folder in [UPLOAD_FOLDER, STRUCTURED_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    index_all_resumes()
    app.run(debug=True)

