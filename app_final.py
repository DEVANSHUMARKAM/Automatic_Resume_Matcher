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
import PyPDF2 # NEW: For PDF text extraction

UPLOAD_FOLDER = 'resumes'
# --- MODIFIED: Allow both txt and pdf files for upload 
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_secure_secret_key_for_flash_messages'

vectorizer = TfidfVectorizer()
tfidf_matrix = None
resume_files = []
file_index_map = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- NEW: Function to extract text from a PDF file ---
def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
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

def index_resumes():
    global vectorizer, tfidf_matrix, resume_files, file_index_map
    # We only index .txt files, as PDFs are converted to .txt upon upload
    resume_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.txt')]
    if not resume_files:
        print("WARNING: No resumes found.")
        tfidf_matrix = None
        return
    print(f"Indexing {len(resume_files)} resumes...")
    cleaned_resumes = []
    file_index_map = {filepath: i for i, filepath in enumerate(resume_files)}
    for filepath in resume_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            cleaned_resumes.append(preprocess_text(file.read()))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_resumes)
    print("Resumes indexed successfully.")

def perform_search(query_vector, tfidf_matrix, files):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    scores = cosine_similarities.flatten()
    ranked_results = sorted(zip(scores, files), reverse=True)
    return [(score, os.path.basename(filepath)) for score, filepath in ranked_results]

# --- Web Page Routes ---
@app.route('/', methods=['GET', 'POST'])
def search_page():
    initial_results, refined_results, job_description = [], [], ""
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        feedback_resume_filename = request.form.get('feedback_resume', '')
        if job_description and tfidf_matrix is not None:
            cleaned_jd = preprocess_text(job_description)
            jd_vector = vectorizer.transform([cleaned_jd])
            feedback_resume_full_path = os.path.join(app.config['UPLOAD_FOLDER'], feedback_resume_filename)
            if feedback_resume_filename and feedback_resume_full_path in file_index_map:
                relevant_resume_index = file_index_map[feedback_resume_full_path]
                relevant_vector = tfidf_matrix[relevant_resume_index]
                new_jd_vector = 1.0 * jd_vector + 0.75 * relevant_vector
                refined_results = perform_search(new_jd_vector, tfidf_matrix, resume_files)
                initial_results = perform_search(jd_vector, tfidf_matrix, resume_files)
            else:
                initial_results = perform_search(jd_vector, tfidf_matrix, resume_files)
    return render_template('index.html', initial_results=initial_results, refined_results=refined_results, job_description=job_description)

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        files_uploaded_count = 0
        for file in files:
            if file.filename == '':
                continue
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # --- MODIFIED: Handle PDF and TXT files differently ---
                if filename.lower().endswith('.pdf'):
                    # For PDFs, extract text and save as .txt
                    text_content = extract_text_from_pdf(file.stream)
                    if text_content:
                        # Create a new filename with a .txt extension
                        txt_filename = os.path.splitext(filename)[0] + ".txt"
                        save_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
                        with open(save_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(text_content)
                        files_uploaded_count += 1
                else:
                    # For .txt files, save them directly
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    files_uploaded_count += 1
        
        if files_uploaded_count > 0:
            flash(f'{files_uploaded_count} resume(s) processed successfully! Re-indexing...')
            index_resumes() # Trigger re-indexing
            flash('Re-indexing complete.')
        else:
            flash('No valid .txt or .pdf files were selected.')
        return redirect(url_for('upload_page'))
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    index_resumes()
    app.run(debug=True)

