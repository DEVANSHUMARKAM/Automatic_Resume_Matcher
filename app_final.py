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

# --- Configuration ---
UPLOAD_FOLDER = 'resumes'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_secure_secret_key_for_flash_messages' # Needed for flash messages

# --- Global variables to hold the indexed data in memory ---
vectorizer = TfidfVectorizer()
tfidf_matrix = None
resume_files = []
file_index_map = {}

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_text(text):
    """Cleans and preprocesses a piece of text."""
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    return " ".join(stemmed_tokens)

def index_resumes():
    """(Re)Loads and (re)indexes all resumes from the UPLOAD_FOLDER."""
    global vectorizer, tfidf_matrix, resume_files, file_index_map
    
    # Find all .txt files in the resume directory
    resume_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.txt')]
    
    if not resume_files:
        print("WARNING: No resumes found in the 'resumes' directory. The search will not work.")
        # Ensure tfidf_matrix is None if no files are found
        tfidf_matrix = None
        return

    print(f"Indexing {len(resume_files)} resumes...")
    cleaned_resumes = []
    file_index_map = {filepath: i for i, filepath in enumerate(resume_files)}
    
    for filepath in resume_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            cleaned_resumes.append(preprocess_text(file.read()))
    
    # Use the global vectorizer to learn the vocabulary and create the matrix
    tfidf_matrix = vectorizer.fit_transform(cleaned_resumes)
    print("Resumes indexed successfully.")

def perform_search(query_vector, tfidf_matrix, files):
    """Calculates cosine similarity and returns a ranked list."""
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    scores = cosine_similarities.flatten()
    # This new version extracts the filename before returning the list
ranked_results = sorted(zip(scores, files), reverse=True)
# Return score and the base filename, not the full path
return [(score, os.path.basename(filepath)) for score, filepath in ranked_results]

# --- Web Page Routes ---

@app.route('/', methods=['GET', 'POST'])
def search_page():
    """Handles the main search page and relevance feedback."""
    initial_results, refined_results, job_description = [], [], ""
    
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        feedback_resume = request.form.get('feedback_resume', '')

        # Ensure the index exists before trying to search
        if job_description and tfidf_matrix is not None:
            cleaned_jd = preprocess_text(job_description)
            jd_vector = vectorizer.transform([cleaned_jd])
            
            if feedback_resume and feedback_resume in file_index_map:
                # --- Relevance Feedback Logic ---
                relevant_resume_index = file_index_map[feedback_resume]
                relevant_vector = tfidf_matrix[relevant_resume_index]
                
                # Rocchio Algorithm to refine the query vector
                new_jd_vector = 1.0 * jd_vector + 0.75 * relevant_vector
                
                refined_results = perform_search(new_jd_vector, tfidf_matrix, resume_files)
                initial_results = perform_search(jd_vector, tfidf_matrix, resume_files) # Also show initial for comparison
            else:
                # --- Initial Search Logic ---
                initial_results = perform_search(jd_vector, tfidf_matrix, resume_files)

    return render_template('index.html', 
                           initial_results=initial_results,
                           refined_results=refined_results,
                           job_description=job_description)

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    """Handles the file upload page."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'files[]' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        files_uploaded_count = 0
        for file in files:
            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                continue
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                files_uploaded_count += 1
        
        if files_uploaded_count > 0:
            flash(f'{files_uploaded_count} resume(s) uploaded successfully! Re-indexing the database...')
            index_resumes() # Trigger re-indexing
            flash('Re-indexing complete. The system is ready for searching.')
        else:
            flash('No valid .txt files were selected for upload.')
        return redirect(url_for('upload_page'))

    return render_template('upload.html')

# --- Main execution block ---
if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Index all resumes once at the start of the application
    index_resumes()
    
    # Start the Flask web server
    app.run(debug=True)