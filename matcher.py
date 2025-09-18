import os
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """
    Cleans and preprocesses a piece of text by tokenizing, removing stop words,
    and stemming.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            cleaned_tokens.append(token)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    return " ".join(stemmed_tokens)

if __name__ == '__main__':
    resumes_path = "resumes/"
    resume_files = [os.path.join(resumes_path, f) for f in os.listdir(resumes_path) if f.endswith('.txt')]

    if not resume_files:
        print(f"No .txt files found in the '{resumes_path}' directory. Please add sample resumes.")
    else:
        print(f"Reading and processing {len(resume_files)} resumes...")
        cleaned_resumes = []
        for filepath in resume_files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                cleaned_resumes.append(preprocess_text(file.read()))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_resumes)
        print("Resumes have been indexed successfully.")

        print("\n--- Resume Matcher Activated ---")
        while True:
            # 1. Get user input for the job description
            job_description = input("\nPlease paste the job description (or type 'quit' to exit):\n> ")

            if job_description.lower() == 'quit':
                break

            # 2. Preprocess and vectorize the job description
            cleaned_jd = preprocess_text(job_description)
            jd_vector = vectorizer.transform([cleaned_jd])

            # 3. Calculate similarity and rank
            cosine_similarities = cosine_similarity(jd_vector, tfidf_matrix)
            scores = cosine_similarities.flatten()
            ranked_resumes = sorted(zip(scores, resume_files), reverse=True)

            # 4. Print the top results
            print("\n--- Top Matching Resumes ---")
            for i, (score, resume_file) in enumerate(ranked_resumes[:3]):
                print(f"Rank {i+1}: {os.path.basename(resume_file)} (Score: {score:.4f})")
            print("----------------------------------------")