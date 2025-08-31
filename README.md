## Automatic Resume Matcher
An interactive command-line tool that uses Information Retrieval techniques to rank resumes based on their relevance to a given job description. This project was developed as a practical application of the concepts from the Information Retrieval.
## The Problem
Recruiters often have to manually sift through hundreds of resumes for a single job opening. This process is time-consuming, costly, and prone to human error. This tool aims to automate the initial screening process by providing a ranked list of the most suitable candidates.

## Core Concepts Implemented
This project is built upon fundamental Information Retrieval principles:
- Text Preprocessing (Unit I): Cleaning and preparing raw text for analysis through tokenization, stop-word removal, and stemming.

- Vector Space Model (Unit II): Representing text documents (resumes and job descriptions) as numerical vectors in a multi-dimensional space.

- TF-IDF (Term Frequency-Inverse Document Frequency): A weighting scheme used to score the importance of each word in a document relative to the entire collection of documents.

- Cosine Similarity (Unit II): A metric used to calculate the similarity between the job description vector and each resume vector, resulting in a relevance score.

  ## How to Set Up and Run the Project
Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
   ```
   git clone https://github.com/DEVANSHUMARKAM/Automatic_Resume_Matcher
   
   cd ResumeMatcher
   ```
### 2. Create and Activate a Virtual Environment<br/>
   It's highly recommended to use a virtual environment to manage project dependencies.
   ```
   # Create the environment
   python3 -m venv venv

   Activate it (on Linux/macOS)
   source venv/bin/activate
   ```
### 3. Install Dependencies<br/>
   First, create a requirements.txt file. This file lists all the libraries your project needs. Run this command to generate it automatically:
   ```
   pip freeze > requirements.txt
   ```

   Now, anyone (including you on a new machine) can install the required libraries with one command:
   ```
   pip install -r requirements.txt
   ```
### 4. One-Time NLTK Data Download<br/>
   The Natural Language Toolkit (NLTK) requires some data packages for tokenization and stop words. Run the Python interpreter and download them:
   ```
   python
   ```
   Then, inside Python:
   ```
   >>> import nltk
   >>> nltk.download('punkt')
   >>> nltk.download('stopwords')
   >>> nltk.download('punkt_tab') # Required by newer versions
   >>> exit()
   ```

### 5. Add Sample Resumes<br/>
   Place your sample resumes as .txt files inside the resumes/ directory.

   ## How to Use
   Once the setup is complete, run the main script from your terminal:
   ```
   python matcher.py
   ```
  The program will index the resumes and then prompt you to paste a job description. After you paste the text and press Enter, a ranked list of the top 3 matching resumes with their corresponding similarity scores will be output.
  ```
  --- Top Matching Resumes ---
  Rank 1: resume_fullstack_dev.txt (Score: 0.8841)
  Rank 2: resume_data_analyst.txt (Score: 0.4520)
  Rank 3: resume_human_resource.txt (Score: 0.1098)
  ```
### Type ``` quit ``` to exit the program.

## Future Improvements
Support for .pdf and .docx files: Add functionality to parse different file formats.

Lemmatization: Use lemmatization instead of stemming for more accurate word normalization.

Web Interface: Develop a simple front-end using a framework like Flask or Django to enhance the tool's user-friendliness.






   
