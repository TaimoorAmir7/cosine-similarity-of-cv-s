from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

app = Flask(__name__)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resumes = []
        for pdf in request.files.getlist('resumes'):
            if pdf and pdf.filename.endswith('.pdf'):
                text = extract_text_from_pdf(pdf)
                resumes.append(text)

        # Preprocessing and vectorization
        vectorizer = TfidfVectorizer().fit_transform(resumes + [job_description])
        vectors = vectorizer.toarray()

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(vectors[:-1], vectors[-1].reshape(1, -1)).flatten()

        # Combine the results
        results = list(zip(resumes, cosine_similarities))

        return render_template('index.html', results=results, job_description=job_description)

    return render_template('index.html', results=[], job_description="")

if __name__ == "__main__":
    app.run(debug=True)
