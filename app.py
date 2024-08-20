from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

app = Flask(__name__)

# Load the saved SVM model
with open('models\sentiment_class_svm_vectorizer.pkl', 'rb') as f:
    clf_svm = pickle.load(f)

with open('models\sentiment_class_svm.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# # # Define a function to preprocess the text input
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text


# Define the route for the form page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        #vectorizer = TfidfVectorizer()
        review = request.form['review']
        review = preprocess_text(review)
        test_1_vect = vectorizer.transform([review])
        prediction = clf_svm.predict(test_1_vect)[0]
        # score=clf_svm.score(prediction)
        result = 'Your review was Positive 😊' if prediction == "POSITIVE" else 'Your review was Negative 😞'

    return render_template('home.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
