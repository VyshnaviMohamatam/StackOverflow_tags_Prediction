# StackOverflow_tags_Prediction
** Problem Statement **

 StackOverflow is a widely-used platform where developers ask and answer programming-related questions. Each question is usually tagged with relevant topics (like `python`, `deep-learning`, `pandas`, etc.) to help categorize it and make it easily searchable.
The aim of this project is to build a machine learning model that automatically predicts relevant tags for a given StackOverflow question, based on its **title** and **body (description)**.  
This helps in reducing manual effort, improving tag accuracy, and enhancing the discoverability of questions.


## 🧰 Tools Used
- Python
- Pandas, NumPy
- Scikit-learn, NLTK
- TF-IDF, CountVectorizer
- One VS Rest Classifier(Logistic Regression),SGD Classifier
- Streamlit (optional)


## Data Preprocessing 

The raw dataset consists of text data scraped from StackOverflow API, which includes:
- `title`: The short summary or question heading  
- `body`: The full question description  
- `tags`: Target labels (can be multiple)  
  

### Preprocessing Steps:

1. **Text Cleaning**:  
   - Removed HTML tags using `BeautifulSoup`
   - Lowercased all text  
   - Removed special characters and punctuation  

2. **Tokenization & Stop Word Removal**:  
   - Used `nltk.word_tokenize()`  
   - Removed common English stop words (e.g., “is”, “the”, “and”)  

3. **Vectorization**:  
   Converted cleaned text into numerical format using:
   - **TF-IDF Vectorizer**: Calculates importance of each word in a document relative to all documents  
   - **CountVectorizer**: (alternative approach) simply counts word frequencies  

4. **Multi-label Encoding**:  
   Since each question can have multiple tags, we used:
   - `MultiLabelBinarizer()` from Scikit-learn  
   to convert tag lists into a binary matrix format


## 🧠 Model Building 

We experimented with different multi-label classification algorithms:

### 1. **One-vs-Rest with Logistic Regression**
- A separate logistic regression classifier is trained for each tag.
- Each classifier predicts the presence or absence of its corresponding tag.

### 2. **SGDClassifier**
- A linear classifier optimized with stochastic gradient descent.
- Works well on large-scale datasets and sparse data like TF-IDF.

Both models were wrapped in a `OneVsRestClassifier` to support multi-label classification.


## 📈 Evaluation Metrics

###  Precision, Recall & F1-Score (Micro & Macro)
- **Precision**: What proportion of predicted tags were correct?
- **Recall**: What proportion of actual tags were correctly predicted?
- **F1-Score**: Harmonic mean of precision and recall

- **Micro**: Averages across all instances (good for imbalance)
- **Macro**: Averages across all tags equally
