# Financial Complaint Management System

## 1. About the Dataset
This project focuses on developing an automated system for managing financial complaints. The dataset consists of text data from financial complaints, which is used to train a model for classifying and routing complaints based on their content.

## 2. Objective
The primary objectives of this project are to:
- Develop an automated classification system to categorize financial complaints using Natural Language Processing (NLP) techniques.
- Route complaints efficiently based on their classification.
- Create a user-friendly web application to interact with the classification system using Streamlit.

## 3. Approach
The project employs the following approach:
- **Data Preprocessing:**
  - Applied Near-Miss undersampling method to address class imbalance.
  - Performed tokenization, stopword removal, stemming, and lemmatization to clean and prepare the text data.
- **Feature Extraction:**
  - Transformed text into vector representations using techniques such as Bag of Words (BoW), n-Grams, TF-IDF, and Word2Vec.
- **Model Training:**
  - Utilized various classification models including:
    - Logistic Regression (One-vs-Rest, OVR)
    - Naive Bayes
  - Employed Recurrent Neural Networks (RNN) and Bidirectional Long Short-Term Memory (Bi-LSTM) models.
  - Applied word embeddings, dropout, batch normalization, and stratified K-fold Cross-Validation (CV) for model evaluation.

## 4. Results & Key Features
- **Accuracy:** Achieved 95% accuracy with the Bi-LSTM model, indicating high performance in classifying and routing complaints.
- **Evaluation:** Obtained strong results in the confusion matrix, demonstrating effective classification.
- **Web Application:** Developed an interactive web application using Streamlit to allow users to interact with the classification system.

## 5. License
This project is licensed under the MIT License.
## 6. Contact
For any questions or feedback regarding this project, please contact:
- **Vivek Radadiya**
