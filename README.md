### Introduction

#### Employs PDF text extraction and NLP preprocessing to extract and prepare text from PDF documents. Utilizes diverse embedding models for numerical representation generation. Through user interaction via a Streamlit interface, enables efficient querying and retrieval of relevant text content from PDFs.

### Methodology

#### -> PDF Text Extraction: The system extracts text from PDF documents using the fitz library and preprocesses it using Natural Language Processing (NLP) techniques.

#### -> Text Preprocessing: Text undergoes preprocessing steps such as tokenization, stop word removal, lemmatization, and punctuation removal to prepare it for analysis.

#### ->Text Embedding Models: Various embedding models including Bag-of-Words, TF-IDF, Word2Vec (CBOW and Skip Gram), GloVe, FastText, BERT, and Sentence Transformers are applied to the preprocessed text to generate numerical representations.

#### -> User Interaction: Users upload a PDF document, input a question, and select an embedding model through a Streamlit interface.

#### -> Answer Generation: The system computes the similarity between the user question and sentences in the document using the chosen embedding model. The most relevant sentences are then presented as answers to the user's query.

### Results

#### The system adeptly retrieves the most relevant sentences from PDF documents based on user queries, ensuring accurate and informative answers.
![image](https://github.com/LoheshM/Document-QA-System/assets/116341584/65212952-44de-4858-853c-159e83554368)
