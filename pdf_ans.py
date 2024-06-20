import fitz
import nltk
import torch
import os
import tempfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Preprocess for Skipgram and CBOW
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if not (token.isdigit() or (token[:-1].isdigit() and token[-1] == '.'))]
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)) for token, pos_tag in pos_tags]
    lemmatized_tokens = [token for token in lemmatized_tokens if token] 
    return lemmatized_tokens

def preprocess_alt_text(text):

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = []

    for sentence in text:
        if sentence:
            s = ''
            for word, pos_tag in nltk.pos_tag(word_tokenize(sentence)):
                if not any(char.isdigit() for char in word) and word.lower() not in stop_words:
                    word_without_punct = ''.join(char for char in word if char not in string.punctuation)
                    pos_tag = get_wordnet_pos(pos_tag)
                    lemma = lemmatizer.lemmatize(word_without_punct, pos=pos_tag)
                    s += lemma.lower() + ' '
            tokens.append(s.strip())  
        
    return tokens

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  
    elif treebank_tag.startswith('V'):
        return 'v'  
    elif treebank_tag.startswith('N'):
        return 'n' 
    elif treebank_tag.startswith('R'):
        return 'r'  
    else:
        return 'n'

# PDF-Text Extraction 
def process_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    preprocessed_text_corpus = []
    altt_corpus=[]
    alt_corpus=[]
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        page_text = page.get_text()

        sentences = nltk.sent_tokenize(page_text)
        alt_corpus=preprocess_alt_text(sentences)
        altt_corpus+=alt_corpus
        preprocessed_page_text = []
        for sentence in sentences:
            preprocessed_sentence = preprocess_text(sentence)
            preprocessed_page_text.append((sentence, preprocessed_sentence))
        preprocessed_text_corpus.append(preprocessed_page_text)
    pdf_document.close()
    return preprocessed_sentence,preprocessed_text_corpus,altt_corpus

def get_sentence_embedding(sentence_tokens, model):
    word_embeddings = []
    for token in sentence_tokens:
        if token in model.wv.key_to_index:  
            word_embeddings.append(model.wv[token])
    
    if len(word_embeddings) == 0:
        return None
    
    sentence_embedding = sum(word_embeddings) / len(word_embeddings)
    return sentence_embedding


# Finding most Relevant Sentence for Word2Vec Model
def find_most_relevant_sentences_using_word_embeddings(question, corpus, model, top_n=3):
    preprocessed_question = preprocess_text(question)
    question_embedding = np.zeros((1, model.vector_size))  
    count = 0 

    for token in preprocessed_question:
        if token in model.wv.key_to_index:
            question_embedding += model.wv[token]
            count += 1

    if count == 0:
        return "Unable to find relevant sentences."

    question_embedding /= count  

    top_sentences = []
    for sentence_tokens, original_sentence in corpus:
        sentence_embedding = np.zeros((1, model.vector_size))  
        count = 0 

        for token in sentence_tokens:
            if token in model.wv.key_to_index:
                sentence_embedding += model.wv[token]
                count += 1

        if count > 0:
            sentence_embedding /= count
            similarity = cosine_similarity(question_embedding, sentence_embedding)
            top_sentences.append((original_sentence, similarity))

    top_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = top_sentences[:top_n]

    return top_sentences


# Finding most Relevant Sentence for BERT
BERT_Model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(BERT_Model)
model_bert = AutoModel.from_pretrained(BERT_Model)
def sent_embedding(sent):
    tokens = tokenizer.encode_plus(sent, max_length=128, truncation=True,
                                    padding='max_length', return_tensors='pt')
    with torch.no_grad():
        outputs = model_bert(**tokens)
        embedding = outputs.pooler_output.detach().numpy()
    return embedding

def calculate_similarity(sent_embedding1, sent_embedding2):
    sent_embedding1 = torch.tensor(sent_embedding1)
    sent_embedding2 = torch.tensor(sent_embedding2)
    return torch.nn.functional.cosine_similarity(sent_embedding1, sent_embedding2).item()

def find_most_relevant_sentences_using_bert(question, corpus, model, top_n=3):
    question_embedding = sent_embedding(question)

    top_sentences = []
    for sentence_tokens, original_sentence in corpus:
        sentence = ' '.join(sentence_tokens)
        sentence_embedding = sent_embedding(sentence)
        similarity = calculate_similarity(question_embedding, sentence_embedding)
        top_sentences.append((original_sentence, similarity))

    top_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = top_sentences[:top_n]

    return top_sentences


def capitalize_first_letter(sentence):
    if sentence:
        return sentence[0].upper() + sentence[1:]
    return ""  


def save_uploaded_pdf(uploaded_file):
    """
    Save the uploaded PDF file temporarily and return the file path.
    """
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    st.title("PDF-Answering-AI")
    st.write("Upload a PDF document")

    # Upload PDF document
    uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")

    if uploaded_file is not None:
        pdf_path = save_uploaded_pdf(uploaded_file)

        # Process PDF to extract text
        preprocessed_sentence, preprocessed_text_corpus, altt_corpus = process_pdf(pdf_path)
        corpus = []
        corpuss = []
        for page_corpus in preprocessed_text_corpus:
            for sentence, tokens in page_corpus:
                corpuss.append((tokens, sentence))
                corpus.append(tokens)

        # Choose embedding model
        embedding_model = st.selectbox("Choose an Embedding Model", ["CBOW", "Skip Gram", "GloVe", "FastText","BERT"])

        # Ask user question
        user_question = st.text_input("Ask your question")

        if st.button("Get Answer"):
            if embedding_model == "CBOW":
                model_cbow = Word2Vec(corpus, min_count=1, vector_size=60, window=2, sg=0)
                model_cbow.train(corpus, total_examples=len(corpus), epochs=250)
                top_relevant_sentences = find_most_relevant_sentences_using_word_embeddings(user_question, corpuss, model_cbow, top_n=3)
            elif embedding_model == "Skip Gram":
                model_skip = Word2Vec(corpus, min_count=1, vector_size=60, window=2, sg=1)
                model_skip.train(corpus, total_examples=len(corpus), epochs=200)
                top_relevant_sentences = find_most_relevant_sentences_using_word_embeddings(user_question, corpuss, model_skip, top_n=3)
            elif embedding_model == "GloVe":
                model_glo = Word2Vec(sentences=corpus, min_count=1, vector_size=60, window=2, sg=0)
                model_glo.train(corpus, total_examples=len(corpus), epochs=250)
                top_relevant_sentences = find_most_relevant_sentences_using_word_embeddings(user_question, corpuss, model_glo, top_n=3)
            elif embedding_model == "FastText":
                model_fast = FastText(sentences=corpus, vector_size=60, window=5, min_count=1, sg=1)
                model_fast.train(corpus, total_examples=len(corpus), epochs=250)
                top_relevant_sentences = find_most_relevant_sentences_using_word_embeddings(user_question, corpuss, model_fast, top_n=3)
            elif embedding_model == "BERT":
                top_relevant_sentences = find_most_relevant_sentences_using_bert(user_question, corpuss, model_bert, top_n=3)
            else:
                st.error("Invalid embedding model selected.")

            # Display answer
            required_words = ["In addition", "Moreover"]
            modified_content = ""
            if not top_relevant_sentences:
                modified_content = "Unable to find relevant sentences."
            elif top_relevant_sentences == "Unable to find relevant sentences.":
                modified_content = top_relevant_sentences
            else:
                for i, (sentence, _) in enumerate(top_relevant_sentences):
                    sentence = sentence.replace('\n', '')
                    modified_sentence = capitalize_first_letter(' '.join(sentence.split()))
                    if modified_sentence[-1] == '.':
                        modified_content += modified_sentence
                    else:
                        modified_content += modified_sentence + '.'
                    if i < len(required_words) and len(top_relevant_sentences) > i:
                        modified_content += ' ' + required_words[i] + ' '

            st.write("Suggested Result:")
            st.write(modified_content)


if __name__ == "__main__":
    main()