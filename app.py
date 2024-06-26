from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import firebase_admin
from firebase_admin import credentials, firestore,storage
import requests

from io import BytesIO
import os 
from dotenv import load_dotenv
import datetime 

app = Flask(__name__)
CORS(app)
load_dotenv()
config={
    "apiKey": "AIzaSyBQQ4kVs0hlvK1YUsIT1YLa9r0INhjPWwU",
    "authDomain": "docs-app-47fad.firebaseapp.com",
    "databaseURL": "https://docs-app-47fad.firebaseio.com",
    "projectId": "docs-app-47fad",
    "storageBucket": "docs-app-47fad.appspot.com",
    "messagingSenderId": "869820582771",
    "appId": "1:869820582771:web:0028f17fff858c560cd306",
    "measurementId": "G-CM158SXNJQ"
}
cred = credentials.Certificate('docs-app-47fad-firebase-adminsdk-ker9u-14ecb629ae.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': config["storageBucket"]
})
bucket = storage.bucket()
blobs = bucket.list_blobs()
api_key = os.getenv("GOOGLE_API_KEY")

def list_files_in_blob(blob_name):
    blobs = bucket.list_blobs(prefix=blob_name)
    urls = [blob.generate_signed_url(expiration=datetime.timedelta(hours=1)) for blob in blobs]
    return urls
def get_pdf_text_from_url(pdf_url):
    response = requests.get(pdf_url)
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


@app.route('/get_pdf_urls', methods=['POST'])
def get_pdf_urls():
    data = request.json
    blob_name = data.get("blob_name")
    if not blob_name:
        return jsonify({"error": "Blob name not provided"}), 400
    urls = list_files_in_blob(blob_name)
    print("urlsssss doneeeee")
    raw_text = ""
    for pdf_url in urls:
        if pdf_url:
            pdf_text = get_pdf_text_from_url(pdf_url)
            raw_text += pdf_text
    print("raw text doneeeee")
    text_chunks = get_text_chunks(raw_text)
    print("text_chunks doneeeee")
    get_vector_store(text_chunks)
    print("vector store doneeeee")
    return jsonify({"message": "PDFs processed successfully", "urls": urls})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get("question")
    if user_question:
        response = user_input(user_question)
        return jsonify({"reply": response})
    return jsonify({"error": "Question not provided"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)