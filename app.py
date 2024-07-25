from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
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
from dotenv import load_dotenv,dotenv_values
import datetime 
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage,SystemMessage
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": ""}})
apiKey= os.getenv("APIKEY")
authDomain= os.getenv("AUTHDOMAIN")
projectId= os.getenv("PROJECTID")
storageBucket= os.getenv("STORAGEBUCKET")
messagingSenderId= os.getenv("MESSAGINGSENDERID")
appId= os.getenv("APPID")
measurementId=os.getenv("MEASUREMENTID")
config={
    apiKey: apiKey,
    authDomain: authDomain,
    projectId: projectId,
    storageBucket: storageBucket,
    messagingSenderId: messagingSenderId,
    appId: appId,
    measurementId:measurementId
}
cred = credentials.Certificate('docs-app2.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': config[storageBucket]
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
@cross_origin()
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
@cross_origin()
def ask_question():
    data = request.json
    user_question = data.get("question")
    if user_question:
        response = user_input(user_question)
        return jsonify({"reply": response})
    return jsonify({"error": "Question not provided"}), 400



def classify_text_with_gemini(text_chunks):
    full_text = " ".join(text_chunks)
    my_prompt=PromptTemplate.from_template("""*Attention! Forget everything you learned before. Treat this as a new document.* \n You are an expert document classifier and you have to classify the given text below into \n Education, \n Finance, \n Healthcare, \n Legal, \n Technology, \n Nationalism, \n Media, \n Entertainment.\n\n
    It is necessary for you to classify only into one of the above category.Give the reply only in one word 
    Document Text:\n
    {full_text}
    """)
    my_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)
    chain=LLMChain(
        llm=my_llm,
        prompt=my_prompt,
        verbose=True
    )
    response=chain.invoke(input=full_text)
    sector=response["text"]
    print("Model response received:", response["text"])
    return sector


@app.route('/categorize', methods=['POST'])
@cross_origin()
def categorize():
    data = request.json
    email = data.get("blob_name")
    filename=data.get("fileName")
    if not email:
        return jsonify({"error": "Blob name not provided"}), 400
    urls = list_files_in_blob(email)
    print("urlsssss doneeeee")
    filtered_url = [url for url in urls if filename in url]
    print(filtered_url)
    raw_text = ""
    for pdf_url in filtered_url:
        if pdf_url:
            pdf_text = get_pdf_text_from_url(pdf_url)
            raw_text += pdf_text
    print("raw text doneeeee")
    text_chunks = get_text_chunks(raw_text)
    print("text_chunks doneeeee")
    sector = classify_text_with_gemini(text_chunks)
    print("After Gemini call")
    source_path = f'{email}/{filename}'
    destination_path = f'{email}/{sector}/{filename}'
    blob = bucket.blob(source_path)
    new_blob = bucket.copy_blob(blob, bucket, destination_path)
    print(f'File copied to {destination_path}')
    blob.delete()
    print(f'Original file {source_path} deleted')
    return sector

@app.route('/check', methods=['POST'])
@cross_origin()
def check():
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)