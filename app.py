## KORTEX V1.5 - FINAL MASTER BLUEPRINT ##
import os
import hashlib
import pickle
import io
import uuid
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
import docx
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime, func, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from langchain_core.documents import Document
from sqlalchemy import desc
from serpapi import GoogleSearch

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")

if all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
else:
    DATABASE_URL = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
OLLAMA_BASE_URL = "http://host.docker.internal:11434" if os.environ.get("DOCKER_ENV") == "true" else "http://localhost:11434"
CHAT_MODEL_NAME = "llama3"

# --- DATABASE SETUP ---
engine = create_engine(DATABASE_URL) if DATABASE_URL else None
Base = declarative_base()
Session = sessionmaker(bind=engine)

class DocumentStore(Base):
    __tablename__ = 'document_store'
    doc_hash = Column(String(32), primary_key=True)
    chunks = Column(LargeBinary)
    faiss_index = Column(LargeBinary)

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    doc_hash = Column(String(32), nullable=True)
    chat_history = Column(LargeBinary, default=pickle.dumps([]))
    tutor_curriculum = Column(Text, nullable=True)

def setup_database():
    if not engine: return
    with app.app_context():
        Base.metadata.create_all(engine)

setup_database()

# --- HELPER FUNCTIONS ---
def get_document_text(filepath, file_content):
    text = ""
    file_extension = os.path.splitext(filepath)[1].lower()
    if file_extension == '.pdf':
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if page_text:
                        text += page_text + "\\n"
            if not text.strip():
                pdf_images = fitz.open(stream=file_content, filetype="pdf")
                for page_num in range(len(pdf_images)):
                    page = pdf_images.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img, lang='eng') + "\\n"
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
    elif file_extension == '.docx':
        try:
            doc = docx.Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\\n"
        except Exception as e:
            print(f"Error processing DOCX: {e}")
            return ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_ollama_embeddings():
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=CHAT_MODEL_NAME)

def get_vector_store(text_chunks, embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=CHAT_MODEL_NAME)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

# --- UI TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Kortex</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #111827; }
        main::-webkit-scrollbar { display: none; }
        main { -ms-overflow-style: none; scrollbar-width: none; }
        .mic-button.recording svg { color: #ef4444; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
        .loader-dots div { animation-name: loader-dots; animation-duration: 1.2s; animation-iteration-count: infinite; animation-timing-function: ease-in-out; }
        .loader-dots div:nth-child(2) { animation-delay: 0.15s; } .loader-dots div:nth-child(3) { animation-delay: 0.3s; }
        @keyframes loader-dots { 0% { transform: scale(0); opacity: 0; } 50% { transform: scale(1); opacity: 1; } 100% { transform: scale(0); opacity: 0; } }
        .kortex-gradient { background: -webkit-linear-gradient(45deg, #38bdf8, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        #send-button:disabled { opacity: 0.5; cursor: not-allowed; }
        .message-content { white-space: pre-line; }
    </style>
</head>
<body class="text-gray-200">
    <div class="flex flex-col h-screen w-full">
        <header id="main-header" class="p-4 flex justify-between items-center flex-shrink-0 border-b border-gray-700">
             <h1 class="text-2xl font-bold text-white">Kortex</h1>
             <div id="feature-panel" class="flex items-center gap-2 hidden">
                 <button class="bg-gray-700 hover:bg-gray-600 text-white font-medium py-2 px-4 rounded-lg transition-colors" onclick="startTutorSession()">Teach Me</button>
                 <button id="deep-dive-button" class="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded-lg transition-colors" onclick="startDeepDive()">Deep Dive</button>
             </div>
        </header>
        <main class="flex-1 flex flex-col overflow-y-auto">
            <div id="chat-history" class="flex-1 w-full max-w-3xl mx-auto px-4 md:px-0 py-8">
                 <div id="welcome-message" class="flex flex-col items-center justify-center h-full text-center">
                    <h1 class="text-5xl font-bold kortex-gradient">Kortex</h1>
                    <p class="text-xl mt-2 text-gray-400">Your AI Partner for Deep Research</p>
                </div>
            </div>
        </main>
        <footer id="input-footer" class="p-4 flex-shrink-0">
            <div class="max-w-3xl mx-auto">
                <form id="chat-form" class="bg-gray-800 rounded-xl shadow-lg flex items-center gap-2 p-2">
                    <button type="button" id="upload-button" onclick="document.getElementById('file-upload-input').click()" class="p-2 text-gray-400 hover:text-white transition-colors" title="Analyze Document">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.414a4 4 0 00-5.656-5.656l-6.415 6.415a6 6 0 108.485 8.485L17.5 12"></path></svg>
                    </button>
                    <input type="file" id="file-upload-input" class="hidden" onchange="uploadFile()">
                    <textarea id="user-input" rows="1" class="flex-1 bg-transparent text-gray-200 placeholder-gray-500 focus:outline-none resize-none pr-2" placeholder="Message Kortex..."></textarea>
                    <button type="button" id="mic-button" class="p-2 text-gray-400 hover:text-white transition-colors" title="Use Voice">
                        <svg class="w-6 h-6" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24"><path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"></path></svg>
                    </button>
                    <button type="submit" id="send-button" class="p-2 text-gray-400 hover:text-white transition-colors" title="Send Message">
                        <svg class="w-6 h-6" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path></svg>
                    </button>
                </form>
            </div>
        </footer>
    </div>
    <script>
        let conversationId = null;
        let tutorModeActive = false;
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const chatHistory = document.getElementById('chat-history');
        const mainContent = document.querySelector('main');
        const micButton = document.getElementById('mic-button');
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition;
        let isRecording = false;

        marked.setOptions({ breaks: true });

        function scrollToBottom() { mainContent.scrollTop = mainContent.scrollHeight; }
        function updateSendButtonState() { sendButton.disabled = userInput.value.trim() === ''; }

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.onresult = (event) => {
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) { finalTranscript += event.results[i][0].transcript; }
                userInput.value = finalTranscript;
                adjustTextareaHeight();
                updateSendButtonState();
            };
            recognition.onstart = () => { isRecording = true; micButton.classList.add('recording'); };
            recognition.onend = () => { isRecording = false; micButton.classList.remove('recording'); };
            micButton.addEventListener('click', () => { isRecording ? recognition.stop() : recognition.start(); });
        } else { micButton.style.display = 'none'; }

        async function uploadFile() {
            const fileInput = document.getElementById('file-upload-input');
            if (fileInput.files.length === 0) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            setThinkingState(true, 'Performing deep analysis... this may take several minutes.');
            const response = await fetch('/upload', { 
                method: 'POST', 
                body: formData,
                headers: { 'X-Conversation-ID': conversationId || '' }
            });
            const data = await response.json();
            setThinkingState(false);
            if(response.ok) {
                conversationId = data.conversation_id;
                // Don't clear history if a conversation is already active
                if(document.getElementById('welcome-message')) {
                    chatHistory.innerHTML = '';
                }
                document.getElementById('welcome-message')?.remove();
                document.getElementById('feature-panel').classList.remove('hidden');
                if (chatHistory.innerHTML === '') { // Only add message if chat is new
                   addMessage('Kortex', `**${fileInput.files[0].name}** analyzed and ready. You are now in Research Mode.`);
                }
            } else { addMessage('Kortex', `Analysis failed: ${data.error}`); }
        }

        async function startTutorSession() {
            tutorModeActive = true;
            setThinkingState(true, 'Preparing lesson...');
            const response = await fetch('/tutor', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: conversationId, message: "Let's begin. Please start with the first topic." })
            });
            handleStreamedResponse(response);
        }

        async function startDeepDive() {
            tutorModeActive = false;
            setThinkingState(true, 'Initiating Deep Dive... gathering intelligence.');
            const response = await fetch('/deep_dive', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: conversationId })
            });
            handleStreamedResponse(response);
        }

        userInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if(!sendButton.disabled) document.getElementById('chat-form').requestSubmit();
            }
        });

        document.getElementById('chat-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const message = userInput.value.trim();
            if (!message) return;

            let currentConvId = conversationId;

            if (!currentConvId) {
                try {
                    const response = await fetch('/create_conversation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title: message.substring(0, 50) })
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        addMessage('Kortex', `Error creating conversation: ${data.error}`);
                        return;
                    }
                    currentConvId = data.conversation_id;
                    conversationId = data.conversation_id;
                } catch (error) {
                    addMessage('Kortex', 'Could not start a new conversation. Please check the server.');
                    return;
                }
            }

            userInput.value = '';
            adjustTextareaHeight();
            updateSendButtonState();
            addMessage('You', message);
            setThinkingState(true, 'Kortex is thinking...');
            
            const endpoint = tutorModeActive ? '/tutor' : '/chat';
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: currentConvId, message })
            });
            
            if (!response.ok) {
                 addMessage('Kortex', `I'm sorry, I've encountered a system error. Please try again or start a new session.`);
                 setThinkingState(false);
                 return;
            }
            handleStreamedResponse(response);
        });

        async function handleStreamedResponse(response) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiMessageContainer = addMessage('Kortex', '');
            let contentDiv = aiMessageContainer.querySelector('.message-content');
            let accumulatedText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                accumulatedText += decoder.decode(value, {stream: true});
                contentDiv.innerHTML = marked.parse(accumulatedText);
                scrollToBottom();
            }
            setThinkingState(false);
        }

        function addMessage(sender, text) {
            document.getElementById('welcome-message')?.remove();
            
            const messageWrapper = document.createElement('div');
            messageWrapper.className = `w-full flex py-2 ${sender === 'You' ? 'justify-end' : 'justify-start'}`;
            
            const messageBubble = document.createElement('div');
            // User gets a box, AI does not, for the Gemini look
            messageBubble.className = `max-w-xl ${sender === 'You' ? 'bg-blue-600 text-white p-3 rounded-xl' : ''}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'prose prose-invert max-w-none message-content';
            
            if(sender === 'You') {
                contentDiv.textContent = text;
            } else {
                contentDiv.innerHTML = marked.parse(text);
            }
            
            messageBubble.appendChild(contentDiv);
            messageWrapper.appendChild(messageBubble);
            chatHistory.appendChild(messageWrapper);
            scrollToBottom();
            return messageWrapper;
        }

        function setThinkingState(isThinking, text = '') {
             let thinkingDiv = document.getElementById('thinking-indicator');
            if (isThinking) {
                if (!thinkingDiv) {
                    thinkingDiv = addMessage('Kortex', '');
                    thinkingDiv.id = 'thinking-indicator';
                    let thinkingText = text ? `<span class="ml-2 text-gray-400">${text}</span>` : '';
                    thinkingDiv.querySelector('.message-content').innerHTML = `<div class="flex items-center"><div class="loader-dots flex items-center"><div class="w-2 h-2 bg-gray-400 rounded-full"></div><div class="w-2 h-2 bg-gray-400 rounded-full"></div><div class="w-2 h-2 bg-gray-400 rounded-full"></div></div>${thinkingText}</div>`;
                }
            } else { if (thinkingDiv) thinkingDiv.remove(); }
             scrollToBottom();
        }
        
        function adjustTextareaHeight() {
             userInput.style.height = 'auto';
             userInput.style.height = (userInput.scrollHeight) + 'px';
        }
        userInput.addEventListener('input', () => {
            adjustTextareaHeight();
            updateSendButtonState();
        });

        updateSendButtonState();
    </script>
</body>
</html>
"""

# --- API ENDPOINTS ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/create_conversation', methods=['POST'])
def create_conversation():
    data = request.json
    title = data.get('title', 'New Chat')
    session = Session()
    try:
        new_conv = Conversation(title=title)
        session.add(new_conv)
        session.commit()
        return jsonify({"conversation_id": str(new_conv.id)})
    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@app.route('/upload', methods=['POST'])
def upload():
    session = Session()
    try:
        file = request.files.get('file')
        if not file or not file.filename: return jsonify({"error": "No file part"}), 400
        
        filename = secure_filename(file.filename)
        file_content = file.read()
        doc_hash = hashlib.md5(file_content).hexdigest()
        
        doc = session.query(DocumentStore).filter_by(doc_hash=doc_hash).first()
        if not doc:
            raw_text = get_document_text(filename, file_content)
            
            if not raw_text or len(raw_text.strip()) < 50:
                return jsonify({"error": "This document contains no readable text or is too short to analyze."}), 400

            text_chunks = get_text_chunks(raw_text)
            embeddings = get_ollama_embeddings()
            vectorstore = get_vector_store(text_chunks, embeddings)
            new_doc = DocumentStore(doc_hash=doc_hash, chunks=pickle.dumps(text_chunks), faiss_index=pickle.dumps(vectorstore.serialize_to_bytes()))
            session.add(new_doc)
        
        conv_id_header = request.headers.get('X-Conversation-ID')
        conv = None
        if conv_id_header:
            conv = session.query(Conversation).filter_by(id=conv_id_header).first()
        
        if conv:
            conv.doc_hash = doc_hash
            conv.title = filename
        else:
            conv = Conversation(title=filename, doc_hash=doc_hash)
            session.add(conv)

        session.commit()
        return jsonify({"message": "File processed", "conversation_id": str(conv.id)})
    except Exception as e:
        session.rollback()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        session.close()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session = Session()
    try:
        conversation_id, message = data.get('conversation_id'), data.get('message')
        if not all([conversation_id, message]): return jsonify({"error": "Missing conversation_id or message"}), 400
        
        conv = session.query(Conversation).filter_by(id=conversation_id).first()
        if not conv: return jsonify({"error": "Conversation not found"}), 404

        chat_history_list = pickle.loads(conv.chat_history)
        
        if not conv.doc_hash:
             llm = Ollama(base_url=OLLAMA_BASE_URL, model=CHAT_MODEL_NAME)
             def generate_general():
                 response = llm.invoke(message)
                 yield response
                 chat_history_list.append(('langchain', message, response))
                 conv_to_update = session.merge(conv)
                 conv_to_update.chat_history = pickle.dumps(chat_history_list)
                 session.commit()
             return Response(generate_general(), mimetype='text/plain')

        doc_store = session.query(DocumentStore).filter_by(doc_hash=conv.doc_hash).first()
        if not doc_store: return jsonify({"error": "Document data not found"}), 404

        embeddings = get_ollama_embeddings()
        vectorstore = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pickle.loads(doc_store.faiss_index), allow_dangerous_deserialization=True)
        chain = get_conversation_chain(vectorstore)
        langchain_history_format = [(item[1], item[2]) for item in chat_history_list if len(item) == 3 and item[0] == 'langchain']

        def generate_doc_chat():
            response = chain({"question": message, "chat_history": langchain_history_format})
            answer = response.get('answer', 'Sorry, I could not find an answer.')
            yield answer
            chat_history_list.append(('langchain', message, answer))
            conv_to_update = session.merge(conv)
            conv_to_update.chat_history = pickle.dumps(chat_history_list)
            session.commit()

        return Response(generate_doc_chat(), mimetype='text/plain')
    except Exception as e:
        session.rollback()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        session.close()

@app.route('/tutor', methods=['POST'])
def tutor():
    data = request.json
    session = Session()
    try:
        conversation_id, message = data.get('conversation_id'), data.get('message')
        if not all([conversation_id, message]): return jsonify({"error": "Missing conversation_id or message"}), 400
        
        conv = session.query(Conversation).filter_by(id=conversation_id).first()
        if not conv or not conv.doc_hash: return jsonify({"error": "Conversation or document not found"}), 404
        
        doc_store = session.query(DocumentStore).filter_by(doc_hash=conv.doc_hash).first()
        if not doc_store: return jsonify({"error": "Document content not found"}), 404
             
        loaded_chunks = pickle.loads(doc_store.chunks)
        document_context = "\\n".join([chunk.page_content for chunk in loaded_chunks]) if loaded_chunks and isinstance(loaded_chunks[0], Document) else "\\n".join(loaded_chunks)

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        chat_history_list = pickle.loads(conv.chat_history)
        gemini_history = [item for item in chat_history_list if item[0] == 'gemini']

        if not conv.tutor_curriculum:
            curriculum_prompt = f"Analyze the document and create a Socratic curriculum scaled to its length. Respond ONLY with the numbered list.\\n\\nDOCUMENT:\\n{document_context[:10000]}"
            curriculum_response = model.generate_content(curriculum_prompt)
            conv.tutor_curriculum = curriculum_response.text
            session.commit()
            conv = session.query(Conversation).filter_by(id=conversation_id).first()

        def generate_tutor_response():
            try:
                fresh_conv = session.query(Conversation).filter_by(id=conversation_id).first()
                is_first_tutor_message = not any(item[0] == 'gemini' for item in chat_history_list)
                
                prompt_parts = [
                    "## 1. CORE IDENTITY ##",
                    "You are Kortex, an elite AI mentor. Your personality is a blend of a brilliant, charismatic professor and a patient, insightful partner. You are here to help the user think deeper and learn faster.",
                    "",
                    "## 2. CORE MISSION & LOGIC ##",
                    "Your ultimate goal is to create a natural and intelligent conversation that guides the user through the lesson plan. To do this, you MUST follow this logic:",
                    "",
                    f"IF the user's message is the VERY FIRST ONE in the tutoring session (is_first_tutor_message = {is_first_tutor_message}), your primary task is to:",
                    "1. Introduce yourself and the purpose of the lesson.",
                    "2. Present the complete lesson plan you have created.",
                    "3. Ask an engaging, open-ended Socratic question about the very first topic in the plan.",
                    "",
                    "OTHERWISE (for all subsequent messages), your task is to be a dynamic partner:",
                    "1. Analyze the user's latest message. Is it a response to your question or a new command?",
                    "2. If it's a response, continue the Socratic dialogue.",
                    "3. If it's a command (e.g., \"make a quiz\"), be a proactive assistant. If the command is ambiguous, ask clarifying questions. **When the user provides specific parameters (like '15 questions'), you MUST adhere to them precisely.** After gathering sufficient information, execute the command. It is better to provide a good result now than to ask endless questions.",
                    "## 3. CONTEXT FOR YOUR MISSION ##",   
                    f"- **is_first_tutor_message:** {is_first_tutor_message}",
                    f"- **Lesson Plan Roadmap:** {fresh_conv.tutor_curriculum}",
                    f"- **Conversation History:** {gemini_history}",
                    f"- **Document Snippet for Context:** {document_context[:4000]}",
                    "",
                    "## 4. CRITICAL RULES (NON-NEGOTIABLE) ##",
                    "- **NO ROBOTIC INTROS:** NEVER start with \"Excellent!\", \"Great!\", \"Okay, so...\". Find a natural, conversational opening.",
                    "- **PERFECT MCQ FORMATTING TEMPLATE:** When creating a quiz, you MUST format it EXACTLY like this, replacing the bracketed content. Do not add any text before or after this structure:",
                    "  1. [Question 1 Text]",
                    "     A. [Option A]",
                    "     B. [Option B]",
                    "     C. [Option C]",
                    "     D. [Option D]",
                    "",
                    "  2. [Question 2 Text]",
                    "     A. [Option A]",
                    "     B. [Option B]",
                    "     C. [Option C]",
                    "     D. [Option D]",
                    "", 
                    
                    
                    "## 5. THE USER'S LATEST MESSAGE ##",
                    f'"{message}"',
                    "",
                    "## 6. YOUR RESPONSE: ##"
                ]
                master_prompt = "\\n".join(prompt_parts)
                
                response_stream = model.generate_content(master_prompt, stream=True)
                full_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                        full_response += chunk.text
                
                chat_history_list.append(('gemini', 'user', message))
                chat_history_list.append(('gemini', 'model', full_response))

                conv_to_update = session.merge(fresh_conv)
                conv_to_update.chat_history = pickle.dumps(chat_history_list)
                session.commit()

            except Exception as e:
                yield f"I'm sorry, a critical error occurred. Error: {str(e)}"
        
        return Response(generate_tutor_response(), mimetype='text/plain')
    except Exception as e:
        session.rollback()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        session.close()

@app.route('/deep_dive', methods=['POST'])
def deep_dive():
    if not SERPER_API_KEY: return jsonify({"error": "Serper API key not configured"}), 500
    data = request.json
    session = Session()
    try:
        conversation_id = data.get('conversation_id')
        conv = session.query(Conversation).filter_by(id=conversation_id).first()
        if not conv or not conv.doc_hash: return jsonify({"error": "A document must be associated for a Deep Dive."}), 400

        doc_store = session.query(DocumentStore).filter_by(doc_hash=conv.doc_hash).first()
        if not doc_store: return jsonify({"error": "Document content not found."}), 404

        document_context = "\\n".join([chunk.page_content for chunk in pickle.loads(doc_store.chunks)])
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        chat_history_list = pickle.loads(conv.chat_history)

        def generate_deep_dive_response():
            try:
                # Step 1: Generate Queries
                query_gen_prompt = f"Analyze document. Generate JSON array of 3 expert Google queries for risks & trends. ONLY JSON array.\\n\\nDOC:{document_context[:4000]}"
                query_response = model.generate_content(query_gen_prompt)
                
                response_text = query_response.text
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if not json_match: raise ValueError("AI failed to generate search queries.")
                json_str = json_match.group(0)
                queries = json.loads(json_str)

                # Step 2: Search Web
                web_context = ""
                yield "Gathering live intelligence from the web...<br>"
                for query in queries:
                    yield f"Searching for: '{query}'...<br>"
                    search = GoogleSearch({ "q": query, "api_key": SERPER_API_KEY })
                    results = search.get_dict()
                    organic_results = results.get("organic_results", [])
                    for result in organic_results[:3]:
                        if "snippet" in result: web_context += result["snippet"] + "<br>"  
                
                # Step 3: Synthesize
                yield "<br>Synthesizing intelligence brief...<br><br>"  
                synthesis_prompt = f"You are a world-class analyst. Synthesize original doc with live web data. Create a concise, actionable brief. Use Markdown.\\n\\nORIGINAL DOC:{document_context[:4000]}\\n\\nLIVE WEB DATA:{web_context}\\n\\nBRIEF:"
                synthesis_stream = model.generate_content(synthesis_prompt, stream=True)
                
                full_response = ""
                for chunk in synthesis_stream: 
                    if chunk.text:
                        yield chunk.text
                        full_response += chunk.text
                
                chat_history_list.append(('gemini', 'user', "Deep Dive Analysis"))
                chat_history_list.append(('gemini', 'model', full_response))
                
                conv_to_update = session.merge(conv)
                conv_to_update.chat_history = pickle.dumps(chat_history_list)
                session.commit()

            except Exception as e:
                yield f"A critical error occurred during Deep Dive. Error: {str(e)}"

        return Response(generate_deep_dive_response(), mimetype='text/plain')
    
    except Exception as e:
        session.rollback()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        session.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 