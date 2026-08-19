# AI-Based Learning Assistant

An AI-powered learning assistant developed as a **Minor Project** to help students interact with and learn from their PDF study materials.

The application allows users to upload PDF documents, processes their content, and provides a conversational interface where users can ask questions related to their uploaded documents.

## 📌 About the Project

The **AI-Based Learning Assistant** is a Python-based application built using Streamlit. It combines PDF text extraction, text chunking, semantic embeddings, vector similarity search, and a Hugging Face language model to provide document-based question answering.

Instead of manually searching through lengthy study materials, users can upload their PDFs and interact with the content through a simple chat interface.

## 🎯 Objectives

* Provide an interactive way to study from PDF documents.
* Reduce the time required to manually search through study material.
* Extract and process text from uploaded PDFs.
* Retrieve relevant information from documents using semantic similarity.
* Provide AI-generated answers through a conversational interface.
* Maintain conversation history during a session.

## ✨ Key Features

* 📄 **Multiple PDF Upload** — Upload multiple PDF documents at once.
* 🔍 **PDF Text Extraction** — Extract readable text from uploaded documents.
* ✂️ **Text Chunking** — Split extracted content into smaller sections for efficient retrieval.
* 🧠 **Semantic Embeddings** — Generate embeddings for document content.
* 📚 **Vector Search** — Retrieve relevant document sections using FAISS.
* 🤖 **AI Question Answering** — Generate responses using a Hugging Face language model.
* 💬 **Conversational Interface** — Ask multiple questions in a chat-based interface.
* 🧾 **Chat History** — Maintain previous questions and answers during the session.
* 🌐 **Streamlit Interface** — Simple web-based interface for interacting with the assistant.

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG)-style workflow:

```text
        PDF Documents
              │
              ▼
       Text Extraction
              │
              ▼
         Text Chunking
              │
              ▼
     Generate Embeddings
              │
              ▼
       FAISS Vector Store
              │
              ▼
        User Question
              │
              ▼
    Relevant Content Retrieval
              │
              ▼
      Hugging Face LLM
              │
              ▼
        Generated Answer
```

### Workflow

1. The user uploads one or more PDF documents.
2. Text is extracted from the uploaded PDFs.
3. The extracted text is divided into smaller chunks.
4. Embeddings are generated for the text chunks.
5. The embeddings are stored in a FAISS vector store.
6. The user asks a question through the chat interface.
7. Relevant document chunks are retrieved using similarity search.
8. The retrieved information is provided as context to the language model.
9. The model generates an answer based on the available context.
10. The conversation is stored in the application's session history.

## 🛠️ Technology Stack

| Technology                | Purpose                               |
| ------------------------- | ------------------------------------- |
| **Python**                | Core programming language             |
| **Streamlit**             | Web application and user interface    |
| **PyPDF2**                | PDF text extraction                   |
| **LangChain**             | LLM and retrieval workflow components |
| **FAISS**                 | Vector similarity search              |
| **Sentence Transformers** | Text embeddings                       |
| **Hugging Face**          | AI language model/API                 |
| **NumPy**                 | Numerical and vector operations       |
| **Requests**              | API communication                     |
| **python-dotenv**         | Environment variable management       |

## 📁 Project Structure

```text
AI-Based-Learning-Assistant/
│
├── app.py
├── htmlTemplates.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

**`app.py`**
Contains the main Streamlit application, PDF processing, text chunking, vector store creation, retrieval logic, conversational functionality, and Hugging Face API integration.

**`htmlTemplates.py`**
Contains the HTML/CSS templates used for displaying user and assistant messages.

**`requirements.txt`**
Contains the Python dependencies required to run the project.

**`.gitignore`**
Prevents environment files, virtual environments, cache files, and other unnecessary files from being committed to the repository.

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Shashank-712/AI-Based-Learning-Assistant.git
```

### 2. Navigate to the Project

```bash
cd AI-Based-Learning-Assistant
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

The application requires a Hugging Face API token.

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

> **Important:** Never upload your `.env` file or API token to GitHub.

The `.env` file is excluded through `.gitignore`.

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 📖 How to Use

1. Start the Streamlit application.
2. Upload one or more PDF study materials from the sidebar.
3. Click **Process** to process the documents.
4. Wait for the documents to be converted into searchable content.
5. Enter a question in the chat interface.
6. The assistant retrieves relevant information from the uploaded documents.
7. The AI generates and displays the answer.
8. Continue asking questions as needed.

## 🎓 Minor Project Context

This project was developed as an academic **Minor Project** with the objective of exploring practical applications of Artificial Intelligence, Natural Language Processing, semantic search, and document-based question answering.

The project demonstrates how different AI and software components can be integrated into a single application to create an interactive learning tool.

## 🔮 Future Scope

Possible future improvements include:

* Support for additional document formats.
* Improved document summarization.
* Automatic generation of study notes.
* Question generation from uploaded material.
* Quiz and assessment generation.
* Improved conversation context management.
* User authentication and personalized study sessions.
* Persistent document and conversation storage.
* Deployment as a publicly accessible web application.
* More advanced AI models and retrieval techniques.

## 👨‍💻 Author

**Shashank Rawat**

Computer Science Engineering Student

GitHub:
https://github.com/Shashank-712

## 📄 License

This project is developed as an academic Minor Project.
