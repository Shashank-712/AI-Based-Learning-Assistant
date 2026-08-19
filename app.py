import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template
import os
import traceback

# --- Text splitter compatibility ---
try:
    from langchain.text_splitter import CharacterTextSplitter
    print("Using: langchain.text_splitter")
except Exception:
    try:
        from langchain.text_splitters import CharacterTextSplitter
        print("Using: langchain.text_splitters")
    except Exception:
        from langchain_text_splitters import CharacterTextSplitter
        print("Using: langchain_text_splitters")

# --- Vectorstore / FAISS compatibility & local wrapper ---
try:
    from langchain.vectorstores import FAISS as LC_FAISS
    print("Using: langchain.vectorstores.FAISS")
    FAISS = LC_FAISS
except Exception:
    try:
        from langchain.vectorstores.faiss import FAISS as LC_FAISS2
        print("Using: langchain.vectorstores.faiss.FAISS")
        FAISS = LC_FAISS2
    except Exception:
        import faiss
        import numpy as np
        import os
        import pickle
        print("Using: local SimpleFAISS wrapper (langchain FAISS missing)")

        class SimpleFAISS:
            def __init__(self, index: faiss.Index, texts: list[str], embedding, dimension: int):
                self.index = index
                self.texts = list(texts)
                self.embedding = embedding
                self.dimension = dimension

            @classmethod
            def from_texts(cls, texts: list[str], embedding):
                vectors = embedding.embed_documents(texts)
                if len(vectors) == 0:
                    raise ValueError("No vectors returned from embedding.embed_documents()")
                arr = np.array(vectors, dtype="float32")
                dim = arr.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(arr)
                return cls(index=index, texts=texts, embedding=embedding, dimension=dim)

            def similarity_search(self, query: str, k: int = 4):
                qvec = np.array([self.embedding.embed_query(query)], dtype="float32")
                if qvec.shape[1] != self.dimension:
                    raise ValueError(
                        f"Query embedding dimension {qvec.shape[1]} does not match index dimension {self.dimension}"
                    )
                distances, indices = self.index.search(qvec, k)
                results = []
                for idx in indices[0]:
                    if idx < 0 or idx >= len(self.texts):
                        continue
                    results.append(self.texts[int(idx)])
                return results
            
            def as_retriever(self, **kwargs):
                """Return a simple retriever interface for compatibility with ConversationalRetrievalChain"""
                class SimpleRetriever:
                    def __init__(self, vectorstore):
                        self.vectorstore = vectorstore
                    
                    def get_relevant_documents(self, query: str):
                        return self.vectorstore.similarity_search(query, k=4)
                
                return SimpleRetriever(self)

            def save_local(self, folder_path: str):
                os.makedirs(folder_path, exist_ok=True)
                faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
                with open(os.path.join(folder_path, "texts.pkl"), "wb") as f:
                    pickle.dump(self.texts, f)
                with open(os.path.join(folder_path, "meta.pkl"), "wb") as f:
                    pickle.dump({"dimension": self.dimension}, f)

            @classmethod
            def load_local(cls, folder_path: str, embedding):
                index_path = os.path.join(folder_path, "index.faiss")
                texts_path = os.path.join(folder_path, "texts.pkl")
                meta_path = os.path.join(folder_path, "meta.pkl")
                if not os.path.exists(index_path) or not os.path.exists(texts_path):
                    raise FileNotFoundError("Saved index or texts not found in folder.")
                index = faiss.read_index(index_path)
                with open(texts_path, "rb") as f:
                    texts = pickle.load(f)
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                dimension = meta.get("dimension")
                return cls(index=index, texts=texts, embedding=embedding, dimension=dimension)

        FAISS = SimpleFAISS

# --- Embeddings compatibility (HuggingFace / sentence-transformers fallback) ---
try:
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    print("Using langchain.embeddings.HuggingFaceInstructEmbeddings")
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
        print("Using langchain.embeddings.HuggingFaceEmbeddings as fallback")

        class HuggingFaceInstructEmbeddings:
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", **kwargs):
                self._inner = _HuggingFaceEmbeddings(model_name=model_name, **kwargs)

            def embed_documents(self, texts):
                return self._inner.embed_documents(texts)

            def embed_query(self, text):
                return self._inner.embed_query(text)

    except Exception:
        try:
            from sentence_transformers import SentenceTransformer
            print("Using sentence-transformers fallback as HuggingFaceInstructEmbeddings")

            class HuggingFaceInstructEmbeddings:
                def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", **kwargs):
                    self.model_name = model_name
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, texts):
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                    return [e.tolist() for e in embeddings]

                def embed_query(self, text):
                    emb = self.model.encode(text, convert_to_numpy=True)
                    return emb.tolist()

        except Exception as e:
            raise ImportError(
                "No HuggingFace embeddings available. Install langchain or sentence-transformers. Original error: " + str(e)
            )

# --- ConversationBufferMemory compatibility ---
try:
    from langchain.memory import ConversationBufferMemory
    print("Using: langchain.memory.ConversationBufferMemory")
except Exception:
    try:
        from langchain_community.memory import ConversationBufferMemory
        print("Using: langchain_community.memory.ConversationBufferMemory")
    except Exception:
        print("Using fallback ConversationBufferMemory (simple implementation)")

        class ConversationBufferMemory:
            """
            Minimal replacement for langchain's ConversationBufferMemory.
            """

            def __init__(self, memory_key: str = "chat_history", return_messages: bool = False):
                self.memory_key = memory_key
                self.return_messages = return_messages
                self.chat_memory = []

                try:
                    import streamlit as _st
                    if self.memory_key not in _st.session_state:
                        _st.session_state[self.memory_key] = []
                    self.chat_memory = list(_st.session_state[self.memory_key])
                    self._st = _st
                except Exception:
                    self._st = None

            def _sync_to_session(self):
                if self._st is not None:
                    self._st.session_state[self.memory_key] = list(self.chat_memory)

            def add_user_message(self, text: str):
                self.chat_memory.append({"role": "user", "text": text})
                self._sync_to_session()

            def add_ai_message(self, text: str):
                self.chat_memory.append({"role": "ai", "text": text})
                self._sync_to_session()

            def load_memory_variables(self, inputs: dict):
                if self.return_messages:
                    return {self.memory_key: list(self.chat_memory)}
                joined = ""
                for msg in self.chat_memory:
                    prefix = "User: " if msg["role"] == "user" else "AI: "
                    joined += prefix + msg["text"] + "\n"
                return {self.memory_key: joined}

            def save_context(self, inputs: dict, outputs: dict):
                user_text = None
                for k in ("input", "question", "query", "text", "prompt"):
                    if k in inputs and inputs[k]:
                        user_text = inputs[k]
                        break
                
                ai_text = None
                for k in ("output", "answer", "result", "response"):
                    if k in outputs and outputs[k]:
                        ai_text = outputs[k]
                        break

                if user_text:
                    self.add_user_message(str(user_text))
                if ai_text:
                    text = ai_text if isinstance(ai_text, str) else str(ai_text)
                    self.add_ai_message(text)

            def clear(self):
                self.chat_memory = []
                self._sync_to_session()

# --- ConversationalRetrievalChain compatibility ---
try:
    from langchain.chains import ConversationalRetrievalChain
    print("Using: langchain.chains.ConversationalRetrievalChain")
except Exception:
    try:
        from langchain_community.chains import ConversationalRetrievalChain
        print("Using: langchain_community.chains.ConversationalRetrievalChain")
    except Exception:
        print("Using fallback ConversationalRetrievalChain (simple implementation)")
        
        class ConversationalRetrievalChain:
            """
            Minimal replacement for langchain's ConversationalRetrievalChain.
            """
            def __init__(self, llm, retriever, memory):
                self.llm = llm
                self.retriever = retriever
                self.memory = memory
            
            @classmethod
            def from_llm(cls, llm, retriever, memory):
                return cls(llm=llm, retriever=retriever, memory=memory)
            
            def __call__(self, inputs: dict):
                question = inputs.get("question", "")
                
                # Get relevant documents
                try:
                    docs = self.retriever.get_relevant_documents(question)
                    context = "\n\n".join(docs[:4])  # Use top 4 docs
                except Exception as e:
                    print(f"Error retrieving documents: {e}")
                    traceback.print_exc()
                    context = ""
                
                # Get chat history
                memory_vars = self.memory.load_memory_variables({})
                chat_history = memory_vars.get(self.memory.memory_key, "")
                
                # Build prompt
                if isinstance(chat_history, list):
                    history_text = ""
                    for msg in chat_history:
                        if isinstance(msg, dict):
                            role = msg.get("role", "unknown")
                            text = msg.get("text", "")
                            history_text += f"{role}: {text}\n"
                        else:
                            history_text += str(msg) + "\n"
                    chat_history = history_text
                
                prompt = f"""Based on the following context, answer the question.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
                
                # Get answer from LLM
                try:
                    # Try multiple methods to get LLM response
                    if hasattr(self.llm, 'invoke'):
                        result = self.llm.invoke(prompt)
                        # Handle different return types
                        if isinstance(result, str):
                            answer = result
                        elif hasattr(result, 'content'):
                            answer = result.content
                        elif hasattr(result, 'text'):
                            answer = result.text
                        else:
                            answer = str(result)
                    elif hasattr(self.llm, 'predict'):
                        answer = self.llm.predict(prompt)
                    elif hasattr(self.llm, '__call__'):
                        result = self.llm.__call__(prompt)
                        answer = result if isinstance(result, str) else str(result)
                    else:
                        # Last resort - try generate
                        result = self.llm.generate([prompt])
                        answer = str(result)
                except Exception as e:
                    print(f"LLM Error: {e}")
                    print(f"Error type: {type(e).__name__}")
                    traceback.print_exc()
                    # Fallback: return the context from documents
                    if context:
                        answer = f"Here's what I found in your documents:\n\n{context[:500]}..."
                    else:
                        answer = "I couldn't generate an answer. Please try rephrasing your question."
                
                # Save to memory
                self.memory.save_context(
                    inputs={"question": question},
                    outputs={"answer": answer}
                )
                
                # Return in expected format
                return {
                    "question": question,
                    "answer": answer,
                    "chat_history": self.memory.chat_memory
                }


# ------------------ App functions ------------------

def get_pdf_text(pdf_docs):
    """Read uploaded Streamlit file objects (list) and return concatenated text."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {getattr(pdf, 'name', '<uploaded>')}: {e}")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not text_chunks:
        return None
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create conversation chain with improved error handling"""
    
    # Check for API token
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_token:
        print("⚠️ No HUGGINGFACEHUB_API_TOKEN found in environment")
        print("⚠️ Add it to .env file or use basic retrieval mode")
    
    llm = None
    
    # Try HuggingFace with pure REST API (bypass all provider issues)
    if api_token:
        try:
            import requests
            print("Trying HuggingFace REST API...")
            
            # Create a wrapper that uses requests directly with OpenAI-compatible format
            class RestAPIHuggingFaceLLM:
                def __init__(self, model_id, token):
                    self.model_id = model_id
                    # Use OpenAI-compatible endpoint
                    self.api_url = "https://router.huggingface.co/v1/chat/completions"
                    self.headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    }
                    print(f"Initialized RestAPIHuggingFaceLLM with {model_id}")
                    print(f"Using API endpoint: {self.api_url}")
                
                def invoke(self, prompt):
                    try:
                        # Use OpenAI-compatible chat format
                        payload = {
                            "model": self.model_id,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": 200,
                            "temperature": 0.7
                        }
                        response = requests.post(
                            self.api_url, 
                            headers=self.headers, 
                            json=payload,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            # Handle OpenAI-style response
                            if 'choices' in result and len(result['choices']) > 0:
                                return result['choices'][0]['message']['content']
                            return str(result)
                        else:
                            error_msg = f"API Error {response.status_code}: {response.text}"
                            print(error_msg)
                            raise Exception(error_msg)
                            
                    except requests.exceptions.Timeout:
                        return "Request timed out. The model might be loading. Please try again."
                    except Exception as e:
                        print(f"RestAPIHuggingFaceLLM error: {e}")
                        raise e
                
                def predict(self, prompt):
                    return self.invoke(prompt)
                
                def __call__(self, prompt):
                    return self.invoke(prompt)
            
            # Try a compatible model - use a model available through Inference Providers
            # Popular options: meta-llama/Llama-3.3-70B-Instruct, mistralai/Mixtral-8x7B-Instruct-v0.1
            llm = RestAPIHuggingFaceLLM("meta-llama/Llama-3.3-70B-Instruct", api_token)
            test_response = llm.invoke("What is 2+2?")
            print(f"✅ REST API test successful! Response: {test_response[:100]}")
            print("✅ Using HuggingFace REST API (bypassing all library issues)")
            
        except Exception as e:
            print(f"❌ REST API failed: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            traceback.print_exc()
            llm = None
    
    # Fallback to BasicRetrievalLLM if HuggingFace failed or no token
    if llm is None:
        print("🔄 Using BasicRetrievalLLM (retrieves context from documents)")
        
        if not api_token:
            st.warning("⚠️ No HuggingFace API token found. Using basic retrieval mode. Add HUGGINGFACEHUB_API_TOKEN to .env for AI-powered answers.")
        else:
            st.info("ℹ️ HuggingFace API unavailable. Using basic retrieval mode (returns relevant document excerpts).")
        
        class BasicRetrievalLLM:
            """Basic LLM that returns relevant context from documents"""
            
            def invoke(self, prompt):
                """Extract context and return it"""
                try:
                    if "Context:" in prompt and "Question:" in prompt:
                        # Extract context
                        context_start = prompt.find("Context:") + len("Context:")
                        context_end = prompt.find("Chat History:")
                        if context_end == -1:
                            context_end = prompt.find("Question:")
                        
                        context = prompt[context_start:context_end].strip()
                        
                        # Extract question
                        question_start = prompt.find("Question:") + len("Question:")
                        question_end = prompt.find("Answer:")
                        if question_end == -1:
                            question_end = len(prompt)
                        question = prompt[question_start:question_end].strip()
                        
                        if context:
                            # Return formatted answer with context
                            return f"Based on your documents:\n\n{context[:1000]}\n\n(Note: Using basic retrieval mode. Add HuggingFace API token for better answers)"
                        else:
                            return f"I couldn't find relevant information about '{question}' in your documents."
                    
                    return "Please upload documents and ask a question."
                    
                except Exception as e:
                    print(f"BasicRetrievalLLM error: {e}")
                    return "Error processing your question. Please try again."
            
            def predict(self, prompt):
                return self.invoke(prompt)
            
            def __call__(self, prompt):
                return self.invoke(prompt)
        
        llm = BasicRetrievalLLM()
    else:
        st.success("✅ Using HuggingFace AI model for intelligent answers!")

    # Create memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain


def handle_user_input(user_question):
    """Handle user question - unified chat that uses PDFs when available"""
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    try:
        import requests
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if not api_token:
            st.warning("⚠️ Please add your HuggingFace API token to use chat!")
            return
        
        with st.spinner("Thinking..."):
            # Check if we have a vectorstore (PDFs uploaded)
            has_docs = 'vectorstore' in st.session_state and st.session_state.vectorstore is not None
            
            context = ""
            if has_docs:
                # Try to get relevant context from PDFs
                try:
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.get_relevant_documents(user_question)
                    if docs:
                        context = "\n\n".join(docs[:3])  # Top 3 relevant chunks
                except Exception as e:
                    print(f"Error retrieving docs: {e}")
            
            # Build messages for API
            messages = []
            
            # Add chat history (last 6 messages = 3 exchanges)
            for msg in st.session_state.chat_history[-6:]:
                messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["text"]
                })
            
            # Build the current question with context if available
            if context:
                current_message = f"""Based on the following document context, answer the question. If the answer is not in the context, say "I couldn't find that in your documents, but here's what I know:" and then provide a general answer.

Document Context:
{context}

Question: {user_question}"""
            else:
                current_message = user_question
            
            messages.append({"role": "user", "content": current_message})
            
            # Make API call
            api_url = "https://router.huggingface.co/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                
                # Save to history (save original question, not the one with context)
                st.session_state.chat_history.append({"role": "user", "text": user_question})
                st.session_state.chat_history.append({"role": "assistant", "text": answer})
            else:
                st.error(f"❌ API Error: {response.status_code}")
                
    except Exception as e:
        st.error(f"❌ Error: {e}")
        print(f"Chat error: {e}")
        traceback.print_exc()


def main():
    load_dotenv()
    st.set_page_config(page_title="Learning Assistant", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Learning Assistant :books:")
    
    # Show PDF status
    if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
        st.success("📄 PDFs loaded - I'll use them to answer your questions!")
    
    # Use a form to auto-clear input after submission
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask me anything:", key="user_input")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_question:
        handle_user_input(user_question)
    
    # Display chat history in reverse order (newest first)
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            msg = st.session_state.chat_history[i]
            if msg["role"] == "user":
                st.write(user_template.replace("{{MSG}}", msg["text"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg["text"]), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", 
            accept_multiple_files=True, 
            type=["pdf"]
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
                return
                
            with st.spinner("Processing..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text.strip():
                    st.error("No text could be extracted from the PDFs")
                    return
                
                # Create text chunks
                text_chunks = get_text_chunks(raw_text)
                st.success(f"✅ Created {len(text_chunks)} chunks")

                # Create vectorstore
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    st.error("No text chunks to index. Upload PDFs and try again.")
                    return
                    
                st.session_state["vectorstore"] = vectorstore
                st.success("✅ PDFs processed! Continue chatting - I'll use your documents now!")

                # Remove the old conversation chain approach
                st.session_state.conversation = None


if __name__ == '__main__':
    main()