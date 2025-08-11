# RAG Chatbot (FAQ TontonUp)

This is a Retrieval-Augmented Generation (RAG) chatbot.  
It uses FAISS for semantic search, SentenceTransformers for embeddings, and Google Gemini for answer generation.

---

## Project Structure
- rag_chatbot.py             # Streamlit app
- CHATBOT_ASSESSMENT.ipynb   # Jupyter Notebook for data processing & index creation
- CHATBOT_ASSESSMENT.HTML    # Jupyter Notebook in HTML 
- faq_data.pkl               # Saved FAQ data
- faq_index.faiss            # FAISS index
- FAQ.docx                   # FAQ document
- requirements.txt
- README.md

---

## Prerequisites

1. **Python 3.10+**  
   Download from [python.org](https://www.python.org/downloads/)

2. **Google Gemini API Key**  
   - Sign up at [Google AI Studio]
   - Copy your API key for later

---

## How to Run

1. Open CHATBOT_ASSESSMENT.ipynb and run all cells to:
-  Parse FAQ.docx
-  Create embeddings
-  Save faq_data.pkl and faq_index.faiss

2. To run the CHATBOT
- Open any terminal (Anaconda Prompt, Command Prompt or PowerShell on Windows, Terminal on Mac/Linux)
- Navigate to the project folder
- Set your API key in terminal
- Run Streamlit: 
     ' streamlit run rag_chatbot.py '

3. Open the local URL shown in terminal (usually http://localhost:8501) in your browser.

4. Testing
- Try asking questions from your FAQ document to verify correct retrieval.
- Enable "Tunjuk panel debug" in the sidebar to see the top retrieved documents and check relevance.

# Notes
- If you change the FAQ document, re-run the notebook to rebuild the FAISS index.
- If no API key is provided, the chatbot will return extractive answers from the FAQ without using Gemini.
- Top-k retrieval value can be adjusted in the sidebar when debug mode is on.



