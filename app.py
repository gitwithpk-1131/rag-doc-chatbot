from ast import Store
import os, tempfile
import streamlit as st
from dotenv import load_dotenv
from rag.loader import load_and_split
from rag.vectorstore import build_vectorstore
from rag.chain import answer

st.set_page_config(page_title="RAG Document Q&A Chatbot", layout="wide")
load_dotenv()

GITHUB_URL = "https://github.com/your-username/rag-doc-chatbot"

def get_setting(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

os.environ["GOOGLE_API_KEY"] = get_setting("GEMINI_API_KEY")
for name in ("LLM_MODEL", "EMBEDDING_MODEL"):
    value = get_setting(name)
    if value:
        os.environ[name] = value

if not os.environ["GOOGLE_API_KEY"]:
    st.error("GEMINI_API_KEY not found. Add it to your .env file.")
    st.stop()

# ---------- Header: title on the left, technology buttons on the right ----------
TECH = {
    "RAG":      "RAG (Retrieval-Augmented Generation)" 
                "An LLM only knows what it was trained on, so it can't answer questions about your own PDFs." 
                "RAG fixes this by looking up the relevant parts of your documents first and handing them to the model along with the question." 
                "The model then answers from that text instead of from memory, which cuts down on made-up answers.",
    "LangChain": "LangChain is a toolkit that gives you ready-made pieces for each step: document loaders (PDF, Word, text), text splitters," 
                 "embedding wrappers, vector store wrappers, and chat model wrappers. Its real value is that these pieces are swappable." 
                 "If you later replace ChromaDB with FAISS or Gemini with another model, most of your code stays the same.",
    "Streamlit": "Streamlit turns a Python script into a web app with no HTML or JavaScript." 
                 "For this project you need only a few things: a file uploader, a chat input, chat message bubbles, and a sidebar for settings.",
    "Gemini API": "Google's AI. It turns text into numbers (embeddings) for "
                  "searching, and it writes the final answer from the passages found.",
    "ChromaDB": "Chroma is an open-source vector database that runs inside your Python program with no separate server." 
                "It stores each chunk's text, its embedding, and metadata like filename and page number, and it can search by similarity."
}

head_left, head_right = st.columns([2, 3])
with head_left:
    st.title("RAG Document Q&A Chatbot")
with head_right:
    st.write("")
    tech_cols = st.columns(len(TECH))
    for col, (label, text) in zip(tech_cols, TECH.items()):
        with col:
            with st.popover(label, use_container_width=True):
                st.write(text)

st.divider()

# ---------- Middle: upload box on the left, info on the right ----------
left, right = st.columns(2)

with left:
    st.subheader("Upload your documents")
    files = st.file_uploader("Choose one or more PDFs", type="pdf",
                             accept_multiple_files=True)
    if files and st.button("Process documents"):
        chunks = []
        with st.spinner("Reading and indexing your documents..."):
            for up in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(up.read())
                file_chunks = load_and_split(f.name)
                for c in file_chunks:
                    c.metadata["source_name"] = up.name
                chunks.extend(file_chunks)
            st.session_state.vs = build_vectorstore(chunks)
            st.session_state.messages = []
        st.success(f"Indexed {len(files)} file(s). Ask your first question below.")

with right:
    with st.expander("How to use it", expanded=True):
        st.markdown(
            "1. Upload one or more PDF files\n"
            "2. Click **Process documents**\n"
            "3. Ask questions in the chat box below\n"
            "4. Open **Sources** under an answer to see where it came from"
        )
    with st.expander("About the project"):
        st.markdown("""
**Document Q&A Chatbot** lets you upload a PDF and ask questions about it in plain English.

**How it works**

1. **Upload:** you add a PDF and the app reads the text page by page.
2. **Split:** the text is cut into chunks of about 1,000 characters, with a 200-character overlap so sentences at the edges aren't lost.
3. **Embed:** each chunk is turned into a list of numbers (an embedding) by a Gemini embedding model. Chunks with similar meaning get similar numbers.
4. **Store:** the embeddings, the chunk text, and the file name and page number are saved in ChromaDB on your computer.
5. **Retrieve:** when you ask a question, it is embedded the same way, and the 4 closest chunks are pulled from ChromaDB.
6. **Answer:** those chunks and your question go to a Gemini chat model, which is told to answer only from that text and to say "I don't know" if the answer isn't there.
7. **Show sources:** the answer appears with the file name and page number of each chunk used.

**Built with:** Python, LangChain, ChromaDB, Google Gemini API, Streamlit

**Good to know**

- Your chat history stays in the session and clears when you refresh the page.
- Scanned PDFs (images of text) can't be read, only PDFs with real selectable text.
- The free Gemini tier has rate limits, so very large files may take a little longer.
- Avoid uploading confidential documents while using the free tier.
""")
    st.link_button("GitHub link", GITHUB_URL)

st.divider()

# ---------- Chat ----------
if "vs" in st.session_state:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if q := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                ans, sources = answer(st.session_state.vs, q)
            st.write(ans)
            with st.expander("Sources"):
                for d in sources:
                    name = d.metadata.get("source_name", "document")
                    st.caption(f"{name}, page {d.metadata.get('page', '?')}")
                    st.write(d.page_content[:300] + "...")
        st.session_state.messages.append({"role": "assistant", "content": ans})
else:
    st.info("Upload a PDF and click Process documents to start chatting.")