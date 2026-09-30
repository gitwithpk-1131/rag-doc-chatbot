import os, tempfile
import streamlit as st
from dotenv import load_dotenv
from rag.loader import load_and_split
from rag.vectorstore import build_vectorstore
from rag.chain import answer

st.set_page_config(page_title="RAG Document Q&A Chatbot", layout="wide")
load_dotenv()

GITHUB_URL = "https://github.com/gitwithpk-1131/rag-doc-chatbot"


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

# ---------- Text shown when the mouse is placed on a button ----------
TECH = {
    "RAG": (
        "**RAG (Retrieval-Augmented Generation)**\n\n"
        "An LLM only knows what it was trained on, so it can't answer questions about your own PDFs. "
        "RAG fixes this by looking up the relevant parts of your documents first and handing them to the model along with the question. "
        "The model then answers from that text instead of from memory, which cuts down on made-up answers."
    ),
    "LangChain": (
        "**LangChain**\n\n"
        "A toolkit that gives you ready-made pieces for each step: document loaders (PDF, Word, text), text splitters, "
        "embedding wrappers, vector store wrappers, and chat model wrappers. Its real value is that these pieces are swappable. "
        "If you later replace ChromaDB with FAISS or Gemini with another model, most of your code stays the same."
    ),
    "Streamlit": (
        "**Streamlit**\n\n"
        "Turns a Python script into a web app with no HTML or JavaScript. "
        "For this project it provides the file uploader, the chat input, the chat message bubbles, and the buttons you see here."
    ),
    "Gemini API": (
        "**Gemini API**\n\n"
        "Google's AI. It turns text into numbers (embeddings) for searching, "
        "and it writes the final answer from the passages found."
    ),
    "ChromaDB": (
        "**ChromaDB**\n\n"
        "An open-source vector database that runs inside your Python program with no separate server. "
        "It stores each chunk's text, its embedding, and metadata like file name and page number, and it can search by similarity."
    ),
}

# ---------- Header: title on the left, technology buttons on the right ----------
head_left, head_right = st.columns([2, 3])
with head_left:
    st.markdown(
        "<h1 style='white-space:nowrap; font-size:2.2rem; margin:0; padding-top:0.6rem;'>"
        "RAG Document Q&amp;A Chatbot</h1>",
        unsafe_allow_html=True,
    )
with head_right:
    tech_cols = st.columns(len(TECH))
    for col, (label, text) in zip(tech_cols, TECH.items()):
        with col:
            st.button(label, help=text, key=f"tech_{label}", width="stretch")

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

GITHUB_LOGO = (
    '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>'
    "</svg>"
)
GITHUB_ROW = (
    f'<a href="{GITHUB_URL}" target="_blank" rel="noopener" '
    'style="display:flex; align-items:center; gap:0.6rem; padding:0.6rem 1rem; '
    "border:1px solid rgba(128,128,128,0.3); border-radius:0.5rem; "
    'color:inherit; text-decoration:none; font-size:0.95rem;">'
    f"{GITHUB_LOGO}<span>GitHub link</span></a>"
)

with right:
    with st.expander("How to use it"):
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
4. **Store:** the embeddings, the chunk text, and the file name and page number are saved in ChromaDB.
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
    st.markdown(GITHUB_ROW, unsafe_allow_html=True)

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