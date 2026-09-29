import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

def answer(vectorstore, question: str):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
        temperature=0,
    )
    response = llm.invoke(PROMPT.format(context=context, question=question))
    return response.content, docs