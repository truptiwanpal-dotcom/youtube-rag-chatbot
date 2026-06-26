import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url):
    """Pull the video ID out of any standard YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url.strip()          # assume raw ID was pasted


def build_qa_chain(video_id):
    """Fetch transcript, embed it, and return a ready-to-use RAG chain."""

    # 1. Get transcript
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en"])
    full_text = " ".join(snippet.text for snippet in transcript)

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    # 3. Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Retriever – fetch top-4 relevant chunks for a question
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 5. Prompt
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant that answers questions about a YouTube video.
        Answer ONLY from the transcript context below.
        If the context is insufficient, say you don't know.

        Context:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    # 6. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # 7. Chain: retrieve → format → prompt → llm → parse
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("YouTube Video Chatbot")
st.write("Paste a YouTube URL, load the video, then ask anything about it.")

youtube_url = st.text_input("YouTube Video URL")

if st.button("Load Video"):
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(youtube_url)
        with st.spinner("Fetching transcript and building index..."):
            try:
                qa_chain = build_qa_chain(video_id)
                st.session_state.qa_chain = qa_chain
                st.success("Video loaded! You can now ask questions below.")
            except TranscriptsDisabled:
                st.error("This video has no English transcript available.")
            except Exception as e:
                st.error(f"Error: {e}")

question = st.text_input("Ask a question about the video")

if st.button("Get Answer"):
    if "qa_chain" not in st.session_state:
        st.warning("Please load a video first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.invoke(question)
        st.markdown("### Answer")
        st.write(answer)
