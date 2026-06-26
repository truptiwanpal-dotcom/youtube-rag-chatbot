from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import sys
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding="utf-8")
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

video_id = "Gfr50f6ZBvo"
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

try:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=["en"])
    transScript = " ".join(snippet.text for snippet in transcript)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transScript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    fetch_chunk = vector_store.get_by_ids(["de9f21e3-7513-440f-a080-b451cde85629"])

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
            input_variables = ['context', 'question']
        )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    chain_answer = parallel_chain.invoke("who is Demis")
   # print(chain_answer)

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    answer = main_chain.invoke("Can you summarize the video")
    print(answer)
except TranscriptsDisabled as e:
    print(e.cause)