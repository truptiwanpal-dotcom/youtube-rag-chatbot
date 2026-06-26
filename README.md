# YouTube Video Chatbot

A RAG-based chatbot that lets you ask questions about any YouTube video using its transcript.

## How it works

```
YouTube URL → Fetch Transcript → Split into Chunks → Embed (OpenAI) → Store (FAISS)
                                                                              ↓
                                          Answer ← GPT-4o-mini ← Prompt ← Retrieve top-4 chunks
```

## Project files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app (main entry point) |
| `youtube_chatbot.py` | Core RAG logic (script version) |
| `youtube_chatbot_using_langchain.py` | LangChain chain version |
| `.env` | Your API keys (not committed to git) |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Clone the repo and create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Paste a YouTube video URL (the video must have English captions)
2. Click **Load Video** — the transcript is fetched and indexed
3. Type any question about the video and click **Get Answer**

## Requirements

- Python 3.9+
- OpenAI API key
- Internet connection (to fetch YouTube transcripts)
