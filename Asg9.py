from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts.lang import tts_langs
from gtts import gTTS
import streamlit as st
import re
import os
from PyPDF2 import PdfReader

# Constants
API_KEY = "AIzaSyDGmiz57W57FfGlpX5oN_F2qidHDG9_86Q"  # Replace with your actual API key
STATIC_IMAGE_URL = "https://miro.medium.com/v2/resize:fit:1000/1*NmTQQ4TJpRH7L4K5b3yK_A.jpeg"

# Initialize LangChain and ChatGoogleGenerativeAI
langs = tts_langs().keys()
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Please always respond to the user's query in pure Urdu language."),
        ("human", "{human_input}"),
    ]
)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
chain = chat_template | model | StrOutputParser()

# Streamlit app layout
st.title("RAG Based Voice Assistant")
st.image(STATIC_IMAGE_URL, use_column_width=True)

# Document upload section
uploaded_file = st.file_uploader("Upload an Urdu PDF document for answering questions", type=["pdf"])

# Function to extract text from the uploaded PDF document
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# If a document is uploaded, extract its text
document_text = ""
if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)
    st.success("Document uploaded and processed successfully!")
else:
    st.warning("No document uploaded. The chatbot will answer questions without a document.")

# Custom CSS for colorful chat bubbles
st.markdown(
    """
    <style>
    .user-bubble {
        background: linear-gradient(135deg, #a4508b, #5f0a87);
        color: white;
        padding: 10px;
        border-radius: 15px;
        text-align: right;
        margin-left: 25%;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    .ai-bubble {
        background: linear-gradient(135deg, #f093fb, #f5576c, #4facfe);
        color: white;
        padding: 10px;
        border-radius: 15px;
        text-align: left;
        margin-right: 25%;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Voice recording section
st.subheader("Record Your Voice:")
text = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT")

# Handling voice input and generating a response
if text:
    st.subheader("Text Generating")
    with st.spinner("Converting to Speech..."):
        try:
            if document_text:
                # If document is uploaded, use it to answer the question
                response = chain.invoke({"human_input": f"Document: {document_text}. Question: {text}"})
            else:
                # If no document, proceed with the standard query
                response = chain.invoke({"human_input": text})

            # Clean the response to remove unwanted characters like '**'
            full_response = "".join(res or "" for res in response)
            cleaned_response = re.sub(r"\**\*|__", "", full_response)

            # Display the conversation in colorful chat bubbles
            st.markdown(f'<div class="user-bubble">{text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-bubble">{cleaned_response}</div>', unsafe_allow_html=True)

            # Convert cleaned text to speech
            tts = gTTS(text=cleaned_response, lang='ur')
            tts.save("output.mp3")
            st.audio("output.mp3")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.error("Could not recognize speech. Please speak again.")
