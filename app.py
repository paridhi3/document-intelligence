import os
import streamlit as st
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Load environment variables
load_dotenv()
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
model_id = os.getenv("DOCUMENT_CLASSIFIER_MODEL_ID")

# Initialize Azure Document Intelligence client
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Streamlit UI
st.set_page_config(page_title="Legal Document Classifier", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Document Classifier")
st.write("Upload a legal document (PDF or image) to classify as **Judgement** or **Non-Judgement**.")

uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.info("Processing document... Please wait ⏳")

    poller = client.begin_classify_document(
        model_id=model_id,
        document=uploaded_file.read()
    )
    result = poller.result()

    if not result.documents:
        st.error("⚠️ Could not classify this document. Try again with another file.")
    else:
        doc = result.documents[0]
        st.success(f"✅ Document classified as **{doc.doc_type}**")
        st.json({
            "category": doc.doc_type,
            "confidence": doc.confidence
        })
