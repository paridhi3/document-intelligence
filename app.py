import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ClassifyDocumentRequest

# Load environment variables
load_dotenv()
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
classifier_id = os.getenv("DOCUMENT_CLASSIFIER_MODEL_ID")

# Initialize Azure Document Intelligence client
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Streamlit UI
st.set_page_config(page_title="Legal Document Classifier", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Document Classifier")
st.write("Upload one or more legal documents (PDF or image) to classify as **Judgement** or **Non-Judgement**.")

uploaded_files = st.file_uploader("Upload PDF or image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.info("Processing documents... Please wait ⏳")

    results = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        request = ClassifyDocumentRequest(bytes_source=file_bytes)

        try:
            poller = client.begin_classify_document(
                classifier_id=classifier_id,
                classify_request=request
            )
            result = poller.result()

            if result.documents:
                doc = result.documents[0]
                results.append({
                    "File Name": uploaded_file.name,
                    "Category": doc.doc_type,
                    "Confidence": f"{doc.confidence:.2%}"
                })
            else:
                results.append({
                    "File Name": uploaded_file.name,
                    "Category": "Unclassified",
                    "Confidence": "N/A"
                })

        except Exception as e:
            results.append({
                "File Name": uploaded_file.name,
                "Category": "Error",
                "Confidence": str(e)
            })

    # Display results in a table
    st.success("✅ Classification complete!")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
