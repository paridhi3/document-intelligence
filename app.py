import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import OpenAI

# Load environment variables
load_dotenv()

# Azure Document Intelligence setup
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# OpenAI / Azure OpenAI setup
llm_client = OpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"))

# --- Helper functions ---

from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import base64

def extract_text_from_file(file_bytes: bytes) -> str:
    """Extract text using Azure Document Intelligence OCR (prebuilt-read)."""
    request = AnalyzeDocumentRequest(bytes_source=file_bytes)

    poller = client.begin_analyze_document(
        model_id="prebuilt-read", 
        analyze_request=request
    )
    result = poller.result()

    text_output = []
    for page in result.pages:
        for line in page.lines:
            text_output.append(line.content)

    return "\n".join(text_output)




def process_with_llm(extracted_text: str) -> str:
    """Send extracted OCR text and images to LLM for classification (Judgement vs Non-Judgement)."""
    response = llm_client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[
        {
            "role": "system", 
            "content": (
                "You are a legal document classifier. Classify the given text as either 'Judgement' or 'Non-Judgement'. "
                "Judgments often explicitly state JUDGMENT, 'FINAL JUDGMENT,' 'ORDER AND JUDGMENT,' or 'DECREE' at the top. This is a very strong indicator. "
                "Case Caption: All court documents have this, but a judgment will clearly state the court, case name, and docket number. "
                "Signature Block: The judge's signature, date of judgment, and sometimes the clerk's attestation. "
                "Keywords indicating Non-Judgment: "
                "'COMPLAINT' (especially in the title), "
                "'MOTION to...', "
                "'BRIEF in Support/Opposition', "
                "'NOTICE of...', "
                "'STIPULATION', "
                "'AFFIDAVIT', "
                "'REQUEST for Production', "
                "'VERIFIED PETITION'."
            )
        },
        {
            "role": "user", 
            "content": extracted_text
        }
    ],
    temperature=0.0
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Legal Document Classifier", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Document Classifier")
st.write("Upload one or more legal documents (PDF or image) to classify as **Judgement** or **Non-Judgement** using OCR + LLM.")

uploaded_files = st.file_uploader(
    "Upload PDF or image files", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing documents... Please wait ⏳")

    results = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()

        try:
            # Extract text
            extracted_text = extract_text_from_file(file_bytes)

            if not extracted_text.strip():
                results.append({
                    "File Name": uploaded_file.name,
                    "Category": "Unclassified",
                    "Confidence": "No text found"
                })
                continue

            # Classify with LLM
            category = process_with_llm(extracted_text)

            results.append({
                "File Name": uploaded_file.name,
                "Category": category,
                "Confidence": "LLM output"  # optional since LLM doesn't return a numeric score
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

# import os
# import streamlit as st
# import pandas as pd
# from dotenv import load_dotenv
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import ClassifyDocumentRequest

# # Load environment variables
# load_dotenv()
# endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
# key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
# classifier_id = os.getenv("DOCUMENT_CLASSIFIER_MODEL_ID")

# # Initialize Azure Document Intelligence client
# client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# # Streamlit UI
# st.set_page_config(page_title="Legal Document Classifier", page_icon="⚖️", layout="centered")
# st.title("⚖️ Legal Document Classifier")
# st.write("Upload one or more legal documents (PDF or image) to classify as **Judgement** or **Non-Judgement**.")

# uploaded_files = st.file_uploader("Upload PDF or image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

# if uploaded_files:
#     st.info("Processing documents... Please wait ⏳")

#     results = []

#     for uploaded_file in uploaded_files:
#         file_bytes = uploaded_file.read()
#         request = ClassifyDocumentRequest(bytes_source=file_bytes)

#         try:
#             poller = client.begin_classify_document(
#                 classifier_id=classifier_id,
#                 classify_request=request
#             )
#             result = poller.result()

#             if result.documents:
#                 doc = result.documents[0]
#                 results.append({
#                     "File Name": uploaded_file.name,
#                     "Category": doc.doc_type,
#                     "Confidence": f"{doc.confidence:.2%}"
#                 })
#             else:
#                 results.append({
#                     "File Name": uploaded_file.name,
#                     "Category": "Unclassified",
#                     "Confidence": "N/A"
#                 })

#         except Exception as e:
#             results.append({
#                 "File Name": uploaded_file.name,
#                 "Category": "Error",
#                 "Confidence": str(e)
#             })

#     # Display results in a table
#     st.success("✅ Classification complete!")
#     df = pd.DataFrame(results)
#     st.dataframe(df, use_container_width=True)
