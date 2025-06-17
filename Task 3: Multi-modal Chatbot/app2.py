import google.generativeai as genai
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv



# Load API Key
load_dotenv()
genai.configure(api_key="AIzaSyCn_BD5aJ7SpWLbDzsd-RH7sdYuAykmXR4")

# Initialize models
text_model = genai.GenerativeModel('gemini-1.5-pro-latest')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# ✅ Check Gemini Model Status
def check_model_status():
    try:
        test_response = text_model.generate_content("Hello, are you active?")
        if "hello" in test_response.text.lower():
            return True, "✅ Gemini API is working."
        else:
            return False, "⚠️ Unexpected response from Gemini."
    except Exception as e:
        return False, f"❌ Gemini API error: {str(e)}"

status_ok, status_msg = check_model_status()

# UI
st.set_page_config(page_title="Multi-Modal Chatbot", page_icon="🤖")
st.title("🤖 Multi-Modal Chatbot with Gemini")
st.info(status_msg if status_ok else status_msg)

