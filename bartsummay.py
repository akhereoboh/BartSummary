import streamlit as st
import transformers
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Set Streamlit page config
st.set_page_config(page_title="BART Text Summarizer", layout="wide")

# Load the trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    try:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model and tokenizer once
model, tokenizer = load_model()

st.title("📜 AI-Powered Text Summarizer")
st.markdown("**Using BART Transformer Model for Abstractive Summarization**")

# Text input with character limit
user_input = st.text_area("✍️ Enter text to summarize:", max_chars=1024, height=200)

if st.button("🚀 Summarize"):
    if user_input:
        if model and tokenizer:
            try:
                st.info("Generating summary... Please wait ⏳")
                inputs = tokenizer(user_input, return_tensors="pt", max_length=1024, truncation=True)

                # Move model to CUDA if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                inputs = {key: val.to(device) for key, val in inputs.items()}

                # Generate summary
                summary_ids = model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Display result
                st.subheader("✨ Summary:")
                st.success(summary)

            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
        else:
            st.error("Model failed to load. Please refresh the app.")
    else:
        st.warning("⚠️ Please enter text before clicking Summarize. ")