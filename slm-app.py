from transformers import AutoModelForTokenClassification, AutoTokenizer
import streamlit as st

model = AutoModelForTokenClassification.from_pretrained("/content/drive/MyDrive/NER-BERT/model/pst-pdr-bc5cdr/biomed")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/NER-BERT/model/pst-pdr-bc5cdr/biomed")

def home_page():
    st.title("Name Entity Recognition using BERT")

    user_input = st.text_input("Input your text:")
    is_click = st.button(label="Process")

    if is_click and (user_input.strip() != ""):
        # Tokenisasi input
        inputs = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
        
        # Panggil model dengan input yang sudah di-tokenisasi
        with st.spinner('Processing...'):
            ner_output = model(**inputs)
        
        st.write(f"Found {len(ner_output)} entity(es):")
        for item in ner_output:
            st.markdown("* " + item["word"] + " ***[" + item["entity"] + "]***", unsafe_allow_html=True)

if __name__ == "__main__":
    home_page()
