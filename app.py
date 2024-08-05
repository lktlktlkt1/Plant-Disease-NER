from cProfile import label
from turtle import onclick
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import streamlit as st

model = AutoModelForTokenClassification.from_pretrained("bertner")
tokenizer = AutoTokenizer.from_pretrained("bertner")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def infer(input, model):
    return model(input)

# model = innit_model()
# print(model('England'))

def home_page():
    st.title("Name entity recognition using BERT")

    input, button = st.beta_columns([4,1])

    with input:
        user_input = st.text_input("Press enter or button to process")
        ner_output = infer(user_input, ner_model)
    
    # st.write(user_input)
    with button:
        st.text("")
        st.text("")
        is_click = st.button(label="Process")
        if is_click and (user_input != ""):
            ner_output = infer(user_input, ner_model)
    st.write(f"Found {len(ner_output)} entity(es):")
    for item in ner_output:
        st.markdown("* " + item["word"] + " ***[" + item["entity"] + "]***", unsafe_allow_html=True)



if __name__ == "__main__":
    home_page()