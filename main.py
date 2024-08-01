import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import login

login(token="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")

# Initialize Streamlit framework
st.title('LangChain Demo with Hugging Face API')
input_text = st.text_input('Search whatever you want')

# Initialize Hugging Face pipeline
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Function to generate text
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(inputs['input_ids'], max_new_tokens=50, do_sample=True)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Display the generated text
if input_text:
    result = generate_text(input_text)
    st.write(result)
