import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the CodeGen model and tokenizer

@st.cache_resource
def load_codegen_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

model = load_codegen_model()

# Load the Excel data
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# Function to retrieve relevant data based on a query
def retrieve_data(data, query):
    # Simple keyword-based retrieval
    relevant_data = data[data.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    return relevant_data

# Function to generate a response using the CodeGen model
def generate_response(prompt, model):
    try:
        response = model(prompt, max_length=150, do_sample=True, temperature=0.7)
        return response[0]['generated_text'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to create a prompt for the LLM based on user query and retrieved data
def create_prompt(retrieved_data, query):
    # Convert the retrieved data to a string format for the model
    data_str = retrieved_data.to_string(index=False)
    prompt = f"""
You are an AI assistant that provides air quality information based on the following data:

{data_str}

User query: {query}

Please provide a detailed response based on the data.
"""
    return prompt

# Streamlit app

# App title and description
st.title("🌬️ Air Quality Chatbot")
st.markdown("""
This app uses a language model to answer questions about air quality data.
""")

# File uploader for Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Load data from the uploaded Excel file
    data = load_data(uploaded_file)

    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the air quality..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Retrieve relevant data
        retrieved_data = retrieve_data(data, prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                # Create a prompt for the LLM
                llm_prompt = create_prompt(retrieved_data, prompt)
                response = generate_response(llm_prompt, model)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

