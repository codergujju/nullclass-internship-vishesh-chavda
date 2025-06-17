import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
import time 
from langchain_ollama import OllamaLLM

model={
    'LLaMa2 2-13B': 'llama3.2:latest',
    'Mistral 7B': 'Mistral:latest',
    'Falcon 7B' : 'Falcon:latest'
}

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     
    " 🤖ིྀ Artical Generator ChatBot 🤖ིྀ"),
    
    ("user", "Question: {question}")
])

    

st.title("🧠 Article Generator")
st.markdown("Select a Model of different open-source LLMs ")
selected_model_name = st.selectbox("Choose Model", list(model.keys()))
model_id = model[selected_model_name]
user_prompt = st.text_area("Enter your article prompt")
generate_button = st.button("🚀 Generate Article")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "🤖ིྀ Article Generator ChatBot 🤖ིྀ"),
    ("user", "Question: {question}")
])


if "results" not in st.session_state:
    st.session_state["results"] = {} 
    
# Generate using Ollama
if generate_button:
    st.subheader("📝Generated Article")
    with st.spinner(f"Generating Article using {selected_model_name}..."):
        try:
            final_prompt = prompt_template.format(question=user_prompt)

            llm = OllamaLLM(model=model_id)

            start = time.time()
            response = llm.invoke(final_prompt)
            end = time.time()
            duration = round(end - start, 2)

            if "results" not in st.session_state:
                st.session_state["results"] = {} 
            st.session_state["results"][selected_model_name] = {
                "Response Time (s)": duration,
                "Characters Generated": len(response),
            }   
            st.success(f"Generated in {round(end - start, 2)} seconds")
            st.text_area("📄 Output", response, height=300)


        except Exception as e:
            st.error(f"⚠️ Error: {e}")


#Comparison Table

if st.session_state.results:
    st.markdown("### 📊 Model Performance Comparison")
    st.table([
        {"Model": model_name, **data}
        for model_name, data in st.session_state.results.items()
    ])
    