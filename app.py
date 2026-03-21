import streamlit as st
from llm import GroqLLM

# -- Page Configuration & Aesthetics --
st.set_page_config(
    page_title="Groq AI Intelligence - Advanced Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium "dark/neon" aesthetic or refined UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 15px;
        padding: 10px;
    }
    .stMarkdown h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(#f0f2f6, #6c7080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-content {
        background-color: #1a1c24;
    }
</style>
""", unsafe_allow_html=True)

# -- State Management --
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = GroqLLM()

# -- Sidebar Layout --
with st.sidebar:
    st.header("🧠 Groq AI Settings")
    st.divider()
    
    selected_model = st.selectbox(
        "AI Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0,
        help="Choose the underlying model for responses."
    )
    
    st.subheader("Fine-tuning")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, 
                           help="Lower values are more deterministic, higher values more creative.")
    max_tokens = st.slider("Max Length (tokens)", 128, 4096, 1024, 128)
    
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.info("💡 **Pro Tip:** Llama 3.3 70b is currently the most versatile model on Groq.")

# -- Main Chat Interface --
st.title("Groq Intelligence")
st.caption("Ultrafast In-context AI Chat Powered by GroqCloud Inference Engine")

# Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Message your AI assistant..."):
    # Append User Message to State
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and Stream AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # We prepare the messages including history
        # (Optional: Limit history if it gets too long)
        history = st.session_state.messages
        
        try:
            for chunk in st.session_state.llm.generate_response(
                messages=history,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Append Assistant Message to State
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c7080;'>Built with ❤️ using Groq and Streamlit</div>", 
    unsafe_allow_html=True
)