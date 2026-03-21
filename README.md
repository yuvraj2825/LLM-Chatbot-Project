# LLM-Chatbot-Project

A fast, beautifully designed in-context AI Chatbot powered by the [GroqCloud Inference Engine](https://groq.com/) and [Streamlit](https://streamlit.io/).

## Features
- **Multiple State-of-the-Art Models**: Support for Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B, and Gemma.
- **Ultrafast Inference**: Near-instantaneous streaming responses powered by Groq's LPU inference engine.
- **Fine-Tuning Controls**: Adjust generation temperature and max tokens dynamically in the UI.
- **Premium Aesthetic**: Clean, dark-mode interface with floating inputs and smooth markdown rendering.

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**: Make sure your `.env` file is set up with your `GROQ_API_KEY`.
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```
