import os
from groq import Groq
from dotenv import load_dotenv
from typing import Generator, Union, List, Dict

load_dotenv()

class GroqLLM:
    """
    A class to handle interactions with the Groq LLM API.
    """
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # We don't raise value error here to avoid app crashing during startup 
            # if the key is missing; we'll handle it in the UI.
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = True,
    ) -> Union[Generator[str, None, None], str]:
        """
        Generates a response from the Groq API based on the conversation history.
        
        Args:
            messages: A list of message dictionaries (role/content).
            model: The LLM model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            
        Returns:
            A generator that yields response chunks if stream=True, or a string otherwise.
        """
        if not self.client:
            error_msg = "Groq API key is missing. Please check your .env file."
            if stream:
                yield error_msg
            return error_msg

        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Groq API Error: {str(e)}"
            if stream:
                yield error_msg
            return error_msg