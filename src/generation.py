import os
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

class Generation:
            
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        llm_model = "gpt-3.5-turbo"
        self.chat = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),temperature=0.0, model=llm_model)
        
    def chats(self, message):
        return self.chat(message)
    
    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str = 'gpt-3.5-turbo-1106',
        max_tokens=1000,
        temperature=0,
        stop=None,
        seed=123,
        tools=None,
        logprobs=None,
        top_logprobs=None,
    ) -> str:
        """Return the completion of the prompt."""
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if tools:
            params["tools"] = tools

        completion = self.client.chat.completions.create(**params)
        return completion