import os
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

class Generation:
            
    def __init__(self, model_name: str):
        """
        Initialize the Generation instance with OpenAI and ChatOpenAI.

        Parameters:
            model_name (str): Name of the model.
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.chat = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.0, model=model_name)
        
    def chats(self, message):
        """
        Get a response for a given message using the ChatOpenAI model.

        Parameters:
            message (str): Input message.

        Returns:
            str: Response from the model.
        """
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
        """
        Get the completion of the prompt using the OpenAI chat API.

        Parameters:
            messages (list[dict[str, str]]): List of message dictionaries.
            model (str): Name of the model.
            max_tokens (int): Maximum number of tokens in the completion.
            temperature (float): Sampling temperature for randomness.
            stop (str): Text to stop generation at.
            seed (int): Seed for randomness.
            tools (str): Additional tools to use.
            logprobs (int): Include log probabilities in the response.
            top_logprobs (int): Include top log probabilities in the response.

        Returns:
            str: Completion of the prompt.
        """
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
    
    def get_keyword(self, prompt , query):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "user", "content": prompt.format(query=query)},
        ]
        )
        return response.choices[0].message.content
