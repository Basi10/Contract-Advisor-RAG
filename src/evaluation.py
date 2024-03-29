import numpy as np
from sentence_transformers import CrossEncoder
from generation import Generation

class Evaluation:
    
    def __init__(self):
        """
        Initialize the Evaluation instance with a Generation instance.
        """
        self.generator = Generation("gpt-3.5-turbo")
        
    def ranking_query(self, matching_documents: list, query: str):
        """
        Rank the matching documents based on a query using a CrossEncoder.

        Parameters:
            matching_documents (list): List of documents to rank.
            query (str): Query for ranking.

        Returns:
            list: Ranked indices of matching documents.
        """
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, doc] for doc in matching_documents]
        scores = cross_encoder.predict(pairs)
        new_scores = [original_index for original_index in np.argsort(scores)[::-1]]
        return new_scores   

    def evaluate(self, prompt: str, user_message: str, context: str, use_test_data: bool = False) -> str:
        """
        Evaluate the hallucination classification.

        Parameters:
            prompt (str): Prompt for generation.
            user_message (str): User message.
            context (str): Context for generation.
            use_test_data (bool): Flag to use test data (default is False).

        Returns:
            str: Classification of the hallucination.
        """
        API_RESPONSE = self.generator.get_completion(
            [
                {
                    "role": "system",
                    "content": prompt.replace("{Context}", context).replace("{Question}", user_message)
                }
            ],
            model='gpt-3.5-turbo',
            logprobs=True,
            top_logprobs=1,
        )

        system_msg = str(API_RESPONSE.choices[0].message.content)

        for i, logprob in enumerate(API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1):
            output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob.logprob}, \naccuracy: {np.round(np.exp(logprob.logprob)*100,2)}%\n'
            if system_msg == 'true' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
                classification = 'true'
            elif system_msg == 'false' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
                classification = 'false'
            else:
                classification = 'false'
        return classification
