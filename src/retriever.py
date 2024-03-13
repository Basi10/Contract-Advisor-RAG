import os
import re
from openai import OpenAI
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import UnstructuredPDFLoader
import numpy as np
from evaluation import Evaluation
from generation import Generation

class Retriever:
    def __init__(self, file_path, eval_path, weviate_instance, model_name):
        """
        Initialize Retriever class.

        Parameters:
            file_path (str): Path to the PDF file.
            eval_path (str): Path to the evaluation file.
            weviate_instance: Instance of Weaviate.
            model_name (str): Name of the model.
        """
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.data = UnstructuredPDFLoader(file_path).load()
        self.file_content = self._read_file(eval_path)
        self.weaviate_instance = weviate_instance
        self.evaluation = Evaluation()
        self.generation = Generation(model_name)
        
        
    def _read_file(self, file_path):
        """
        Read contents of a file.

        Parameters:
            file_path (str): Path to the file.

        Returns:
            str: Contents of the file.
        """
        with open(file_path, 'r') as file:
            return file.read() 

    def retrieve_query(self, query):
        """
        Evaluate a query.

        Parameters:
            query (str): Query to be evaluated.

        Returns:
            list: List of relevant documents.
        """
        true_values = []
        matching_documents = self._find_matching_documents(query)

        if not matching_documents:
            return true_values

        true_values.extend(self._evaluate_matching_documents(query, matching_documents))
        true_values.extend(self._evaluate_similar_documents(query))

        return true_values

    def _find_matching_documents(self, query):
        """
        Find matching documents for a given query.

        Parameters:
            query (str): Query to search for matching documents.

        Returns:
            list: List of matching documents.
        """
        file = self._read_file('./prompts/keywords.txt')
        attempts = 3
        for attempt in range(attempts):
            keyword = self.generation.get_keyword(file,query)
            matching_documents = [doc for doc in self.data[0].page_content.split("\n\n") if re.search(re.escape(keyword), doc, re.IGNORECASE)]
            if matching_documents:
                return matching_documents
        return []

    def _evaluate_matching_documents(self, query, matching_documents):
        """
        Evaluate matching documents for a given query.

        Parameters:
            query (str): Query to be evaluated.
            matching_documents (list): List of matching documents.

        Returns:
            list: List of relevant matching documents.
        """
        true_values = []
        match_pairs = [[query, doc] for doc in matching_documents]
        scores = self.cross_encoder.predict(match_pairs)
        match_scores = [o for o in np.argsort(scores)[::-1]]
        
        for i in match_scores[0:3]:
            if i < len(matching_documents):
                classification = self.evaluation.evaluate(self.file_content, query, matching_documents[i])
                if classification == 'true':
                    true_values.append([matching_documents[i]])

        return true_values

    def _evaluate_similar_documents(self, query):
        """
        Evaluate similar documents for a given query.

        Parameters:
            query (str): Query to be evaluated.

        Returns:
            list: List of relevant similar documents.
        """
        true_values = []
        ans = self.weaviate_instance.similarity_search(query=query, k=10)
        pairs = [[query, doc.page_content] for doc in ans]
        scores = self.cross_encoder.predict(pairs)
        db_scores = [o for o in np.argsort(scores)[::-1]]

        for i in db_scores[0:3]:
            if i < len(ans):
                classification = self.evaluation.evaluate(self.file_content, query, ans[i].page_content)
                if classification == 'true':
                    true_values.append([ans[i]])

        return true_values
