import os
import re
import weaviate
from database import Database
from generation import Generation
from evaluation import Evaluation
from documentloader import PDFProcessor
from logger import Logger
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate




class Retriever:
    
    def __init__(self):
        """
        Initialize the Retriever instance with Weaviate, OpenAIEmbeddings, PDFProcessor, Generation, and Evaluation instances.
        """
        auth_config = weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))
        weaviate_client =  weaviate.Client(
            url=os.environ.get("WEAVIATE_URL"),
            auth_client_secret=auth_config
        )
        embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
        self.pdf = PDFProcessor('../data/Raptor Contract.docx.pdf')
        self.evaluation = Evaluation()
        self.weaviate_client = weaviate_client
        self.embeddings = embeddings
        self.generator = Generation("gpt-4-turbo-preview")
        self.pdf_context = self.pdf.process_pdf()
        
    def key_word(self, query: str):
        """
        Extract a keyword from the given query using a structured output prompt.

        Parameters:
            query (str): Query for keyword extraction.

        Returns:
            str: Extracted keyword.
        """
        keyword = ResponseSchema(name="keyword",
                    description="A single word that would be a good keyword to describe the question",)
        response_schemas = [keyword]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        template = open('../src/prompts/keywords.txt').read()
        prompt = ChatPromptTemplate.from_template(template=template)

        messages = prompt.format_messages(question=query, 
                                format_instructions=format_instructions)
        
        response = self.generator.chats(messages)
        output_dict = output_parser.parse(response.content)
        return output_dict.get('keyword')
    
    def key_word_search(self, keyword: str, data: list):
        """
        Search for documents containing the given keyword in the provided data.

        Parameters:
            keyword (str): Keyword to search for.
            data (list): List of documents.

        Returns:
            list: Matching documents.
        """
        matching_documents = [doc for doc in data if re.search(re.escape(keyword), doc, re.IGNORECASE)]
        return matching_documents
    
    def selecting_context(self, keyword_context: list, vectordb_context: list, query: str):
        """
        Select context documents for evaluation based on keyword and vector database context.

        Parameters:
            keyword_context (list): List of documents from keyword search.
            vectordb_context (list): List of documents from vector database.
            query (str): Query for ranking.

        Returns:
            list: Selected context documents for evaluation.
        """
        eval_docs = []
        rank = self.evaluation.ranking_query(keyword_context, query)
        for i in rank[0:3]:
            eval_docs.append(keyword_context[i])
        s = []
        for i in range(len(vectordb_context)):
            s.append(vectordb_context[i].page_content)
        rank = self.evaluation.ranking_query(s, query)
        print(rank)
        for i in rank[0:3]:
            eval_docs.append(s[i])
        return eval_docs
    
    def evaluate_context(self, query: str, eval_docs: list):
        """
        Evaluate context documents using the Evaluation class.

        Parameters:
            query (str): Query for evaluation.
            eval_docs (list): List of context documents.

        Returns:
            list: Valid context documents.
        """
        prompt = open('../src/prompts/generic-evaluation-prompt.txt').read()
        valid_context = []
        for doc in eval_docs:
            res = self.evaluation.evaluate(prompt=prompt, user_message=query, context=doc)
            if res == 'true':
                valid_context.append(doc)
                
        return valid_context
    
    def retrieve(self, query: str, file_path: str = '../data/Raptor Contract.docx.pdf'):
        """
        Retrieve valid context documents based on the given query and file path.

        Parameters:
            query (str): Query for retrieval.
            file_path (str): Path to the file.

        Returns:
            list: Valid context documents.
        """
        data = Database(weaviate_client=self.weaviate_client, embeddings=self.embeddings, file_path=file_path)
        key_word = self.key_word(query)
        keyword_context = self.key_word_search(keyword=key_word, data=self.pdf_context)
        context_data = data.retrieve(query)
        eval_docs = self.selecting_context(keyword_context=keyword_context, vectordb_context=context_data, query=query)
        valid_context = self.evaluate_context(query, eval_docs)
        return valid_context
