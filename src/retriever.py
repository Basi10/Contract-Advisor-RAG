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
        self.genertor = Generation("gpt-4-turbo-preview")
        
        
    def key_word(self,query: str):
        keyword = ResponseSchema(name="keyword",
                    description="A single word that would be a good keyword to describe the question",)
        response_schemas = [keyword]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        template = open('../src/prompts/keywords.txt').read()
        prompt = ChatPromptTemplate.from_template(template=template)

        messages = prompt.format_messages(question=query, 
                                format_instructions=format_instructions)
        
        response = self.genertor.chats(messages)
        output_dict = output_parser.parse(response.content)
        return output_dict.get('keyword')
    
    def key_word_search(self,keyword: str, data: list):
        matching_documents = [doc for doc in data if re.search(re.escape(keyword), doc, re.IGNORECASE)]
        return matching_documents
    
    def selecting_context(self,keyword_context: list, vectordb_context: list, query: str):
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
    
    def evaluate_context(self,query: str, eval_docs: list):
        prompt = open('../src/prompts/generic-evaluation-prompt.txt').read()
        valid_context = []
        for doc in eval_docs:
            res = self.evaluation.evaluate(prompt=prompt, user_message=query, context=doc)
            if res == 'true':
                valid_context.append(doc)
                
        return valid_context
    
    def retrieve(self,query: str, file_path: str):
        data = Database(weaviate_client=self.weaviate_client, embeddings=self.embeddings, file_path=file_path)
        pdf_context = self.pdf.process_pdf()
        key_word = self.key_word(query)
        keyword_context = self.key_word_search(keyword=key_word, data=pdf_context)
        context_data = data.retrieve(query)
        eval_docs = self.selecting_context(keyword_context=keyword_context, vectordb_context=context_data, query=query)
        valid_context = self.evaluate_context(query, eval_docs)
        return valid_context
        
        
            
        
    
        
        