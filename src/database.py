import os
from langchain_community.vectorstores import Weaviate
from typing import (
    Dict,
)

class Database:
    
    def __init__(self, weaviate_client, embeddings, file_path):
        
        file_name_without_extension, file_extension = os.path.splitext(os.path.basename(file_path))
        file_name_without_extension = file_name_without_extension.replace('.', ' ')
        new_file_name = file_name_without_extension.replace(' ', '')
        
        attributes = {
            'client': weaviate_client,
            'index_name': new_file_name,
            'embedding': embeddings,
            'text_key': 'text',
            'by_text': False
        }
        self.new_weaviate_instance = Weaviate(**attributes)
        self.file_path = new_file_name
        self.weaviate_client = weaviate_client
        self.embedding = embeddings
        
    def _default_schema(self, index_name: str, text_key: str) -> Dict:
        return {
            "class": index_name,
            "properties": [
                {
                    "name": text_key,
                    "dataType": ["text"],
                }
            ],
        }
        
    def upload_to_weaviate(self, token_split_texts):
        """
        Uploads tokenized text data to the Weaviate database.

        Parameters:
            token_split_texts (list): A list of tokenized text data.
        """
        file_name_without_extension, file_extension = os.path.splitext(os.path.basename(self.file_path))
        file_name_without_extension = file_name_without_extension.replace('.', ' ')
        new_file_name = file_name_without_extension.replace(' ', '')
        self.weaviate_client.schema.create_class(self._default_schema(os.path.basename(new_file_name), "text"))
        Weaviate.from_texts(texts=token_split_texts, client=self.weaviate_client, embedding=self.embedding, index_name= os.path.basename(new_file_name))

    
    def retrieve(self, query: str, k: int = 5):
        return self.new_weaviate_instance.similarity_search(query=query, k = k)
    
    
    
    def as_retriever(self):
        return self.new_weaviate_instance.as_retriever()
    
    