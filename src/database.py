import os
import json
import weaviate

class WeaviatePDFManager:
    def __init__(self, weaviate_url, weaviate_api_key, openai_api_key):
        """
        Initialize the PDFUploader with Weaviate connection details.

        Parameters:
        - weaviate_url (str): URL of the Weaviate instance.
        - weaviate_api_key (str): API key for Weaviate authentication.
        - openai_api_key (str): API key for OpenAI authentication.
        """
        auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
        self.weaviate_client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": openai_api_key,
            }
        )

    def create_schema(self, schema, class_name):
        """
        Create a schema for a Weaviate class.

        Parameters:
        - class_name (str): Name of the Weaviate class.

        Raises:
        - weaviate.WeaviateException: If an error occurs during schema creation.
        """
        try:
            self.weaviate_client.schema.create_class(schema)
            print(f"Schema created successfully for class: {class_name}")
        except weaviate.WeaviateException as e:
            print(f"Error creating schema for class {class_name}: {e}")

    def upload_pdf(self, class_name, result_sections):
        """
        Upload PDF data to Weaviate.

        Parameters:
        - class_name (str): Name of the Weaviate class.
        - result_sections (list): List of text sections to upload.

        Raises:
        - weaviate.WeaviateException: If an error occurs during data upload.
        """
        data_objs = [{"text": result_sections}]

        batch_size = 256
        with self.weaviate_client.batch.configure(batch_size=batch_size) as batch:
            try:
                for data_obj in data_objs:
                    batch.add_data_object(
                        data_obj,
                        class_name,
                    )
                print(f"Data uploaded successfully to class: {class_name}")
            except weaviate.WeaviateException as e:
                print(f"Error uploading data to class {class_name}: {e}")



    def query_data(self, class_name, query_text, limit=5):
        """
        Query data from Weaviate.

        Parameters:
        - class_name (str): Name of the Weaviate class.
        - query_text (str): Text for the query.
        - limit (int): Limit the number of query results.

        Returns:
        - dict: Result of the Weaviate query.

        Raises:
        - weaviate.WeaviateException: If an error occurs during the query.
        """
        query = self.weaviate_client.query.get(class_name, ["text"]).with_hybrid(query=query_text).with_limit(limit)
        try:
            result = query.do()
            print(f"Query executed successfully for class: {class_name}")
            return result
        except weaviate.WeaviateException as e:
            print(f"Error executing query for class {class_name}: {e}")
            return {}

