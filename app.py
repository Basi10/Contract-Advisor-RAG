import os
from flask import Flask, request, jsonify
from src.retriever import Retriever
import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from src.generation import Generation
from src.logger import Logger


auth_config = weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))
embedding = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
weaviate_client =  weaviate.Client(
        url=os.environ.get("WEAVIATE_URL"),
        auth_client_secret=auth_config,
        )

attributes = {
            'client': weaviate_client,
            'index_name': "RaptorContractdocx",
            'embedding': embedding,
            'text_key': 'text',
            'by_text': False
        }
new_weaviate_instance = Weaviate(**attributes)
instance = new_weaviate_instance
log = Logger('../logs/question_answer.log')

app = Flask(__name__)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']

    # Check if the file is present and has a PDF extension
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return "Invalid file", 400

    # Save the uploaded file to a desired location
    file.save('path/to/save/' + file.filename)

    # Optionally, you can perform further processing on the PDF file

    return "File uploaded successfully", 200


@app.route('/process_text', methods=['GET'])
def process_text():
    
    try:
        # Get the 'text' parameter from the request
        
        input_text = request.args.get('text', '')
       
        file_path = '../src/prompts/generic-evaluation-prompt.txt'
        r = Retriever(file_path="../data/Raptor Contract.docx.pdf",eval_path=file_path,weviate_instance=instance, model_name="gpt-3.5-turbo")
        context = r.retrieve_query(input_text)
        
        geneation = Generation("gpt-3.5-turbo")
        
        answer = geneation.generate_answer(context=context,question=input_text)

        return jsonify({'result': answer})

    except Exception as e:
        # Handle any exceptions and return an error response
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
