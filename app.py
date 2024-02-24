from flask import Flask, request, jsonify
from src.retriever import Retriever

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

        retrive = Retriever()
        context = retrive.retrieve(input_text)
        processed_text = f"You sent: {input_text}"

        # Return a JSON response to the frontend
        return jsonify({'result': processed_text})

    except Exception as e:
        # Handle any exceptions and return an error response
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
