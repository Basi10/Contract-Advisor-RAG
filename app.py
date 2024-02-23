from flask import Flask, request

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

if __name__ == '__main__':
    app.run(debug=True)
