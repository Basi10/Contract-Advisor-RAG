from logger import Logger
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

class PDFProcessor:
    def __init__(self, file_path, log_file_name="pdf_processor.log"):
        """
        Initialize the PDFProcessor with the specified file path.

        Parameters:
        - file_path (str): Path to the PDF file.
        - log_file_name (str): Name of the log file.
        """
        self.logger = Logger(log_file_name).get_app_logger()
        self.loader = UnstructuredPDFLoader(file_path)
        self.data = None
        self.pdf_texts = None
        self.character_split_texts = None
        self.token_split_texts = None

    def load_pdf_data(self):
        """
        Load data from the PDF using the specified loader.
        """
        try:
            self.data = self.loader.load()
            self.logger.info("PDF data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading PDF data: {e}")

    def preprocess_text(self):
        """
        Preprocess the text data by splitting it and removing empty strings.
        """
        try:
            text = self.data[0].page_content.split("\n\n")
            text = [t.strip() for t in text]
            self.pdf_texts = [t for t in text if t]
            self.logger.info("Text preprocessing completed.")
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {e}")

    def split_text(self):
        """
        Split the text into chunks using RecursiveCharacterTextSplitter and SentenceTransformersTokenTextSplitter.
        """
        try:
            character_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=350,
                chunk_overlap=100
            )
            self.character_split_texts = character_splitter.split_text('\n\n'.join(self.pdf_texts))
            self.logger.info(f"Text split using character splitter. Total chunks: {len(self.character_split_texts)}")

            token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
            self.token_split_texts = []
            for text in self.character_split_texts:
                self.token_split_texts += token_splitter.split_text(text)

            self.logger.info(f"Text split using token splitter. Total chunks: {len(self.token_split_texts)}")
        except Exception as e:
            self.logger.error(f"Error in text splitting: {e}")

    def process_pdf(self):
        """
        Process the PDF by loading data, preprocessing text, and splitting text into chunks.

        Returns:
        - list: List of token split texts.
        """
        try:
            self.load_pdf_data()
            self.preprocess_text()
            self.split_text()
            return self.token_split_texts
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return None

