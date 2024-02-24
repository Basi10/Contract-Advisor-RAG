# Lizzy AI Contract Q&A RAG System

## Objectives

Lizzy AI is an early-stage Israeli startup aiming to develop the next-generation contract AI. As part of this mission, our objective is to build, evaluate, and improve a RAG (Retrieval-Augmented Generation) system for Contract Q&A. This system will enable users to chat with a contract and ask questions about it, with the ultimate goal of creating a fully autonomous contract bot that can draft, review, and negotiate contracts independently.

## Docker Usage

Prerequisites:

Ensure you have Docker installed on your machine.
Build and Run:

Clone this repository to your local machine.

```bash
https://github.com/Basi10/Contract-Advisor-RAG.git
```

Navigate to the project directory.

```bash
cd Contract-Advisor-RAG
```

Build the Docker image

```bash
docker build -t lizzy-ai-rag-system .
```

Run the Docker container

```bash
docker run -p 3000:3000 lizzy-ai-rag-system
```

Access the system through the provided localhost:3000

Chat with the contract and ask questions to get insights regarding the contract

## Local Development Usage

Prerequisites:

Python, Node.js and npm installed on your machine.
Setup and Run:

Clone this repository to your local machine.

```bash
https://github.com/Basi10/Contract-Advisor-RAG.git
```

Navigate to the project directory.

```bash
cd Contract-Advisor-RAG
```

Initialize the backend

```bash
python app.py
```

Navigate to the frontend

```bash
cd rag-frontend
```

Install Dependencies

```bash
npm i
```

Start the frontend:

```bash
npm run dev
```

Access the system through the provided URL in your web browser.
Chat with the contract and ask questions to evaluate the RAG system's performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributors

- Basilel Birru

## Acknowledgements

We would like to thank the Lizzy AI team for their hard work and dedication in developing this project.

## Resources

- [Lizzy AI Website](https://www.lizzyai.com)
- [Lizzy AI Demo Video](https://www.youtube.com/watch?v=1234567890)
