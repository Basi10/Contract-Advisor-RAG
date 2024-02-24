.PHONY: env build

env:
	@echo "Creating .env file. Please fill in the values for VARIABLE1, VARIABLE2, and VARIABLE3:"
	@echo "WEAVIATE_API_KEY=" >> .env
	@echo "WEAVIATE_URL=" >> .env
	@echo "OPENAI_API_KEY=" >> .env
	@echo "Environment variables set in .env file. Fill in the values for VARIABLE1, VARIABLE2, and VARIABLE3 in the .env file."

build:
	docker build -t lizzy-ai-rag-system .
	docker run -p 3000:3000 lizzy-ai-rag-system
