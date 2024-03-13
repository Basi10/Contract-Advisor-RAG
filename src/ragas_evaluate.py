import os
import weaviate
import pandas as pd
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_relevancy
from ragas.langchain.evalchain import RagasEvaluatorChain

class Ragas:
    def __init__(self):
        # Initialize Weaviate, embeddings, and other necessary components
        auth_config = weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))
        weaviate_client =  weaviate.Client(
            url=os.environ.get("WEAVIATE_URL"),
            auth_client_secret=auth_config
        )
        embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
        attributes = {
            'client': weaviate_client,
            'index_name': "RaptorContractdocx",
            'embedding': embeddings,
            'text_key': 'text',
            'by_text': False
        }
        self.instance = Weaviate(**attributes)

        # Define LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Define prompt template
        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use two sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(template)

    def run_qa(self, examples):
        # Set up RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=self.llm,
                                         chain_type="stuff",  # <-- Fill in the correct chain type
                                         chain_type_kwargs={"prompt": self.prompt},
                                         retriever=self.instance.as_retriever(),
                                         return_source_documents=True)

        # Run QA for the provided examples
        predictions = qa.batch(examples)
        return predictions

    def evaluate_metrics(self, result):
        # Create evaluation chains
        faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
        answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
        context_recall_chain = RagasEvaluatorChain(metric=context_recall)
        context_relevancy_chain = RagasEvaluatorChain(metric=context_relevancy)

        # Evaluate metrics
        faithfulness_score = faithfulness_chain(result)["faithfulness_score"]
        context_recall_score = context_recall_chain(result)["context_recall_score"]
        answer_rel_score = answer_rel_chain(result)['answer_relevancy_score']
        context_relevancy_score = context_relevancy_chain(result)['context_ relevancy_score']

        return faithfulness_score, context_recall_score, answer_rel_score, context_relevancy_score

    def main(self):
        # Sample questions and ground truths
        questions = ["Under what circumstances and to what extent the Sellers are responsible for a breach of representations and warranties?",
                     "How much is the escrow amount?",
                     "Does the Buyer need to pay the Employees Closing Bonus Amount directly to the Company's employees?",
                    ]
        ground_truths = [" Except in the case of fraud, the Sellers have no liability for breach of representations and warranties",
                        "The escrow amount is equal to $1,000,000.",
                        "No"]

        examples = [
            {"query": q, "ground_truths": [ground_truths[i]]}
            for i, q in enumerate(questions)
        ]

        # Run QA
        predictions = self.run_qa(examples)
        
        results_list = []

        # Evaluate metrics
        
        for i, result in enumerate(predictions):
            faithfulness_score, context_recall_score, answer_rel_score, context_relevancy_score = self.evaluate_metrics(result)
            results_list.append({
                "Question": result.get('query'),
                "Context": result.get('source_documents'),  
                "Ground_Truth": result.get('ground_truths'),
                "Answer": result.get('result'),
                "Faithfulness_Score": faithfulness_score,
                "Context_Recall_Score": context_recall_score,
                "Answer_Relevancy_Score": answer_rel_score,
                "Context_Relevancy_Score": context_relevancy_score
            })
            
            
        return pd.DataFrame(results_list)

if __name__ == "__main__":
    ragas_instance = Ragas()
    ragas_instance.main()
