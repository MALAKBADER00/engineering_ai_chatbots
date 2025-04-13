from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAG:
    def __init__(self, input):
        """Initializes the RAG pipeline, including document processing, retrieval, and LLM setup."""
        self.input = input
        
        # Load and split PDF documents
        self.loader = PyPDFDirectoryLoader("data/")
        self.pages = self.loader.load_and_split()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.documents = self.text_splitter.split_documents(self.pages)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Initialize FAISS vector store
        index_path = "faiss_index"
        
        if os.path.exists(index_path):
            print("Loading existing FAISS index...")
            self.vector = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new FAISS index...")
            self.vector = FAISS.from_documents(
                self.documents,
                self.embeddings
            )
            # Save the FAISS index
            self.vector.save_local(index_path)
        
        # Set up the retriever
        self.retriever = self.vector.as_retriever()
        
        # Set up the LLM
        #self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatGroq(model="gemma2-9b-it",api_key =GROQ_API_KEY)
        
        # Set up the output parser
        self.output_parser = StrOutputParser()
        
        # Set up the prompts
        self._setup_prompts()
        
        # Create the chains
        self._setup_chains()

    def _setup_prompts(self):
        """Defines the question reformulation and QA prompts."""

        self.instruction_to_system = """
        Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question.
        DO NOT answer the question, just reformulate it if needed and otherwise return it as it is"""
        
        self.question_maker_prompt = ChatPromptTemplate.from_messages([
            ("system", self.instruction_to_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Main QA prompt
        self.qa_system_prompt = """You are a knowledgeable and highly precise Assistant for Sultan Qaboos University (SQU), specializing in **Undergraduate Academic Regulations**. 
        Your role is to analyze university regulation PDFs and extract **only** the exact answers based on user queries.

        ### 🔹 Rules & Behavior:
        1️⃣ **Strict Answer Extraction**:  
        - Respond **exclusively** using the content from the provided documents.  
        - **Do NOT generate** any information beyond what is explicitly stated in the regulation PDFs.  

        2️⃣ **Smart PDF Section Identification**:  
        - Analyze the user query carefully.  
        - Choose the **most relevant section or article** in the document to locate the correct answer.  

        3️⃣ **Accurate Data Handling**:  
        - Extract and format **tables, rules, and numbered clauses** properly.  
        - Ensure regulation numbers, exceptions, and footnotes are clear and correctly presented.  

        4️⃣ **Strict Relevance**:  
        - If the answer **cannot** be found in the PDFs, clearly state:  
            **"I cannot find this information in the provided documents."**  

        5️⃣ **Citations & Transparency**:  
        - Always include the **exact source** of your response in this format:  
            **"Answer found in [Regulation Name], Page X."**  

        6️⃣ **Handling Conflicting Information**:  
        - If the documents contain **conflicting rules**, present **all relevant variations**.  
        - Mention discrepancies briefly as a **note** for clarity.  

        7️⃣ **Summarized Responses**:  
        - If the answer is lengthy or part of a large article, provide a **clear and concise summary** while maintaining legal precision.  

        8️⃣ **PDF Source Indication**:  
        - Always specify **which PDF(s) or sections were used** to generate the response.    

        ### 📝 FORMATTING REQUIREMENTS:
        - Use proper Markdown formatting in ALL your responses. This includes:
          - # Header 1 for main titles
          - ## Header 2 for major sections
          - ### Header 3 for subsections
          - **Bold text** for important information
          - *Italic text* for emphasis
          - - Bullet points for lists
          - 1. Numbered lists for sequential information
          - > Blockquotes for quoted text
          - `code` for technical terms or small code snippets
          - [text](url) for hyperlinks when relevant
          - Tables formatted with | and - for structured data
        
        ### 📊 TABLE EXAMPLE:
        ```
        | Header 1 | Header 2 | Header 3 |
        |----------|----------|----------|
        | Data 1   | Data 2   | Data 3   |
        | Data 4   | Data 5   | Data 6   |
        ```

        ---

        ### 📝 **Example Response Format with Markdown:**
        ```markdown
    
        ### Key Information
        - Point 1
        - Point 2
        
        ### Details
        Further details or explanation if needed.
        
        **Source:** [pdf Document Name only]
        ```
        ---

        ### 🔎 **Use the following retrieved context to answer the question:**  
        {context}  
        """
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    def _setup_chains(self):
        """Sets up the question chain, retrieval chain, and QA chain."""
        # Create the standalone question chain
        self.question_chain = self.question_maker_prompt | self.llm | self.output_parser

        # Create the retrieval chain
        self.retrieval_chain = RunnablePassthrough.assign(
        context=lambda x: "\n\n".join(
            doc.page_content for doc in self.retriever.get_relevant_documents(x["question"])
        )
    )


        # Create the final QA chain
        self.qa_chain = (
            self.retrieval_chain 
            | self.qa_prompt 
            | self.llm 
            | self.output_parser
        )

    def _convert_streamlit_messages_to_langchain(self, messages):
        """Convert Streamlit message format to LangChain message format"""
        langchain_messages = []
        for message in messages:
            if message["role"] == "user":
                langchain_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                langchain_messages.append(AIMessage(content=message["content"]))
        return langchain_messages    

    def answer_question(self):
        """Process a question and return an answer along with updated chat history."""
        question = self.input["question"]

        # Convert Streamlit messages to LangChain format
        chat_history = self._convert_streamlit_messages_to_langchain(
            self.input.get("chat_history", [])
        )

        # Get the answer using the QA chain
        answer = self.qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        return answer, self.input
    
# test 
""" rag = RAG(input={
    "question": "What is the GPA requirement for graduation?",
    "chat_history": [
        {"role": "user", "content": "What is the GPA requirement for graduation?"},
        {"role": "assistant", "content": "The GPA requirement for graduation is 2.0."}
    ]
})
response, input_context = rag.answer_question()
print("Response:", response)   """  