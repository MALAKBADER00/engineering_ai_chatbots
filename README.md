# Sultan Qaboos University RAG Assistant 🇴🇲📘

This project is a Retrieval-Augmented Generation (RAG) assistant built with LangChain and Groq/OpenAI, designed to answer queries related to **Sultan Qaboos University Undergraduate Academic Regulations** based on PDF documents.

---

## 🔧 Setup Instructions

Follow these steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/MALAKBADER00/engineering_ai_chatbots.git
cd engineering_ai_chatbots
```

2. Set Up a Virtual Environment
bash
Copy
Edit
python -m venv venv
# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Add API Keys to .env
Create a .env file in the root directory and add your OpenAI and Groq API keys like this:

env
Copy
Edit
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
Make sure not to share this file publicly.
