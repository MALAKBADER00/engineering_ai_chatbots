import streamlit as st
from rag import RAG

#Clear Chat History Function
def clear_chat() -> None:
    '''Clears the chat history'''
    if "messages" in st.session_state:
        st.session_state.messages = []  
     
def main() -> None:
    st.title('📑💭ASK About SQU Undergraduates Regulations')

    # Sidebar
    #chat history management
    if st.sidebar.button("Start New Chat"):
        clear_chat()

    
    #chat history 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    #display existing chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar='👤'):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message(message["role"], avatar='🤖'):
                st.markdown(message["content"])

            
    #user prompt        
    if prompt := st.chat_input("Ask a Question for your PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        #display user message in chat message container
        with st.chat_message("user",avatar='👤'):
            st.markdown(prompt)

        input  = {"question": prompt, "chat_history": []}
        input_context = {
            "question": prompt, 
            "chat_history": st.session_state.messages[:-1]  # Exclude the current question
        }
        rag = RAG(input=input_context)  # Initialize RAG with the input context

        response_text, _ = rag.answer_question()
        #get response

        if response_text is not None:
            #display assistant response in chat message container
            with st.chat_message("assistant",avatar='🤖'):
                st.markdown(response_text)
            
            #add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})  
main()