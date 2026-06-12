"""
Chatbot Module
RAG-based conversational chatbot for resume Q&A using LangChain.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage


# RAG Chatbot prompt
CHATBOT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent resume analysis assistant. You help users understand and improve their resumes by answering questions based on the resume content provided.

**Your capabilities:**
- Answer specific questions about the resume (skills, experience, education, etc.)
- Suggest improvements to the resume
- Help identify strengths and weaknesses
- Provide career advice based on the resume content
- Help tailor the resume for specific roles

**Rules:**
- ONLY use information from the provided resume context to answer factual questions
- If the answer is not in the context, clearly say "I don't see this information in the resume"
- Be helpful, specific, and constructive
- Keep responses concise but thorough
- Use bullet points and formatting for clarity

**Resume Context:**
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string with source file metadata."""
    formatted = []
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown Resume")
        formatted.append(f"[Source Resume: {source_name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(retriever, llm):
    """
    Create a RAG (Retrieval-Augmented Generation) chain for resume Q&A.
    
    Args:
        retriever: Vector store retriever
        llm: LangChain LLM instance
    
    Returns:
        Runnable chain for Q&A
    """
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: [],
        }
        | CHATBOT_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def get_chat_response(
    retriever,
    llm,
    question: str,
    chat_history: list = None
) -> str:
    """
    Get a chatbot response for a given question with conversation history.
    
    Args:
        retriever: Vector store retriever
        llm: LangChain LLM instance
        question: User's question
        chat_history: List of previous (question, answer) tuples
    
    Returns:
        AI response string
    """
    if chat_history is None:
        chat_history = []

    # Convert chat history to LangChain message format
    messages = []
    for human_msg, ai_msg in chat_history:
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))

    # Retrieve relevant context
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build and invoke the chain
    chain = CHATBOT_PROMPT | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": messages,
    })

    return response
