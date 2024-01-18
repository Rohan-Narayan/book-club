from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough


def create_template(conversation_history):
    template = """You will be engaging in question-answering tasks about a novel.
    You will be given a question, along with context, to aid in your response.
    Use the pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer consice.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    # template += "\n".join(conversation_history)
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return prompt, llm

def get_response(db, query, conversation_history):
    retriever = db.as_retriever()
    prompt, llm = create_template(conversation_history)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm
    )
    output = rag_chain.invoke(query)
    return output.content