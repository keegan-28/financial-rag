from langchain.retrievers import RePhraseQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

from src.rag.database_interaction.embedding_model import llm
from src.rag.database_interaction.vector_store import vector_store
from src.rag.retrieval.prompt_factory import REPHRASE_PROMPT, QUERY_PROMPT
from src.rag.retrieval.utils import (
    CitedSentences,
    format_cited_answer,
    fetch_parent_docs_wrapper,
    format_docs_with_weight,
)

rephrase_query = RunnableSequence(REPHRASE_PROMPT | llm | StrOutputParser())

child_retriever = RePhraseQueryRetriever(
    retriever=vector_store.as_retriever(), llm_chain=rephrase_query
)

retrieve = RunnableParallel(
    docs=child_retriever,
    query=RunnablePassthrough(),
    original_query=RunnablePassthrough(),
)

fetch_parents = RunnableLambda(fetch_parent_docs_wrapper)

format_context = RunnableParallel(
    context=RunnableLambda(lambda d: format_docs_with_weight(d["docs"])),
    query=RunnableLambda(lambda d: d["query"]),
    original_query=RunnableLambda(lambda d: d["original_query"]),
)

query_llm = RunnableParallel(
    answer=QUERY_PROMPT
    | llm.with_structured_output(CitedSentences)
    | format_cited_answer,
    context=RunnableLambda(lambda d: d["context"]),
    query=RunnableLambda(lambda d: d["query"]),
    original_query=RunnableLambda(lambda d: d["original_query"]),
)

chain = retrieve | fetch_parents | format_context | query_llm

if __name__ == "__main__":
    try:
        while True:
            user_query = input("Enter query (type 'quit' or CTRL+C to quit): ")

            if user_query.lower() == "quit":
                break

            result = chain.invoke({"user_query": user_query})
            print(result["answer"])
    except KeyboardInterrupt:
        print("Qutting....")
