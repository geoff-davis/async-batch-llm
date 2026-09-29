"""Example demonstrating async-batch-llm with LangChain integration.

This example shows how to create custom strategies that integrate with LangChain
runnables (LCEL chains such as ``prompt | llm``), including a small RAG
(Retrieval-Augmented Generation) pipeline. It targets LangChain 1.x.

Install dependencies:
    pip install 'async-batch-llm' 'langchain-core>=1' 'langchain-openai' 'langchain-anthropic'
"""

import asyncio
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


def _message_tokens(message: AIMessage) -> TokenUsage:
    """Read token counts from a chat model's ``usage_metadata`` when present."""
    usage = message.usage_metadata or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


class LangChainStrategy(LLMCallStrategy[str]):
    """Strategy for running a LangChain ``prompt | chat_model`` chain."""

    def __init__(self, chain: Runnable):
        """
        Initialize LangChain strategy.

        Args:
            chain: A runnable that takes ``{"input": ...}`` and returns an AIMessage
        """
        self.chain = chain

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state=None
    ) -> tuple[str, TokenUsage, None]:
        """Execute the LangChain chain.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        message = await self.chain.ainvoke({"input": prompt})
        return message.text, _message_tokens(message), None


# Example 1: Simple LangChain chain with OpenAI
async def example_langchain_openai_chain():
    """Example using LangChain with OpenAI model."""
    print("\n" + "=" * 60)
    print("Example 1: LangChain + OpenAI Chain")
    print("=" * 60 + "\n")

    # Create LangChain LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # reads OPENAI_API_KEY

    # Create prompt template
    template = """You are a helpful assistant that answers questions concisely.

Question: {input}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    # Create chain
    chain = prompt | llm

    # Create strategy
    strategy = LangChainStrategy(chain=chain)

    # Configure processor
    config = ProcessorConfig(concurrency=3, attempt_timeout=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is the capital of Japan?",
            "Explain photosynthesis briefly.",
            "Who wrote 'Romeo and Juliet'?",
        ]

        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"question_{i}",
                    strategy=strategy,
                    prompt=question,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nResults:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 2: LangChain with Anthropic
async def example_langchain_anthropic():
    """Example using LangChain with Anthropic Claude."""
    print("\n" + "=" * 60)
    print("Example 2: LangChain + Anthropic Claude")
    print("=" * 60 + "\n")

    # Create LangChain LLM
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=1.0)  # reads ANTHROPIC_API_KEY

    # Create prompt template for summarization
    template = """Please summarize the following text in 2-3 sentences:

{input}

Summary:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    # Create strategy
    strategy = LangChainStrategy(chain=chain)

    # Configure processor
    config = ProcessorConfig(concurrency=2, attempt_timeout=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        documents = [
            """
            Artificial intelligence (AI) is transforming healthcare through improved
            diagnostics, personalized treatment plans, and drug discovery. Machine
            learning algorithms can analyze medical images with high accuracy, often
            detecting patterns that human radiologists might miss. AI is also being
            used to predict patient outcomes and optimize hospital operations.
            """,
            """
            Renewable energy sources like solar and wind power are becoming increasingly
            cost-competitive with fossil fuels. The technology has improved dramatically
            over the past decade, with solar panel efficiency increasing and costs
            decreasing. Many countries are now investing heavily in renewable
            infrastructure to meet climate goals.
            """,
        ]

        for i, doc in enumerate(documents):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"doc_{i}",
                    strategy=strategy,
                    prompt=doc,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nSummaries:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 3: RAG with LangChain and an in-memory vector store
async def example_langchain_rag():
    """Example using a LangChain RAG pipeline with batch processing."""
    print("\n" + "=" * 60)
    print("Example 3: LangChain RAG Pipeline")
    print("=" * 60 + "\n")

    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    class RAGStrategy(LLMCallStrategy[str]):
        """Retrieve context, then answer with a ``prompt | llm`` chain."""

        def __init__(self, retriever: Runnable, answer_chain: Runnable):
            self.retriever = retriever
            self.answer_chain = answer_chain

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state=None
        ) -> tuple[str, TokenUsage, None]:
            docs = await self.retriever.ainvoke(prompt)
            context = "\n".join(doc.page_content for doc in docs)
            message = await self.answer_chain.ainvoke({"context": context, "question": prompt})
            return message.text, _message_tokens(message), None

    # Sample documents for our knowledge base
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are inspired by the structure of the human brain.",
        "Natural language processing (NLP) enables computers to understand human language.",
        "Deep learning uses multiple layers of neural networks for complex pattern recognition.",
    ]

    # Create embeddings and vector store (the documents are short, so no splitting)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = await InMemoryVectorStore.afrom_texts(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Create LLM and answer chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    answer_prompt = PromptTemplate.from_template(
        "Answer the question using only this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Create strategy
    strategy = RAGStrategy(retriever=retriever, answer_chain=answer_prompt | llm)

    # Configure processor
    config = ProcessorConfig(concurrency=2, attempt_timeout=30.0)

    # Process questions
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is Python?",
            "How do neural networks work?",
            "What is the relationship between AI and machine learning?",
        ]

        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"rag_question_{i}",
                    strategy=strategy,
                    prompt=question,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nRAG Answers:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 4: Different chains for different item types
async def example_langchain_multi_chain():
    """Example using different LangChain chains for different task types."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple LangChain Chains")
    print("=" * 60 + "\n")

    # Create different LLMs and chains
    openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    anthropic_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=1.0)

    # Chain for factual questions
    fact_template = """Answer this factual question concisely:

{input}

Answer:"""
    fact_chain = PromptTemplate.from_template(fact_template) | openai_llm

    # Chain for creative tasks
    creative_template = """Write a creative response to this prompt:

{input}

Creative response:"""
    creative_chain = PromptTemplate.from_template(creative_template) | anthropic_llm

    # Create strategies
    fact_strategy = LangChainStrategy(chain=fact_chain)
    creative_strategy = LangChainStrategy(chain=creative_chain)

    # Configure processor
    config = ProcessorConfig(concurrency=4, attempt_timeout=30.0)

    # Process mixed task types
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        # Factual questions
        await processor.add_work(
            LLMWorkItem(
                item_id="fact_1",
                strategy=fact_strategy,
                prompt="What is the speed of light?",
            )
        )
        await processor.add_work(
            LLMWorkItem(
                item_id="fact_2",
                strategy=fact_strategy,
                prompt="When was the Declaration of Independence signed?",
            )
        )

        # Creative tasks
        await processor.add_work(
            LLMWorkItem(
                item_id="creative_1",
                strategy=creative_strategy,
                prompt="Write a haiku about artificial intelligence.",
            )
        )
        await processor.add_work(
            LLMWorkItem(
                item_id="creative_2",
                strategy=creative_strategy,
                prompt="Describe a futuristic city in one sentence.",
            )
        )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nResults by type:")

    for item_result in result.results:
        if item_result.success:
            task_type = "FACT" if item_result.item_id.startswith("fact") else "CREATIVE"
            print(f"\n[{task_type}] {item_result.item_id}:")
            print(f"  {item_result.output}")


async def main():
    """Run all examples."""
    # Check for required API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("Error: At least one API key must be set:")
        print("  - OPENAI_API_KEY for OpenAI examples")
        print("  - ANTHROPIC_API_KEY for Anthropic examples")
        return

    # Run examples based on available API keys
    if has_openai:
        await example_langchain_openai_chain()
        await example_langchain_rag()

    if has_anthropic:
        await example_langchain_anthropic()

    if has_openai and has_anthropic:
        await example_langchain_multi_chain()


if __name__ == "__main__":
    asyncio.run(main())
