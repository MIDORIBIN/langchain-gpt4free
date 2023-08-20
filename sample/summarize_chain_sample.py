from g4f import Provider, models
from langchain.chains.summarize import load_summarize_chain  # type: ignore
from langchain.document_loaders import PyPDFLoader

from langchain_g4f import G4FLLM


def summarize_pdf(llm: G4FLLM, pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary


def main():
    # pip install pypdf transformers

    llm = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.DeepAi,
    )
    summarize = summarize_pdf(llm, "https://arxiv.org/pdf/2304.09103.pdf")
    print(summarize)
    # None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
    # Token indices sequence length is longer than the specified maximum sequence length for this model (2567 > 1024). Running this sequence through the model will result in indexing errors
    # The given information includes multiple preprint papers accepted for the IEEE SIEDS 2023, discussing various aspects of ChatGPT, an AI language generation model developed by OpenAI. The papers explore the applications, opportunities, and threats of ChatGPT in fields such as business, industry, education, healthcare, infrastructure, environment, communication, arts, culture, lifestyle, and leisure. They also compare the performance of GPT-3.5 and GPT-4, finding that GPT-4 performs significantly better. However, it is acknowledged that ChatGPT lacks the same level of understanding, empathy, and creativity as humans and cannot fully replace them. Several concerns are raised, including bias, privacy, security, ethical issues, and potential errors associated with using ChatGPT. The responsible and ethical use of ChatGPT is emphasized, and strategies to mitigate the negative effects are discussed. The importance of ongoing research, collaboration, critical thinking, and human judgment in conjunction with AI technology is highlighted.


if __name__ == "__main__":
    main()
