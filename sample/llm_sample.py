from g4f import Provider, models
from langchain.llms.base import LLM

from langchain_g4f import G4FLLM


def main():
    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.Aichat,
    )

    res = llm("hello")
    print(res)  # Hello! How can I assist you today?


if __name__ == "__main__":
    main()
