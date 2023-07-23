from g4f import Provider, Model
from langchain.llms.base import LLM
from langchain import PromptTemplate

from langchain_g4f import G4FLLM


def main():
    template = "What color is the {fruit}?"
    prompt_template = PromptTemplate(template=template, input_variables=["fruit"])

    llm: LLM = G4FLLM(
        model=Model.gpt_35_turbo,
        provider=Provider.Aichat,
    )

    res = llm(prompt_template.format(fruit="apple"))
    print(res)
    # The color of an apple can vary, but it is typically red, green, or yellow.

    res = llm(prompt_template.format(fruit="lemon"))
    print(res)
    # The color of a lemon is typically yellow.


if __name__ == "__main__":
    main()
