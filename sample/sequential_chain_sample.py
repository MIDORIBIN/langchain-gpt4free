from g4f import Provider, Model
from langchain.llms.base import LLM
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

from langchain_g4f import G4FLLM


def main():
    llm: LLM = G4FLLM(
        model=Model.gpt_35_turbo,
        provider=Provider.DeepAi,
    )

    prompt_template_1 = PromptTemplate(
        input_variables=["location"],
        template="Please tell us one tourist attraction in {location}.",
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_template_1)

    prompt_template_2 = PromptTemplate(
        input_variables=["location"],
        template="What is the train route from Tokyo Station to {location}?",
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_template_2)

    simple_sequential_chain = SimpleSequentialChain(
        chains=[chain_1, chain_2], verbose=True
    )

    print(simple_sequential_chain("tokyo"))


if __name__ == "__main__":
    main()
