from g4f import Provider, Model
from langchain.llms.base import LLM
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain_g4f import G4FLLM


def main():
    llm: LLM = G4FLLM(
        model=Model.gpt_35_turbo,
        provider=Provider.Aichat,
    )
    prompt_template = PromptTemplate(
        input_variables=["location"],
        template="Where is the best tourist attraction in {location}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    print(chain("tokyo"))


if __name__ == "__main__":
    main()
