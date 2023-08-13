from g4f import Provider, models
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms.base import LLM

from langchain_g4f import G4FLLM


def main():
    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
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
    # {'input': 'tokyo', 'output': 'The train route from Tokyo Station to the Tokyo Skytree is:\n\n1. Take the JR Yamanote Line or the Keihin-Tohoku Line from Tokyo Station to Ueno Station.\n\n2. Transfer to the Tokyo Metro Hibiya Line at Ueno Station and take the train towards Kita-senju.\n\n3. Get off at Oshiage Station, which is the closest station to the Tokyo Skytree.\n\nThe journey takes approximately 20-25 minutes and costs around 240-280 yen depending on the train line and time of day.'}


if __name__ == "__main__":
    main()
