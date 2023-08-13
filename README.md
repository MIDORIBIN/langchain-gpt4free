# LangChain gpt4free

LangChain gpt4free is an open-source project that assists in building applications using LLM (Large Language Models) and provides free access to GPT4/3.5.

## Installation

To install langchain_g4f, run the following command:

```shell
pip install git+https://github.com/MIDORIBIN/langchain-gpt4free.git
```

This command will install langchain_g4f.

## Usage

Here is an example of how to use langchain_g4f

```python
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
```

The above sample code demonstrates the basic usage of langchain_g4f. Choose the appropriate model and provider, initialize the LLM, and then pass input text to the LLM object to obtain the result.

For other samples, please refer to the following [sample directory](./sample/).

## Support and Bug Reports

For support and bug reports, please use the GitHub Issues page. 

Access the GitHub Issues page and create a new issue. Select the appropriate label and provide detailed information.

## Contributing

To contribute to langchain_g4f, follow these steps to submit a pull request:

1. Fork the project repository.
2. Clone the forked repository to your local machine.
3. Make the necessary changes.
4. Commit the changes and push to your forked repository.
5. Create a pull request towards the original project repository.
