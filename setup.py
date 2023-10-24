from setuptools import setup, find_packages


def load_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="langchain_g4f",
    version="0.1",
    packages=find_packages(),
    description="LangChain gpt4free is an open-source project that assists in building applications using LLM (Large Language Models) and provides free access to GPT4/3.5.",
    author="Alexander",
    author_email="alexandkorataev@gmain.com",
    url="https://github.com/AlexanderKorataev/langchain-gpt4free",
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
   install_requires=load_requirements("requirements.txt"), 
   python_requires=">=3.10",
)

