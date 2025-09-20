import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
Pichai Sundararajan (born June 10, 1972), better known as Sundar Pichai (pronounced: /ˈsʊndɜːr pɪˈtʃeɪ/), is an Indian-American business executive.[3][4] He is the chief executive officer (CEO) of Alphabet Inc. and its subsidiary Google.[5]

Pichai began his career as a materials engineer. Following a short stint at the management consulting firm McKinsey & Co., Pichai joined Google in 2004,[6] where he led the product management and innovation efforts for a suite of Google's client software products, including Google Chrome and ChromeOS, as well as being largely responsible for Google Drive. In addition, he went on to oversee the development of other applications such as Gmail and Google Maps.

Pichai was selected to become the next CEO of Google on August 10, 2015, after previously being appointed chief product officer by then CEO Larry Page. On October 24, 2015, he stepped into the new position at the completion of the formation of Alphabet Inc., the new holding company for the Google company family. He was appointed to the Alphabet Board of Directors in 2017.[7] As of May 2025, his net worth is estimated at US$1.1 billion.[8]
"""
    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A Short summary
    2. Two interesting facts about the person
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-5"
    )  # Temperature = 0 for deterministic output, 1 for creative output, so the higher the temperature the more creative the output
    chain = (
        summary_prompt_template | llm
    )  # Chain the prompt template with the LLM, | pipe operator, so it uses the summary prompt template's output is the input to the LLM (variable).
    response = chain.invoke(
        {"information": information}
    )  # Invoke the chain with the information , invoke is the runnable object
    print(response.content)  # Print the response


if __name__ == "__main__":
    main()
