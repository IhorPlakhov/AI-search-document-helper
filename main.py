from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def get_info() -> str:
    with open("instruction.txt", encoding="utf-8") as data:
        return data.read()


def main():

    summary_template = """
    You are an assistant that helps answer questions strictly based on the provided instruction context.
    Given the {information} on {question} you can add a chapter`s number of which
      have information on this question
    """

    llm = ChatOllama(
        temperature=0, model="hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q5_K_XL"
    )

    chain = PromptTemplate(template=summary_template) | llm | StrOutputParser()

    user_question = input("Enter your question: ")

    answer = chain.invoke(
        {
            "information": get_info(),
            "question": user_question,
        }
    )

    print(answer)


if __name__ == "__main__":
    main()
