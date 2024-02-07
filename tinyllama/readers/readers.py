from typing import Callable

from PyPDF2 import PdfReader


def pre_process_corpus_wrapper(parse_func: Callable[..., str]):
    def wrapper(*args, **kwargs):
        corpus = parse_func(*args, **kwargs)
        characters_to_keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !#$%&'[]()*+,-./:;<=>?@^_`{|}~\t\n\r"

        # initialize an empty result strings
        undesirable_characters = ""

        # iterate through each character in the input string
        for char in sorted(list(set(corpus))):
            # check if the character is in the list of characters to keep
            if char not in characters_to_keep:
                # if yes, add it to the result string
                undesirable_characters += char

        # removing undesirable characters
        for char in undesirable_characters:
            corpus = corpus.replace(char, "")
        return corpus

    return wrapper


def pre_process_corpus(corpus):
    allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !#$%&'[]()*+,-./:;<=>?@^_`{|}~\t\n\r"

    # Use list comprehension to filter characters
    filtered_text = [char for char in corpus if char in allowed_characters]

    # Join the filtered characters back into a string
    return "".join(filtered_text)


def get_pdf_text(pdf_path: str) -> str:
    """
    Reads a .pdf file.
    """

    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text(txt_path: str) -> str:
    """
    Reads a .txt file.
    """

    try:
        with open(txt_path, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{txt_path}' not found.")
        return ""
