from typing import Callable

from PyPDF2 import PdfReader

# should be integrated with a custom reader that outputs corpus as a string
def pre_process_corpus(parse_func: Callable[..., str]):
    def wrapper(*args, **kwargs):
        corpus = parse_func(*args, **kwargs)
        characters_to_keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?.,:;'\"\nʼ"

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


@pre_process_corpus
def get_pdf_text(pdf_path: str) -> str:
    """
    Reads a .pdf file.
    """

    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


@pre_process_corpus
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
