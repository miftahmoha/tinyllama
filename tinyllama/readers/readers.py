from PyPDF2 import PdfReader


def filter(corpus):
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
    return filter(text)


def get_text(txt_path: str) -> str:
    """
    Reads a .txt file.
    """

    try:
        with open(txt_path) as file:
            content = file.read()
            return filter(content)
    except FileNotFoundError as e:
        raise e
