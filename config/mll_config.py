# third party imports
from langchain_text_splitters import RecursiveCharacterTextSplitter

# define the path to the input pdf we will be using to analize
INPUT_PDFS = 'datasets\\pdf'

# define the path to the temporary output directories
CHROMA_PATH = 'output\\least'

# difine chunk size Text Splitter
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150, #20
    length_function=len,
    add_start_index=True,
    separators=["\n\n", "\n"],
)

QUERIES = [
    'What is the product name?(please extract product name or test sample or material)',
    'Please return the subject of the document out of the following possibilities:\
        1. Occlusive Patch test or Repeated Insult Patch Test \
        2. Ophthalmology test \
        3. SPF test \
        4. UVA test \
        5. Critical Wavelength test \
        6. Safety Assessment \
        7. None of the above'
]