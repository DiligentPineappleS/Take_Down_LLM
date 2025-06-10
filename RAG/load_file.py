
import fitz
import docx
import re
def extract_file_text(file_path):
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc :
            text += page.get_text()
    elif file_path.endswith('.doc'):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text +"\n"
    else:
        raise ValueError("文件格式")
    text = re.sub(r'\s',' ',text).strip()
    return text


class Text_Extrator:
    @staticmethod
    def extract_text(file_path):



