import PyPDF2
from PIL import Image

input=PyPDF2.PdfFileReader(open("/home/saujanya/OCR/practice/final/ocr_test_pdf.pdf","rb"))
page0=input.getPage(0)
print(page0)