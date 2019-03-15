from wand.image import Image as wi
pdf = wi(filename='/home/saujanya/OCR/practice/final/ocr_test_pdf.pdf',resolution=300)
pdfImage=pdf.convert('jpeg')
n=0
for img in pdfImage.sequence:
    page=wi(image=img)
    page.save(filename=str(n)+".jpg")
    n=n+1