import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text
pdf_path = r"D:\minimal-rag-master\docs_pdf\DSR vol 1.pdf"
doc = fitz.open(pdf_path)
text_chunks = []

for page in doc:
    text = page.get_text("text")
    text_chunks.extend(text.split("\n\n"))  # Chunk by paragraphs


with open("file.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text_chunks))
# Embed and store in FAISS
embeddings = np.array([model.encode(chunk) for chunk in text_chunks])
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"Stored {len(text_chunks)} chunks in FAISS.")
