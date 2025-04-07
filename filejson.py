import fitz  # PyMuPDF
import re
import json

def extract_chunks_for_rag(pdf_path, output_file, max_chunk_size=1000):
    doc = fitz.open(pdf_path)
    chunks = []
    current_chunk = []
    
    section_pattern = re.compile(r"^\d+(\.\d+)*")  # Detects section numbers like 1, 1.1
    current_section = "Introduction"
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            match = section_pattern.match(line)
            if match:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({"section": current_section, "page": page_num, "text": chunk_text})
                    current_chunk = []
                current_section = line  # Update current section title

            current_chunk.append(line)

            # Split if chunk exceeds max size
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) > max_chunk_size:
                chunks.append({"section": current_section, "page": page_num, "text": chunk_text})
                current_chunk = []

    # Save last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({"section": current_section, "page": page_num, "text": chunk_text})

    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"Chunks saved to {output_file}")

# Usage
pdf_path = r"C:\Users\harsh\Downloads\TheArmyAct1950.pdf"
output_file = "rag_chunks.json"
extract_chunks_for_rag(pdf_path, output_file)
