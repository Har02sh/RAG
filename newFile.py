import fitz  # PyMuPDF
import re

def extract_text_by_section(pdf_path, output_file):
    doc = fitz.open(pdf_path)
    chunks = []
    current_chunk = []

    # Regex pattern for section headers (e.g., 1, 1.1, 1.1.1)
    section_pattern = re.compile(r"^\d+(\.\d+)*")  
    current_section = None

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a section heading
            match = section_pattern.match(line)
            if match:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                current_section = line  # Update current section
            current_chunk.append(line)

    # Store the last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # Save chunks to a file
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n{'-'*80}\n\n")

    print(f"Chunks saved to {output_file}")

# Usage Example:
pdf_path = r"D:\minimal-rag-master\docs_pdf\DSR vol 1.pdf"
output_file = "chunked_text.txt"
extract_text_by_section(pdf_path, output_file)
