import re
import json
import textwrap

MAX_CHUNK_SIZE = 1000  # Change as needed

def hierarchical_chunking_resilient(text: str):
    chunks = []
    current_part = None
    current_chapter = None
    current_section = None
    section_text = []

    lines = text.splitlines()
    part_pattern = re.compile(r"^Part\s+([IVXLC]+)(?:\s*-\s*(.*))?$", re.IGNORECASE)
    chapter_pattern = re.compile(r"^Chapter\s+([IVXLC]+)(?:\s*-\s*(.*))?$", re.IGNORECASE)
    section_pattern = re.compile(r"^Section\s+(\d+[A-Z]?)(?:\s*-\s*(.*))?$", re.IGNORECASE)

    def save_chunk():
        if section_text:
            full_text = "\n".join(section_text).strip()
            label = " > ".join(filter(None, [current_part, current_chapter, current_section]))
            metadata = {
                "part": current_part,
                "chapter": current_chapter,
                "section": current_section,
            }

            # Split large sections into smaller chunks
            if len(full_text) > MAX_CHUNK_SIZE:
                for i, piece in enumerate(textwrap.wrap(full_text, MAX_CHUNK_SIZE)):
                    chunks.append({
                        "id": f"{label} [chunk {i+1}]",
                        "content": piece.strip(),
                        "metadata": metadata
                    })
            else:
                chunks.append({
                    "id": label or "Unlabeled Chunk",
                    "content": full_text,
                    "metadata": metadata
                })

    for line in lines:
        line = line.strip()
        if not line:
            continue

        part_match = part_pattern.match(line)
        chapter_match = chapter_pattern.match(line)
        section_match = section_pattern.match(line)

        if part_match:
            save_chunk()
            current_part = f"Part {part_match.group(1)}"
            if part_match.group(2):
                current_part += f" - {part_match.group(2)}"
            current_chapter = None
            current_section = None
            section_text = []

        elif chapter_match:
            save_chunk()
            current_chapter = f"Chapter {chapter_match.group(1)}"
            if chapter_match.group(2):
                current_chapter += f" - {chapter_match.group(2)}"
            current_section = None
            section_text = []

        elif section_match:
            save_chunk()
            current_section = f"Section {section_match.group(1)}"
            if section_match.group(2):
                current_section += f" - {section_match.group(2)}"
            section_text = []

        else:
            section_text.append(line)

    save_chunk()
    return chunks

def write_chunks_to_jsonl(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    input_file = r"D:\test pdfs\DSR vol 1.pdf"
    output_file = "chunks.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        legal_text = f.read()

    chunks = hierarchical_chunking_resilient(legal_text)
    write_chunks_to_jsonl(chunks, output_file)

    print(f"✅ Done! Wrote {len(chunks)} smaller chunks to {output_file}")

# if __name__ == "__main__":
#     input_file = r"D:\test pdfs\DSR vol 1.pdf"      # Replace with your actual file
#     output_file = "chunks.jsonl"

#     import fitz  # PyMuPDF
#     doc = fitz.open(input_file)
#     legal_text = ""
#     for page in doc:
#         legal_text += page.get_text("text")

#     chunks = hierarchical_chunking_resilient(legal_text)
#     write_chunks_to_jsonl(chunks, output_file)

#     print(f"✅ Done! Extracted {len(chunks)} chunks to {output_file}")
