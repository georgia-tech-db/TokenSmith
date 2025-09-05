import fitz  # PyMuPDF
import json
import re
import os

# --- Part 1: Index Parsing ---

def parse_text_to_json(raw_text):
    """
    Parses a plain-text table of contents into a structured JSON format.
    This is tailored to the format of 'silberschatz-index.pdf'.
    """
    toc_data = {}
    current_part_key = "Front Matter"
    toc_data[current_part_key] = {"pages": [], "subtopics": {}}
    lines = raw_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Regex for main parts (e.g., "I. Data Models 35")
        part_match = re.match(r'^([IVX]+\.\s+.+?)\s+(\d+)$', line)
        entry_match = re.match(r'^(.*?)\s+(\d+)$', line)

        if part_match:
            title, page_str = part_match.groups()
            current_part_key = title.strip()
            if current_part_key not in toc_data:
                toc_data[current_part_key] = {"pages": [], "subtopics": {}}
            toc_data[current_part_key]["pages"].append(int(page_str))
        elif entry_match:
            title, page_str = entry_match.groups()
            title = title.strip()
            if title.lower() == 'introduction':
                if current_part_key in toc_data:
                    toc_data[current_part_key]["pages"].append(int(page_str))
            else:
                toc_data[current_part_key]["subtopics"][title] = {"pages": [int(page_str)]}
    return toc_data

def parse_index_pdf(pdf_path):
    """Extracts text from the index PDF and parses it."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Index PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    # A manual correction for the provided index text format
    corrected_text = full_text.replace('Text\n11\nI.', 'Text 11\nI.')
    print(full_text)
    print(corrected_text)
    return parse_text_to_json(corrected_text)

def extract_text_from_pages(pdf_path, start_page, end_page):
    """
    Extracts raw text from a specified range of pages in the main textbook PDF.
    Note: PDF page numbers (e.g., page 141) may not match the document's internal
    page index (e.g., page 150). A manual offset might be needed.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Textbook PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    # Manual offset determined by inspecting the PDF
    page_offset = 12 
    
    for page_num in range(start_page + page_offset, end_page + page_offset + 1):
        if page_num < len(doc):
            page = doc.load_page(page_num -1) # Page index is 0-based
            full_text += page.get_text()
    doc.close()
    return full_text

def clean_textbook_extraction(raw_text):
    """Cleans raw text by removing common PDF artifacts."""
    cleaned_text = re.sub(r'Edited by Foxit PDF Editor.*?For Evaluation Only\.', '', raw_text, flags=re.DOTALL)
    
    # 2. De-hyphenate words broken across lines
    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)

    # 3. Reconstruct paragraphs
    cleaned_text = re.sub(r'(?<![.\?!"])\n(?![\n\t\r])', ' ', cleaned_text)
    
    # 4. Remove structural noise like headers and page numbers
    # Example: "Chapter 4 SQL" or "4.1 Background 141"
    cleaned_text = re.sub(r'^(Chapter\s\d+\s.*?|^\d+\.\d+\s.*?|\s\d+$)\n?', '', cleaned_text, flags=re.MULTILINE)
    
    # 5. Normalize whitespace
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def chunk_text_semantically(cleaned_text, metadata):
    """Chunks cleaned text into paragraphs and attaches metadata."""
    paragraphs = cleaned_text.split('\n\n')
    chunks = []
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            chunk = {"text": paragraph.strip(), "metadata": {**metadata, "chunk_number": i + 1}}
            chunks.append(chunk)
    return chunks

def main():
    """Main function to run the entire textbook processing pipeline."""
    index_pdf_path = 'data/silberschatz-index.pdf'
    textbook_pdf_path = 'data/chapters/silberschatz.pdf'
    output_filename = 'all_chunks.json'

    try:
        # 1. Parse the index to get the structure
        print(f"Parsing index from '{index_pdf_path}'...")
        # For this example, we'll manually define the text for stability
        index_text = """
            Front Matter 1
            Preface 1
            1. Introduction 11
            Text 11
            I. Data Models 35
            Introduction 35
            2. Entity-Relationship Model 36
            3. Relational Model 87
            II. Relational Databases 140
            Introduction 140
            4. SQL 141
        """
        # index_text =  parse_index_pdf(index_pdf_path)
        # print(index_text)
        # In a full run, you would use:
        # textbook_index = parse_index_pdf(index_pdf_path)
        textbook_index = parse_text_to_json(index_text)
        print("Index parsed successfully.")
        print(textbook_index)

        all_chunks = []

        print(f"\nProcessing content from '{textbook_pdf_path}'...")
        
        # Example: Extract, clean, and chunk just the 'SQL' chapter
        primary_topic = "II. Relational Databases"
        sub_topic = "4. SQL"
        
        start_page = textbook_index[primary_topic]["subtopics"][sub_topic]["pages"][0]
        # Let's assume the chapter is 5 pages long
        end_page = start_page + 4 
        
        print(f"Extracting '{sub_topic}' from pages {start_page}-{end_page}...")
        
        raw_text = extract_text_from_pages(textbook_pdf_path, start_page, end_page)

        cleaned_text = clean_textbook_extraction(raw_text)
        
        source_metadata = {
            "source_book": "Database System Concepts, Fourth Edition",
            "start_page": start_page,
            "end_page": end_page,
            "primary_topic": primary_topic,
            "sub_topic": sub_topic
        }
        semantic_chunks = chunk_text_semantically(cleaned_text, source_metadata)
        all_chunks.extend(semantic_chunks)
        
        print(f"Generated {len(semantic_chunks)} chunks for '{sub_topic}'.")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4)
            
        print(f"\nSuccessfully created '{output_filename}'.")
        print("This file contains the processed chunks from the sample chapter.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{index_pdf_path}' and '{textbook_pdf_path}' are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()

