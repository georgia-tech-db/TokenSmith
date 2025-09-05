import fitz  # PyMuPDF
import json
import re
import os

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from all pages of a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: The file '{pdf_path}' was not found.")
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(full_text)
    return full_text

def parse_toc_text(text):
    """
    Parses the raw text from a textbook's table of contents and structures it
    into a hierarchical JSON format.
    """
    toc_data = {}
    current_part_key = "Preamble"  # For content before the first main part
    toc_data[current_part_key] = {
        "pages": [],
        "subtopics": {}
    }

    # Find all individual quoted strings in the document.
    all_quoted_items = re.findall(r'"([^"]+)"', text, re.DOTALL)
    
    # Pair up titles with their page numbers.
    matches = []
    i = 0
    while i < len(all_quoted_items) - 1:
        current_item = all_quoted_items[i]
        next_item = all_quoted_items[i+1]
        
        # Heuristic: If the next item consists only of digits and whitespace,
        # it's considered a page number block for the current item.
        if re.fullmatch(r'[\d\s]+', next_item.strip()):
            matches.append((current_item, next_item))
            i += 2  # Consume both items as a pair
        else:
            i += 1  # Consume only the current item; it's a title without a page

    #Process the identified title-page pairs.
    for title_block, page_block in matches:
        # Clean up and split titles and pages that were clumped together
        titles = [t.strip() for t in title_block.strip().split('\n') if t.strip()]
        pages = [p.strip() for p in page_block.strip().split('\n') if p.strip()]

        # Associate each title with its corresponding page number
        for i, title in enumerate(titles):
            if i < len(pages):
                page_str = pages[i]
                try:
                    # A page number must be an integer
                    page_number = int(page_str)
                except ValueError:
                    print(f"Warning: Could not parse page number for '{title}'. Skipping.")
                    continue

                # Check if the entry is a main part (e.g., "II. Relational Databases")
                part_match = re.match(r'^[IVX]+\.\s+(.+)', title, re.IGNORECASE)
                if part_match:
                    current_part_key = title
                    if current_part_key not in toc_data:
                        toc_data[current_part_key] = {"pages": [], "subtopics": {}}
                    # Sometimes the part's starting page is listed with it
                    toc_data[current_part_key]["pages"].append(page_number)
                    continue
                
                # Handle special cases like "Introduction" which apply to the current part
                if title.lower() == 'introduction':
                     if current_part_key and current_part_key in toc_data:
                         toc_data[current_part_key]["pages"].append(page_number)
                else:
                    # Otherwise, it's a chapter or regular entry (subtopic)
                    if current_part_key in toc_data:
                        toc_data[current_part_key]["subtopics"][title] = {
                            "pages": [page_number]
                        }
                    else:
                         # This case handles entries before any part is defined
                         toc_data[current_part_key]["subtopics"][title] = {
                            "pages": [page_number]
                        }

    return toc_data

def parse_text_to_json(raw_text):
    """
    Parses a plain-text table of contents into a structured JSON format.
    """
    toc_data = {}
    # Start with a key for content before the first main part
    current_part_key = "Front Matter"
    toc_data[current_part_key] = {"pages": [], "subtopics": {}}

    lines = raw_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Regex to find main parts (e.g., "II. Relational Databases 140")
        part_match = re.match(r'^([IVX]+\.\s+.+?)\s+(\d+)$', line)
        # Regex to find any other entry ending with a page number
        entry_match = re.match(r'^(.*?)\s+(\d+)$', line)

        if part_match:
            title, page_str = part_match.groups()
            current_part_key = title
            if current_part_key not in toc_data:
                toc_data[current_part_key] = {"pages": [], "subtopics": {}}
            toc_data[current_part_key]["pages"].append(int(page_str))
        
        elif entry_match:
            title, page_str = entry_match.groups()
            
            # Handle "Introduction" which applies to the current part
            if title.lower() == 'introduction':
                 if current_part_key in toc_data:
                     toc_data[current_part_key]["pages"].append(int(page_str))
            else:
                # Otherwise, it's a chapter or regular entry (subtopic)
                if current_part_key in toc_data:
                    toc_data[current_part_key]["subtopics"][title] = {
                        "pages": [int(page_str)]
                    }
                else:
                    # This case handles entries before any part is defined
                    toc_data[current_part_key]["subtopics"][title] = {
                       "pages": [int(page_str)]
                   }
        else:
            print(f"Skipping line (no page number found): '{line}'")

    return toc_data

def main():
    """Main function to execute the PDF index parsing."""
    pdf_path = 'data/silberschatz-index.pdf'
    json_output_path = 'textbook_index.json'

    try:
        print(f"Extracting text from '{pdf_path}'...")
        raw_text = """
            Computer Science
            Volume 1
            Silberschatz−Korth−Sudarshan • Database System Concepts, Fourth Edition
            Front Matter 1
            Preface 1
            1. Introduction 11
            Text 11
            I. Data Models 35
            Introduction 35
            2. Entity−Relationship Model 36
            3. Relational Model 87
            II. Relational Databases 140
            Introduction 140
            4. SQL 141
            5. Other Relational Languages 194
            6. Integrity and Security 229
            7. Relational−Database Design 260
            III. Object−Based Databases and XML 307
            Introduction 307
            8. Object−Oriented Databases 308
            9. Object−Relational Databases 337
            10. XML 363
            IV. Data Storage and Querying 393
            Introduction 393
            11. Storage and File Structure 394
            12. Indexing and Hashing 446
            13. Query Processing 494
            14. Query Optimization 529
            V. Transaction Management 563
            Introduction 563
            15. Transactions 564
            16. Concurrency Control 590
            17. Recovery System 637
            VI. Database System Architecture 679
            Introduction 679
            18. Database System Architecture 680
            19. Distributed Databases 705
            20. Parallel Databases 750
            VII. Other Topics 773
            Introduction 773
            21. Application Development and Administration 774
            22. Advanced Querying and Information Retrieval 810
            23. Advanced Data Types and New Applications 856
            24. Advanced Transaction Processing 884
    """
        
        print("Parsing text and building structured JSON...")
        json_output_path = 'textbook_index.json'
    
        print("Parsing text and building structured JSON...")
        structured_index = parse_text_to_json(raw_text)
        
        # Write the structured data to a JSON file
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_index, f, indent=4)
            
        print(f"Successfully created '{json_output_path}'")
        print("You can view the contents in the 'textbook_index.json' file.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

