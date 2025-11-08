import re
import json
from typing import List, Dict

def extract_sections_from_markdown(
    file_path: str,
    exclusion_keywords: List[str] = None
) -> List[Dict]:
    """
    Chunks a markdown file into sections based on '##' headings.

    Args:
        file_path : The path to the markdown file.
        exclusion_keywords : List of keywords for excluding sections.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              section with 'heading' and 'content' keys.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    # The regular expression looks for lines starting with '## '
    # This will act as our delimiter for splitting the text.
    # We use a positive lookahead (?=...) to keep the delimiter (the heading)
    # in the resulting chunks.
    heading_pattern = r'(?=^## \d+(\.\d+)* .*)'
    chunks = re.split(heading_pattern, content, flags=re.MULTILINE)

    sections = []
    
    # The first chunk might be content before the first heading
    if chunks[0].strip():
        sections.append({
            'heading': 'Introduction',
            'content': chunks[0].strip()
        })

    # Process the rest of the chunks
    for chunk in chunks[1:]:
        if chunk.strip():
            # Split the chunk into the heading and the rest of the content
            parts = chunk.split('\n', 1)
            heading = parts[0].strip()

            # Exclude sections based on keywords if provided
            if exclusion_keywords is not None:
                if any(keyword.lower() in heading.lower() for keyword in exclusion_keywords):
                    continue

            section_content = parts[1].strip() if len(parts) > 1 else ''
            
            if section_content == '':
                continue
            else:
                # Clean the section content
                section_content = preprocess_extracted_section(section_content)

            sections.append({
                'heading': heading,
                'content': section_content
            })

    return sections


# Pre-compile regex patterns for better performance
_PAGE_NUMBER_RE = re.compile(r'Page \d+')

def preprocess_extracted_section(text: str) -> str:
    """
    Cleans a raw textbook section to prepare it for chunking.

    Args:
        text: The raw text of the section.

    Returns:
        str: The cleaned text.
    """
    # Replaces all newline occurences with single spaces
    text = text.replace('\n', ' ')

    # Removes page number markers (e.g., "Page 1232")
    text = _PAGE_NUMBER_RE.sub('', text)

    # Removes bold formatting markers (**)
    text = text.replace('**', '')

    # Normalizes all whitespace to single spaces
    cleaned_text = ' '.join(text.split())

    return cleaned_text


if __name__ == '__main__':
    # The user uploaded 'book_without_image.md'
    markdown_file = 'data/book_without_image.md'
    
    extracted_sections = extract_sections_from_markdown(markdown_file)

    if extracted_sections:
        print(f"Successfully extracted {len(extracted_sections)} sections.")

        # To save the output to a structured file like JSON:
        output_filename = 'data/extracted_sections.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_sections, f, indent=4, ensure_ascii=False)
        print(f"\nFull extracted content saved to '{output_filename}'")
