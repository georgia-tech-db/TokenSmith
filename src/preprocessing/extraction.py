import re
import json

def extract_sections_from_markdown(file_path):
    """
    Chunks a markdown file into sections based on '##' headings.

    Args:
        file_path (str): The path to the markdown file.

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
            section_content = parts[1].strip() if len(parts) > 1 else ''
            
            if section_content == '':
                continue
            sections.append({
                'heading': heading,
                'content': section_content
            })

    return sections

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
