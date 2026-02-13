from pathlib import Path
import re
import json
from typing import List, Dict
import sys
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

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
    numbering_pattern = re.compile(r"(\d+(?:\.\d+)*)")
    chunks = re.split(heading_pattern, content, flags=re.MULTILINE)

    sections = []
    
    # The first chunk might be content before the first heading.
    if chunks[0].strip():
        sections.append({
            'heading': 'Introduction',
            'content': chunks[0].strip()
        })

    # Process the rest of the chunks
    for chunk in chunks[1:]:
        if not chunk:
            continue
        if chunk.strip():
            # Split the chunk into the heading and the rest of the content
            parts = chunk.split('\n', 1)
            heading = parts[0].strip()
            heading = heading.lstrip('#').strip()
            heading = f"Section {heading}"

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
            
            # Determine the section level based on numbering
            match = numbering_pattern.search(heading)
            if match:
                assert match.lastindex >= 1, f"No capturing group for section number in heading: {heading}"

                section_number = match.group(1)

                assert isinstance(section_number, str) and section_number.strip(), \
                    f"Invalid section number extracted from heading: {heading}"

                assert all(part.isdigit() for part in section_number.split('.')), \
                    f"Malformed section numbering '{section_number}' in heading: {heading}"

                # Logic: "1.8.1" (2 dots) -> Level 3
                current_level = section_number.count('.') + 1
                try:
                    chapter_num = int(section_number.split('.')[0])
                except ValueError:
                    chapter_num = 0
            else:
                current_level = 1
                chapter_num = 0

            sections.append({
                'heading': heading,
                'content': section_content,
                'level': current_level,
                'chapter': chapter_num
            })

    return sections

def extract_index_with_range_expansion(text_content):
    """
    Extracts keywords and page numbers from the raw text of a book index,
    expands page ranges, and returns the data as a JSON string.
    """
    
    # Pre-process the text: remove source tags and page headers/footers
    text_content = re.sub(r'\\', '', text_content)
    text_content = re.sub(r'--- PAGE \d+ ---', '', text_content)
    text_content = re.sub(r'^\d+\s+Index\s*$', '', text_content, flags=re.MULTILINE)
    text_content = re.sub(r'^Index\s+\d+\s*$', '', text_content, flags=re.MULTILINE)

    # Regex to find a keyword followed by its page numbers.
    pattern = re.compile(r'^(.*?),\s*([\d,\s\-]+?)(?=\n[A-Za-z]|\Z)', re.MULTILINE | re.DOTALL)
    
    index_data = {}
    
    for match in pattern.finditer(text_content):
        # Clean up the keyword and the page number string
        keyword = match.group(1).strip().replace('\n', ' ')
        page_numbers_str = match.group(2).strip().replace('\n', ' ')

        # Skip entries that are clearly not valid keywords
        if keyword.lower() in ["mc", "graw", "hill", "education"]:
            continue

        pages = []
        # Split the string of page numbers by comma
        for part in re.split(r',\s*', page_numbers_str):
            part = part.strip()
            if not part:
                continue
            
            # Check for a page range (e.g., "805-807")
            if '-' in part:
                try:
                    start_str, end_str = part.split('-')
                    start = int(start_str)
                    end = int(end_str)
                    # Add all numbers in the range (inclusive)
                    pages.extend(range(start, end + 1))
                except ValueError:
                    # Handle cases where a part with a hyphen isn't a valid range
                    pass 
            else:
                try:
                    # It's a single page number
                    pages.append(int(part))
                except ValueError:
                    # Handle cases where a part is not a valid number
                    pass
        
        if keyword and pages:
            # Add the parsed pages to the dictionary
            if keyword in index_data:
                index_data[keyword].extend(pages)
            else:
                index_data[keyword] = pages

    # Convert the dictionary to a nicely formatted JSON string
    return json.dumps(index_data, indent=2)

def convert_and_save_with_page_numbers(input_file_path, output_file_path):
    """
    Converts a document to Markdown, iterating page by page
    to insert a custom footer with the page number after each page,
    and saves the result to a file.
    
    Args:
        input_file_path (str): The path to the source file (e.g., "/path/to/file.pdf").
        output_file_path (str): The path to the destination .md file.
    """
    
    source = Path(input_file_path)
    if not source.exists():
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        return

    # Disable OCR and table structure extraction for faster processing
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
    format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV2DocumentBackend)
        }
    )
    
    try:
        # Convert the entire document once
        result = converter.convert(source)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return
        
    doc = result.document

    # Define a unique placeholder that won't appear in the text.
    # Using "\n" ensures it's on its own line.
    UNIQUE_PLACEHOLDER = "\n%%%__DOCLING_PAGE_BREAK__%%%\n"

    # Export the entire document at once, using our placeholder.
    # This avoids the fragile doc.filter() method.
    try:
        full_markdown = doc.export_to_markdown(page_break_placeholder=UNIQUE_PLACEHOLDER)
    except Exception as e:
        print(f"Error during final markdown export: {e}", file=sys.stderr)
        print("Falling back to exporting document without page numbers.")
        try:
            # Fallback: just save the raw export
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(doc.export_to_markdown())
            print(f"Successfully saved (fallback, no page numbers) to {output_file_path}")
        except IOError as e_io:
            print(f"Error writing fallback file: {e_io}", file=sys.stderr)
        return

    # Split the full markdown by our unique placeholder.
    # This gives us a list where each item is one page's content.
    markdown_pages = full_markdown.split(UNIQUE_PLACEHOLDER)
    
    final_output_chunks = []
    
    # Iterate through the pages, adding our custom footer.
    # We use enumerate to get a 1-based page number.
    num_pages = len(markdown_pages)
    for i, page_content in enumerate(markdown_pages, 1):
        # Add the content for the current page
        final_output_chunks.append(page_content)
        
        # Add our custom footer, but not after the very last page
        if i < num_pages:
            final_output_chunks.append(f"\n\n--- Page {i} ---\n\n")

    # Write the combined markdown string to the output file
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("".join(final_output_chunks))
        print(f"Successfully converted and saved to {output_file_path}")
    except IOError as e:
        print(f"Error writing to file {output_file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


def preprocess_extracted_section(text: str) -> str:
    """
    Cleans a raw textbook section to prepare it for chunking.

    Args:
        text: The raw text of the section.

    Returns:
        str: The cleaned text.
    """
    # Replaces all newline and image tag occurences with single spaces
    text = text.replace('\n', ' ')
    text = text.replace('<!-- image -->', ' ')

    # Removes bold formatting markers (**)
    text = text.replace('**', '')

    # Normalizes all whitespace to single spaces
    cleaned_text = ' '.join(text.split())

    return cleaned_text


def main():
    # Returns all pdf files under data/chapters/
    project_root = Path(__file__).resolve().parent.parent.parent
    chapters_dir = project_root / "data/chapters"
    pdfs = sorted(chapters_dir.glob("*.pdf"))

    # Ensure at least one PDF is found
    if len(pdfs) == 0:
        print("ERROR: No PDFs found in data/chapters/. Please copy a PDF there first.", file=sys.stderr)
        sys.exit(1)

    # Convert each PDF to Markdown
    markdown_files = []
    for pdf_path in pdfs:
        pdf_name = pdf_path.stem
        output_md = Path("data") / f"{pdf_name}--extracted_markdown.md"

        print(f"Converting '{pdf_path}' to '{output_md}'...")
        convert_and_save_with_page_numbers(str(pdf_path), str(output_md))

        markdown_files.append(output_md)

    # TODO: Add logic to select which markdown file to process
    extracted_sections = extract_sections_from_markdown(markdown_files[0])
    # print(f"Processing markdown file: {markdown_files[0]}")

    if extracted_sections:
        print(f"Successfully extracted {len(extracted_sections)} sections.")
        output_filename = project_root / "data/extracted_sections.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_sections, f, indent=4, ensure_ascii=False)
        print(f"\nFull extracted content saved to '{output_filename}'")


if __name__ == '__main__':
    main()
