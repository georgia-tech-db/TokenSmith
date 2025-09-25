#!/usr/bin/env python3
"""
Enhanced chunk visualization tool for TokenSmith.
Displays full chunks with detailed metadata, statistics, and multiple output formats.
"""

import pickle
import sys
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

def analyze_chunks(chunks: List[str], sources: List[str], metadata: Optional[List[Dict]] = None) -> Dict:
    """Analyze chunks and return statistics."""
    stats = {
        "total_chunks": len(chunks),
        "total_characters": sum(len(c) for c in chunks),
        "total_words": sum(len(c.split()) for c in chunks),
        "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
        "avg_words_per_chunk": sum(len(c.split()) for c in chunks) / len(chunks) if chunks else 0,
        "sources": list(set(sources)),
        "chunk_lengths": [len(c) for c in chunks],
        "word_counts": [len(c.split()) for c in chunks],
    }
    
    if metadata:
        stats["chunking_mode"] = metadata[0].get("mode", "unknown") if metadata else "unknown"
        stats["has_tables"] = sum(1 for m in metadata if m.get("has_table", False))
        stats["table_percentage"] = (stats["has_tables"] / len(metadata)) * 100 if metadata else 0
    
    return stats

def clean_text_for_display(text: str) -> str:
    """Clean text for better display (remove excessive whitespace, etc.)."""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)

def convert_to_chonkie_chunks(chunks: List[str]) -> Tuple[str, List]:
    """Convert plain text chunks to chonkie-compatible format and build full_text.

    We concatenate all chunks with double-newline separators to create a
    deterministic full_text and compute cumulative start/end offsets for
    each chunk within that full_text so chonkie renders the entire document.
    """
    separator = "\n\n"
    full_text_parts: List[str] = []
    start_offsets: List[int] = []
    cursor = 0
    for i, c in enumerate(chunks):
        start_offsets.append(cursor)
        full_text_parts.append(c)
        cursor += len(c)
        if i != len(chunks) - 1:
            full_text_parts.append(separator)
            cursor += len(separator)
    full_text = "".join(full_text_parts)

    try:
        from chonkie.types.base import Chunk
        chonkie_chunks = []
        for i, chunk in enumerate(chunks):
            start_index = start_offsets[i]
            end_index = start_index + len(chunk)
            chonkie_chunk = Chunk(
                text=chunk,
                start_index=start_index,
                end_index=end_index,
                token_count=len(chunk.split())
            )
            chonkie_chunks.append(chonkie_chunk)
        return full_text, chonkie_chunks
    except ImportError:
        # Fallback to simple object if chonkie types not available
        chonkie_chunks = []
        for i, chunk in enumerate(chunks):
            start_index = start_offsets[i]
            end_index = start_index + len(chunk)
            chonkie_chunk = type('Chunk', (), {
                'text': chunk,
                'start_index': start_index,
                'end_index': end_index,
                'token_count': len(chunk.split())
            })()
            chonkie_chunks.append(chonkie_chunk)
        return full_text, chonkie_chunks

def generate_chonkie_html(chunks: List[str], output_file: str) -> str:
    """Generate HTML visualization using chonkie for all chunks.
    
    Args:
        chunks: List of text chunks to visualize
        output_file: Path where to save the HTML file
        
    Returns:
        Path to the generated HTML file
    """
    try:
        from chonkie import Visualizer
        viz = Visualizer()
        full_text, chonkie_chunks = convert_to_chonkie_chunks(chunks)
        viz.save(output_file, chonkie_chunks, full_text=full_text, title="Chunk Visualization")
        return output_file
    except ImportError:
        raise ImportError("chonkie[viz] is required for HTML generation. Install with: pip install 'chonkie[viz]'")
    except Exception as e:
        raise RuntimeError(f"Failed to generate chonkie HTML: {e}")

def generate_html_visualization(chunks: List[str], sources: List[str], metadata: Optional[List[Dict]] = None, 
                               output_file: str = "chunk_visualization.html"):
    """Generate an HTML file with full chunk visualization."""
    
    stats = analyze_chunks(chunks, sources, metadata)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TokenSmith Chunk Visualization</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .chunk {{
            border: 1px solid #dee2e6;
            margin: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chunk-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        .chunk-id {{
            font-size: 1.2em;
            font-weight: bold;
            color: #495057;
        }}
        .chunk-meta {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            font-size: 0.9em;
            color: #6c757d;
        }}
        .chunk-content {{
            padding: 20px;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            line-height: 1.5;
            max-height: 500px;
            overflow-y: auto;
            background: #fafafa;
        }}
        .search-box {{
            margin: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .search-box input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 16px;
        }}
        .controls {{
            margin: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 8px 16px;
            border: 1px solid #ced4da;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .btn:hover {{
            background: #e9ecef;
        }}
        .btn.active {{
            background: #007bff;
            color: white;
            border-color: #007bff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 TokenSmith Chunk Visualization</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{stats['total_chunks']}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['total_characters']:,}</div>
                <div class="stat-label">Total Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['total_words']:,}</div>
                <div class="stat-label">Total Words</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['avg_chunk_length']:.0f}</div>
                <div class="stat-label">Avg Chunk Length</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['avg_words_per_chunk']:.0f}</div>
                <div class="stat-label">Avg Words/Chunk</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(stats['sources'])}</div>
                <div class="stat-label">Source Files</div>
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search chunks by content..." onkeyup="filterChunks()">
        </div>
        
        <div class="controls">
            <button class="btn active" onclick="toggleView('all')">Show All</button>
            <button class="btn" onclick="toggleView('short')">Short Chunks</button>
            <button class="btn" onclick="toggleView('long')">Long Chunks</button>
            <button class="btn" onclick="toggleView('tables')">With Tables</button>
        </div>
        
        <div id="chunksContainer">
"""

    # Add each chunk
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        meta = metadata[i] if metadata and i < len(metadata) else {}
        cleaned_chunk = clean_text_for_display(chunk)
        
        # Determine chunk characteristics
        char_len = len(chunk)
        word_len = len(chunk.split())
        has_table = meta.get('has_table', False)
        
        # Create chunk classes for filtering
        chunk_classes = ['chunk-item']
        if char_len < stats['avg_chunk_length'] * 0.8:
            chunk_classes.append('short-chunk')
        if char_len > stats['avg_chunk_length'] * 1.2:
            chunk_classes.append('long-chunk')
        if has_table:
            chunk_classes.append('table-chunk')
        
        html_content += f"""
            <div class="chunk {' '.join(chunk_classes)}" data-content="{chunk.lower()}">
                <div class="chunk-header">
                    <div class="chunk-id">Chunk {i+1}/{len(chunks)}</div>
                    <div class="chunk-meta">
                        <span>📄 {source}</span>
                        <span>📏 {char_len:,} chars</span>
                        <span>📝 {word_len:,} words</span>
                        {f'<span>📊 Table: {"Yes" if has_table else "No"}</span>' if 'has_table' in meta else ''}
                        {f'<span>🔧 Mode: {meta.get("mode", "unknown")}</span>' if 'mode' in meta else ''}
                    </div>
                </div>
                <div class="chunk-content">{cleaned_chunk}</div>
            </div>
        """

    html_content += """
        </div>
    </div>
    
    <script>
        function filterChunks() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const chunks = document.querySelectorAll('.chunk-item');
            
            chunks.forEach(chunk => {
                const content = chunk.getAttribute('data-content');
                if (content.includes(searchTerm)) {
                    chunk.style.display = 'block';
                } else {
                    chunk.style.display = 'none';
                }
            });
        }
        
        function toggleView(type) {
            const chunks = document.querySelectorAll('.chunk-item');
            const buttons = document.querySelectorAll('.btn');
            
            // Update button states
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            chunks.forEach(chunk => {
                let show = false;
                switch(type) {
                    case 'all':
                        show = true;
                        break;
                    case 'short':
                        show = chunk.classList.contains('short-chunk');
                        break;
                    case 'long':
                        show = chunk.classList.contains('long-chunk');
                        break;
                    case 'tables':
                        show = chunk.classList.contains('table-chunk');
                        break;
                }
                chunk.style.display = show ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file


