import json
import os
import pathlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from src.config import RAGConfig

class IndexManager:
    """
    Centralized manager for index discovery, validation, and lifecycle.
    Indexes are stored in index/<name>/ and contain a manifest.json.
    """
    def __init__(self, base_dir: str = "index"):
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_index_dir(self, name: str) -> pathlib.Path:
        return self.base_dir / name

    def list_indexes(self) -> List[Dict[str, Any]]:
        """Discovers and returns metadata for all valid indexes."""
        indexes = []
        if not self.base_dir.exists():
            return []
        
        for item in self.base_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            manifest['name'] = item.name
                            indexes.append(manifest)
                    except Exception:
                        continue
        return sorted(indexes, key=lambda x: x.get('last_updated', ''), reverse=True)

    def write_manifest(self, name: str, config: RAGConfig, chapters: List[int], source_file: str):
        """Writes or updates the manifest for a specific index."""
        target_dir = self.get_index_dir(name)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = target_dir / "manifest.json"
        
        # Merge if existing
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                "created_at": datetime.now().isoformat(),
                "chapters": []
            }

        # Update fields
        manifest.update({
            "last_updated": datetime.now().isoformat(),
            "embedding_model": config.embed_model,
            "chunk_strategy": config.chunk_mode,
            "source_file": source_file,
        })
        
        # Merge chapters and deduplicate
        existing_chapters = set(manifest.get("chapters", []))
        new_chapters = set(chapters or [])
        manifest["chapters"] = sorted(list(existing_chapters.union(new_chapters)))

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)

    def validate_index(self, name: str, current_config: RAGConfig) -> bool:
        """Validates that an index is compatible with the current configuration."""
        target_dir = self.get_index_dir(name)
        manifest_path = target_dir / "manifest.json"
        
        if not manifest_path.exists():
            print(f"ERROR: Index '{name}' has no manifest.json.")
            return False
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        if manifest.get("embedding_model") != current_config.embed_model:
            print(f"ERROR: Index '{name}' was built with model '{manifest.get('embedding_model')}', "
                  f"but current config uses '{current_config.embed_model}'.")
            return False
            
        return True
