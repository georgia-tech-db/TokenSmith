#!/usr/bin/env python3
"""
Quick script to check if metadata exists in the index.
"""

import pathlib
import pickle
import sys

def check_metadata():
    """Check if metadata file exists and has content."""
    artifacts_dir = pathlib.Path("index/sections")
    index_prefix = "textbook_index"
    metadata_path = artifacts_dir / f"{index_prefix}_meta.pkl"
    
    if not metadata_path.exists():
        print("❌ Metadata file NOT found!")
        print(f"   Path: {metadata_path}")
        print("\n   To enable citations, rebuild the index:")
        print("   make run-index")
        return False
    
    try:
        metadata = pickle.load(open(metadata_path, "rb"))
        if not metadata or len(metadata) == 0:
            print("⚠️  Metadata file exists but is EMPTY!")
            print(f"   Path: {metadata_path}")
            print("\n   Rebuild the index to populate metadata:")
            print("   make run-index")
            return False
        
        # Check if metadata has actual content
        sample = metadata[0] if metadata else {}
        has_content = sample.get('page_number') or sample.get('section') or sample.get('chapter')
        
        if has_content:
            print(f"✅ Metadata file found with {len(metadata)} entries")
            print(f"   Path: {metadata_path}")
            print(f"   Sample entry keys: {list(sample.keys())}")
            if sample.get('page_number'):
                print(f"   Sample page number: {sample.get('page_number')}")
            return True
        else:
            print("⚠️  Metadata file exists but entries are EMPTY!")
            print(f"   Path: {metadata_path}")
            print("\n   Rebuild the index to populate metadata:")
            print("   make run-index")
            return False
            
    except Exception as e:
        print(f"❌ Error reading metadata file: {e}")
        return False

if __name__ == "__main__":
    success = check_metadata()
    sys.exit(0 if success else 1)



