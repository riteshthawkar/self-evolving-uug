
import sys
import os
import inspect
from huggingface_hub import snapshot_download

# Force download/locate
repo_id = "BLIP3o/BLIP3o-Model"
# Force a recursive search for the file
print(f"Checking snapshot for {repo_id}...")
try:
    path = snapshot_download(repo_id=repo_id, allow_patterns=["pipeline_llava_gen.py", "diffusion-decoder/**"])
    print(f"Snapshot root: {path}")

    # Recursive search
    target_file = "pipeline_llava_gen.py"
    found_dir = None
    
    for root, dirs, files in os.walk(path):
        if target_file in files:
            found_dir = root
            print(f"Found {target_file} in: {found_dir}")
            break
            
    if found_dir:
        sys.path.append(found_dir)
        print(f"Added {found_dir} to sys.path")
    else:
        print(f"ERROR: Could not find {target_file} in {path}")
        print("Directory listing:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

    print("Importing pipeline_llava_gen...")
    try:
        from pipeline_llava_gen import EmuVisualGenerationPipeline
        print("Successfully imported EmuVisualGenerationPipeline.")
        
        sig = inspect.signature(EmuVisualGenerationPipeline.__call__)
        print(f"\nSignature of EmuVisualGenerationPipeline.__call__:\n{sig}")
        
    except ImportError as e:
        print(f"Failed to import: {e}")
        # Fallback dump
        if found_dir:
            full_path = os.path.join(found_dir, target_file)
            print(f"\nReading {full_path} directly...")
            with open(full_path, "r") as f:
                content = f.read()
                if "def __call__" in content:
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if "def __call__" in line:
                            print(f"Line {i+1}: {line.strip()}")
                            for j in range(1, 15):
                                if i+j < len(lines):
                                    print(f"Line {i+1+j}: {lines[i+j].rstrip()}")
                            break

except Exception as e:
    print(f"An error occurred: {e}")
