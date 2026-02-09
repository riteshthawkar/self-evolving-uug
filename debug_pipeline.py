
import sys
import os
import inspect
from huggingface_hub import snapshot_download

# Force download/locate
repo_id = "BLIP3o/BLIP3o-Model"
print(f"Downloading/Locating snapshot for {repo_id}...")
try:
    path = snapshot_download(repo_id=repo_id, allow_patterns=["pipeline_llava_gen.py", "diffusion-decoder/**"])
    print(f"Snapshot path: {path}")
    sys.path.append(path)

    print("Importing pipeline_llava_gen...")
    try:
        from pipeline_llava_gen import EmuVisualGenerationPipeline
        print("Successfully imported EmuVisualGenerationPipeline.")
        
        sig = inspect.signature(EmuVisualGenerationPipeline.__call__)
        print(f"\nSignature of EmuVisualGenerationPipeline.__call__:\n{sig}")
        
        print(f"\nDocstring of EmuVisualGenerationPipeline.__call__:\n{EmuVisualGenerationPipeline.__call__.__doc__}")
        
    except ImportError as e:
        print(f"Failed to import EmuVisualGenerationPipeline: {e}")
        # Try to read file content directly to find def __call__
        pipeline_file = os.path.join(path, "pipeline_llava_gen.py")
        if os.path.exists(pipeline_file):
            print(f"\nReading {pipeline_file} directly...")
            with open(pipeline_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "def __call__" in line:
                        print(f"Line {i+1}: {line.strip()}")
                        # print next few lines
                        for j in range(1, 10):
                            if i+j < len(lines):
                                print(f"Line {i+1+j}: {lines[i+j].rstrip()}")
        
except Exception as e:
    print(f"An error occurred: {e}")
