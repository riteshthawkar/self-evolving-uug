from huggingface_hub import snapshot_download
import pathlib
import json
import os

repos = ["BLIP3o/BLIP3o-Model-8B", "BLIP3o/BLIP3o-Model"]

# Check if HF_HOME is set in python env
# print(f"HF_HOME: {os.environ.get('HF_HOME')}")

for repo_id in repos:
    print(f"--- Checking {repo_id} ---")
    try:
        # replicate allow_patterns from generation.py
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "diffusion-decoder/**",
                "pipeline_llava_gen.py",
                "pipeline_ar_gen.py",
            ]
        )
        print(f"Snapshot Path: {path}")
        path = pathlib.Path(path)
        
        # Check diffusion-decoder folder
        diff_dir = path / "diffusion-decoder"
        if not diff_dir.exists():
             print(f"diffusion-decoder NOT FOUND at {diff_dir}")
             continue
             
        model_index = diff_dir / "model_index.json"
        if model_index.exists():
            with open(model_index) as f:
                content = f.read()
                print(f"Content of model_index.json:\n{content}")
                
                # Check patch status
                if '["transformers", "PreTrainedModel"]' in content:
                    print(f"✅ PATCH VERIFIED: Using PreTrainedModel")
                elif '["transformers", "AutoModelForCausalLM"]' in content:
                    print(f"⚠️ PATCH OLD: Using AutoModelForCausalLM")
                elif "transformers_modules" in content:
                     print(f"❌ PATCH MISSING: Using transformers_modules")
                else:
                     print(f"❓ Unknown content")
        else:
            print("model_index.json NOT FOUND!")
            
    except Exception as e:
        print(f"Error accessing {repo_id}: {e}")
