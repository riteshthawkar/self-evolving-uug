**SigLIP2 + Sana Reconstruction**

This setup combines the SigLIP2 encoder (`siglip2-so400m-patch16-512`) with the Sana decoder (`Efficient-Large-Model/Sana_1600M_512px_diffusers`).

1. **Download**
  Sana decoder weights from: [BLIP3o/SigLIP2_SANA](https://huggingface.co/BLIP3o/SigLIP2_SANA)
3. **Run Inference**
   
   With your the same environment:

   ```bash
   python inference.py /path/to/your/model
   ```

   The script will output the reconstructed images.
