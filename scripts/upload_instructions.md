# Upload Model to Hugging Face Hub

## Quick Start (3 Steps)

### 1. Install Hugging Face Hub
```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face
```bash
huggingface-cli login
```
Enter your Hugging Face token when prompted. Get your token from: https://huggingface.co/settings/tokens

### 3. Run Upload Script
```bash
python upload_to_huggingface.py
```

---

## Alternative: Manual Upload via Web Interface

1. Go to https://huggingface.co/new
2. Create a new model repository (e.g., `handwriting-recognition-iam`)
3. Click "Files" → "Add file" → "Upload files"
4. Upload:
   - `best_model.pth`
   - `README.md`
   - `requirements.txt`
   - `train_colab.ipynb`
   - `training_history.png`

---

## Alternative: Upload from Python (Colab/Script)

```python
from huggingface_hub import HfApi, create_repo, upload_file

# Login first (in Colab)
from huggingface_hub import notebook_login
notebook_login()

# Create repository
api = HfApi()
repo_id = "your-username/handwriting-recognition-iam"
create_repo(repo_id, repo_type="model", exist_ok=True)

# Upload model
upload_file(
    path_or_fileobj="best_model.pth",
    path_in_repo="best_model.pth",
    repo_id=repo_id,
    repo_type="model"
)

print(f"✓ Uploaded! View at: https://huggingface.co/{repo_id}")
```

---

## What Gets Uploaded

- ✅ `best_model.pth` - Trained model checkpoint (105MB)
- ✅ `README.md` - Project documentation
- ✅ `requirements.txt` - Dependencies
- ✅ `train_colab.ipynb` - Training notebook
- ✅ `training_history.png` - Training metrics visualization

---

## Customization

Edit `upload_to_huggingface.py` to change:
- `REPO_NAME` - Your preferred repository name
- `private=False` - Set to `True` for private repository
- `FILES_TO_UPLOAD` - Add/remove files to upload

---

## Troubleshooting

### "Authentication required"
```bash
huggingface-cli login
```

### "Repository already exists"
- The script uses `exist_ok=True`, so it will update existing repo
- Or change `REPO_NAME` to create a new one

### Large file upload fails
- Hugging Face supports files up to 50GB
- Your model (105MB) should upload fine
- If it fails, try uploading via web interface
