"""
Upload handwriting recognition model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file

# Configuration
MODEL_PATH = "best_model.pth"
REPO_NAME = "handwriting-recognition-iam"  # Change this to your preferred name
USERNAME = "IsmatS"  # Will use your HF username automatically

# Files to upload
FILES_TO_UPLOAD = [
    "best_model.pth",
    "README.md",
    "requirements.txt",
    "train_colab.ipynb",
    "training_history.png",
    "charts/01_sample_images.png",
    "charts/02_text_length_distribution.png",
    "charts/03_image_dimensions.png",
    "charts/04_character_frequency.png",
    "charts/05_summary_statistics.png"
]

def upload_model_to_hf():
    """Upload model and related files to Hugging Face Hub"""

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found!")
        print("   Please ensure the model file exists in the current directory.")
        return

    print("🚀 Starting upload to Hugging Face Hub...")
    print(f"   Repository: {REPO_NAME}")
    print()

    try:
        # Initialize API
        api = HfApi()

        # Get username from token
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{REPO_NAME}"

        print(f"✓ Authenticated as: {username}")
        print(f"✓ Repository ID: {repo_id}")
        print()

        # Create repository (if it doesn't exist)
        print("📦 Creating repository...")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False  # Set to True if you want a private repo
            )
            print(f"✓ Repository created/verified: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"⚠️  Repository may already exist: {e}")

        print()

        # Upload files
        print("📤 Uploading files...")
        for file_path in FILES_TO_UPLOAD:
            if os.path.exists(file_path):
                print(f"   Uploading {file_path}...", end=" ")
                try:
                    upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=file_path,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    print("✓")
                except Exception as e:
                    print(f"❌ Failed: {e}")
            else:
                print(f"   ⚠️  Skipping {file_path} (not found)")

        print()
        print("=" * 60)
        print("🎉 Upload complete!")
        print(f"🔗 View your model: https://huggingface.co/{repo_id}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error during upload: {e}")
        print()
        print("Make sure you're logged in to Hugging Face:")
        print("  Run: huggingface-cli login")
        print("  Or set HF_TOKEN environment variable")

if __name__ == "__main__":
    upload_model_to_hf()
