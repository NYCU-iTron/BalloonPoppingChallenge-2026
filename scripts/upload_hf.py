from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()

repo_id = "nycu-itron/balloon-popping-challenge-2026"
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)

folder_path = Path(__file__).resolve().parent / "runs"
if not folder_path.exists():
    print(f"The folder {folder_path} does not exist.")

api.upload_folder(
    folder_path=str(folder_path),
    path_in_repo="runs",
    repo_id=repo_id,
    repo_type="model"
)

