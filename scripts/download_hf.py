from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "nycu-itron/balloon-popping-challenge-2026"

local_download_dir = Path(__file__).resolve().parent / "downloaded_runs"
local_download_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_download_dir,
)
