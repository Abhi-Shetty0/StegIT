import kagglehub
import os

# Get the absolute path of the directory where this script is located (the repo root)
repo_dir = os.path.dirname(os.path.abspath(__file__))

# Define the destination path inside the repo (using the 'data' folder which is ignored by git)
destination = os.path.join(repo_dir, "data", "chest-xray-masks-and-labels")

print(f"Starting download to: {destination}")

# Download dataset directly to the destination (bypasses cache and shutil.copytree)
# kagglehub will automatically show a progress bar if tqdm is installed
path = kagglehub.dataset_download(
    "nikhilpandey360/chest-xray-masks-and-labels",
    output_dir=destination,
    force_download=True
)

print("\nDataset successfully downloaded to:")
print(path)
