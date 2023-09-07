from pathlib import Path
import setuptools 

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setuptools.setup(
    name="starter",
    version="0.0.0",
    description="Starter code.",
    author="Student",
)
