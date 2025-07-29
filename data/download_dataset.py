"""
Script to download Fabric Defect Dataset from Kaggle.
REQUIRED: Replace kaggle.json path with your own Kaggle API key file configured locally.
"""
import os

def download_dataset():
    os.system("kaggle datasets download -d rmshashi/fabric-defect-dataset -p data/ --unzip")
    print("Dataset downloaded and extracted in 'data/' folder.")

if __name__ == "__main__":
    download_dataset()
