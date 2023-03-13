import gdown
import os
import subprocess
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def get_data():
    if os.path.isdir("data") == False:
        print("Downloading data")
        process = subprocess.Popen(
            "chmod +x ./src/data/get_data.sh",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        stdout, stderr = process.communicate()
        subprocess.call(["./src/data/get_data.sh"])
        with zipfile.ZipFile("./data.zip", "r") as file:
            file.extractall("./data")
        os.remove("./data.zip")
        process = subprocess.Popen(
            "mv ./data/data/* data/",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        stdout, stderr = process.communicate()
        process = subprocess.Popen(
            "rm -rf ./data/data",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        stdout, stderr = process.communicate()
    else:
        print("Data already exists")


def upload_file(file_path):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    gfile = drive.CreateFile(
        {
            "parents": [{"id": "1sdBtDxKMqO0esCKaezY2TJqLhzt9ozeS"}],
            "title": f"{file_path.split('/')[1]}.json",
        }
    )
    gfile.SetContentFile(f"results/{file_path}.json")
    gfile.Upload()  # Upload the file.


# if __name__ == "__main__":
#     #get_data()
#     upload_file('compare_results/test')
