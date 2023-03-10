import gdown
import os
import subprocess
import zipfile


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
        with zipfile.ZipFile("data.zip", "r") as file:
            file.extractall("data")
        os.remove("data.zip")
    else:
        print("Data already exists")


def upload_file():
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    gfile = drive.CreateFile(
        {"parents": [{"id": "1UmD1_yE5nz5jcZPGDyK2iI-I-7a-P7Mj"}], "title": "hello123.txt"}
    )
    gfile.SetContentString("Hello Hello!")
    gfile.Upload()  # Upload the file.


####
# drive.ListFile('1IGN2yqPeUmJpzifrOE8Y4VMjn4Qmrr3x')
# file_list = drive.ListFile({'q': "1IGN2yqPeUmJpzifrOE8Y4VMjn4Qmrr3x in parents and trashed=false"}).GetList()
# file_list = drive.ListFile({'q': "'1w-TyCSLDGQbantlvK93imJaNABazXwy1' in parents and trashed=false"}).GetList()
# file_list[0]

# drive.ListFile({'q': "'1w-TyCSLDGQbantlvK93imJaNABazXwy1'"}).GetList()

# gauth.LocalWebserverAuth()

if __name__ == "__main__":
    get_data()
# upload_file()
