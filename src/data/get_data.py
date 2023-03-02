#import subprocess
import os
from google.cloud import storage
from tqdm import tqdm
import torchvision
torchvision.__version__
import torch

# def get_data():
#     if os.path.isdir('data/'):
#         print("Data already downloaded")
#     else:
#         bash_cmd = "gsutil -m cp -R gs://data_bachelor_buck/data ./"
#         process = subprocess.Popen(bash_cmd.split(),stdout=subprocess.PIPE)
#         output,error=process.communicate()
#         if error:
#             print("Downloading of data failed")
#             print(f"Error:{error}")
#         else:        
#             print("Downloading of data succesful")
# def get_data(dataset):
#     if os.path.isdir('data/'):
#         print("Data already downloaded")
#     else:
#         os.mkdir('data')
#         os.mkdir('data/color_mapping')
#         os.mkdir('data/processed')
#         os.mkdir('data/raw')
#         for data in ['membrane','warwick','DIC_C2DH_Hela','PhC-C2DH-U373']:
#             os.mkdir(f"data/raw/{data}")
#             os.mkdir(f"data/raw/{data}/image")
#             os.mkdir(f"data/raw/{data}/label")

#         client = storage.Client('bachelor-dev-377908')
#         bucket = client.get_bucket('data_bachelor_buck')
#         blobs=bucket.list_blobs()
#         print("Downloading data")
#         for blob in tqdm(blobs):
#             filename = blob.name
#             if (dataset and dataset in filename) or ('Color_mapping' in filename):
#                 blob.download_to_filename(filename)
#         print("Download of data succesful")

# get_data('membrane')



# import pandas as pd
# hej=pd.DataFrame({'hej':[1,2,3,4,4,5,6,7,7,7,]})
# hej.to_json('test.json')


