import gdown
import os

def download_data(url='https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'):
    output = 'data.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall('data.zip')
    os.remove('data.zip')
    print('Data downloaded successfully!')
   
    
if __name__ == '__main__':
    download_data()