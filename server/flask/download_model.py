import gdown

url = 'https://drive.google.com/file/d/11jkIkekxcQscVeEiNQ9sqz7SrG-es1P1'
output = "model.h5"
gdown.download(url, output, quiet=False)
