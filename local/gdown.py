import gdown

url = 'https://drive.google.com/drive/folders/1n7u72D_cKkXUSIdVTNJ0_PvzZkIP4HPG'
casia = "casia2.zip"
gdown.download(url, casia, quiet=False)


url = 'https://drive.google.com/file/d/11jkIkekxcQscVeEiNQ9sqz7SrG-es1P1'
model = "model.h5"
gdown.download(url, model, quiet=False)
