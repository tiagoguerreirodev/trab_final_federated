import torch
import glob

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

image_path = './test_images/'
imgs = glob.glob(image_path + "*.png")

result = model(image_path,size=640)

img_df = result.pandas().xyxy

linhas_txt = []
for idx,df in enumerate(img_df):
  if 'display' in df.values:
    inf_esq = df[df['name'].str.match(r'sinal_apenas|sinal_e_um|sinal_zero|sinal_mais')==True]
    if inf_esq['name'].values[0] == 'sinal_apenas':
      inf_esq='-'
    elif inf_esq['name'].values[0] == 'sinal_e_um':
      inf_esq='-1'
    elif inf_esq['name'].values[0] == 'sinal_zero':
      inf_esq='+'
    elif inf_esq['name'].values[0] == 'sinal_mais':
      inf_esq='+1'
    sup_esq=df[df['name'].str.match(r'sup_esq.*')==True].values[0][-1].split('_')[-1]
    sup_dir=df[df['name'].str.match(r'sup_dir.*')==True].values[0][-1].split('_')[-1]
    inf_dir=df[df['name'].str.match(r'inf_dir.*')==True].values[0][-1].split('_')[-1]

    linhas_txt.append(f"image_{idx+1} {imgs[idx].split('/')[-1]}\n{sup_esq}{sup_dir}\n{inf_esq}{inf_dir}\n")
  else:
    linhas_txt.append(f"image_{idx+1} {imgs[idx].split('/')[-1]}\n-9999\n-9999\n")
  
with open('output.txt', 'w+') as file:
  file.writelines(linhas_txt)

