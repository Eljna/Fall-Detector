from PIL import Image
import glob

folder = 'C:\\Users\\Elina_UNI\\Pictures\\justonecat'
new_folder = 'C:\\Users\\Elina_UNI\\Pictures\\cattos'

image_list = []
resized_images = []

for filename in glob.glob(f'{folder}*.jpg'):
    print(filename)
    img = Image.open(filename)
    image_list.append(img)

for image in image_list:
    image = image.resize((400, 266))
    resized_images.append(image)

for (i, new) in enumerate(resized_images):
    new.save(f'{new_folder}val.fine.{i+1}.jpg')
