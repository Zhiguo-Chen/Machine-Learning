import PIL.Image as Image
from img_path import img_path


im = Image.open(img_path())
im.show()
