from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

text = "Python Imaging Library in Habr :)"
color = (0, 0, 120)
img = Image.new('RGB', (200, 100), color)
imgDrawer = ImageDraw.Draw(img)
imgDrawer.text((10, 20), text)
img.save("pil-example.png")

img = Image.open('doctorwho.jpg')  # открываем картинку
size = img.size  # размер картинки
format = img.format  # формат картинки
mode = img.mode  # мод(RGBA...)
arr = [size, format, mode]  # создаем пустой массив
print(arr)  # выводим массив
print(img.info)
# img.show()
box = (100, 100, 400, 400)
region = img.crop(box)
# region.show()

region = region.transpose(Image.ROTATE_180)
img.paste(region, box)


# img.show()


def roll(image, delta):
    "Roll an image sideways"
    xsize, ysize = image.size
    delta = delta % xsize
    if delta == 0:
        return image
    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize - delta, ysize))
    image.paste(part1, (xsize - delta, 0, xsize, ysize))
    return image


# img = roll(img, 300)
# img.show()

r, g, b = img.split()
print(r, g, b)
# img = Image.merge("RGB", (r, g, b))
# img.show()

# img.show()
# print(img.size)
# img = img.resize((size[0] * 2, size[1] * 2))
# img.show()
# print(img.size)
# img = img.resize((size[0] // 2, size[1] // 2))
# img.show()

# img.rotate(45).show()
# img.transpose(Image.FLIP_LEFT_RIGHT).show()
# img.transpose(Image.FLIP_TOP_BOTTOM).show()
# img.transpose(Image.ROTATE_90).show()
# img.convert('L').show()
# img.convert('RGB').show()
# img.show()
# img.filter(ImageFilter.DETAIL).show()
points = []

def f(i):
    points.append(i)
    return i

# img.point(f).show()
# print(points[:10])
# source = img.split()
# R, G, B = 0, 1, 2
# mask = source[G].point(lambda i: i < 100 and 255).show()

enh = ImageEnhance.Contrast(img)
shrp = ImageEnhance.Sharpness(img)
clr = ImageEnhance.Color(img)
# img.show()
# enh.enhance(1.3).show("30% more contrast")
# shrp.enhance(1.5).show()
# clr.enhance(1.5).show()

# import StringIO
# im = Image.open(StringIO.StringIO(buffer))

# import TarIO
# fp = TarIO.TarIO("Imaging.tar", "Imaging/test/lena.ppm")
# im = Image.open(fp)

image_data = img.getdata()
print(list(image_data)[:10])
pixel = img.getpixel((1, 0))
print(pixel)
#print(size[0] * size[1])
img.putpixel((1, 0), (0,0,0))
print(img.getpixel((1, 0)))
image_data = img.getdata()
print(list(image_data)[:10])

# img.show()
# bm = img.tobitmap()

#img = Image.fromstring('RGB', (100, 100), )

img = img.convert('L')
image_data = img.getdata()
print(list(image_data)[:10])

# r, g, b = im.split()
# im = Image.merge("RGB", (b, g, r))



