from PIL import Image


def print_image_data1(image):
    print('Image data:')
    print('format:', image.format)
    print('mode:', image.mode)
    print('width:', image.width)
    print('height:', image.height)


def print_image_data(image):
    print('IMAGE DATA:')
    print('format:', image.format, 'mode:', image.mode)
    print('width:', image.width, 'height:', image.height)


def handle_image(image):
    print('image handling running...')

    print_image_data(image)


    # stub = Image.open('../neural-networks-sandbox/images/doctorwho.jpg')

    stub = image.resize((image.width * 2, image.height * 2))

    return stub
