import os

OPENFILE_INITIALDIR_WINDOWS = r'C:\Users\User\PycharmProjects\neural-networks-sandbox\images'
SAVEFILE_INITIALDIR_WINDOWS = r'C:\Users\User\PycharmProjects\neural-networks-sandbox\images'
BACKGROUND_IMAGE_PATH_WINDOWS = '../neural-networks-sandbox/resourses/gradient.png'

OPENFILE_INITIALDIR_LINUX = r'.'
SAVEFILE_INITIALDIR_LINUX = r'.'
BACKGROUND_IMAGE_PATH_LINUX = '../neural-networks-sandbox/resourses/gradient.png'

print('Your OS is', os.name)

if os.name == 'nt':
    OPENFILE_INITIALDIR = OPENFILE_INITIALDIR_WINDOWS
    SAVEFILE_INITIALDIR = SAVEFILE_INITIALDIR_WINDOWS
    BACKGROUND_IMAGE_PATH = BACKGROUND_IMAGE_PATH_WINDOWS
elif os.name == 'posix':
    OPENFILE_INITIALDIR = OPENFILE_INITIALDIR_LINUX
    SAVEFILE_INITIALDIR = SAVEFILE_INITIALDIR_LINUX
    BACKGROUND_IMAGE_PATH = BACKGROUND_IMAGE_PATH_LINUX
else:
    print('UNKNOWN OS')
    exit()