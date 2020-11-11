import numpy as np
from keras.models import load_model
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from PIL import Image, ImageChops, ImageEnhance
import os

# https://gist.github.com/cirocosta/33c758ad77e6e6531392
# error level analysis of an image

ELA_EXT = ".ela.png"


def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""

    basename, ext = os.path.splitext(img_path)

    ela_fname = "test" + ELA_EXT

    TEMP = 'ela_test_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)

    except:

        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()

    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            #print(d[x, y])
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(diff).enhance(scale)

    ela_im.save(ela_fname)

    #print(f'************************* ela diff {diff}')
    return diff


def test():
    """
    docstring
    """

    # original repo

    #path = "casia2/Au/Au_ani_00001.jpg"

    # fake repo
    #path = "casia2/Tp/Tp_D_CRN_S_N_cha10122_nat10123_12168.jpg"

    path = "images/cc_number.jpg"
    #path = "test.jpg"

    X_f = np.array(ELA(path).resize((128, 128))).flatten() / 255.0

    X_f = X_f.reshape(-1, 128, 128, 3)

    model = load_model('new_model_casia_03.h5')

    # ========================================

    y_pred_cnn = model.predict(X_f)

    result = y_pred_cnn[0]

    #print(f'===> model predict {result}')
    print(f'===> model predict max {result[0]}')
    print(f'===> model predict min {result[1]}')


if __name__ == '__main__':
    test()
