import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

ELA_EXT = ".ela.png"
TMP_EXT = ".temp.jpg"
SAVED_MODEL_PATH = "model.h5"


class _Tampering_Detection_Service:
    """
    """

    model = None
    _instance = None

    def predict(self, file_path, image_name):
        """
        """

        diff = np.array(self.ELA(file_path, image_name).resize(
            (128, 128))).flatten() / 255.0

        diff = diff.reshape(-1, 128, 128, 3)

        predicted = self.model.predict(diff)

        return predicted[0]

    def ELA(self, file_path, image_name):
        """Performs Error Level Analysis over a image"""

        ela_key = image_name + ELA_EXT

        TEMP = image_name + TMP_EXT
        SCALE = 10
        original = Image.open(file_path)
        try:
            original.save(TEMP, quality=90)
            temporary = Image.open(TEMP)
            diff = ImageChops.difference(original, temporary)

        except:

            original.convert('RGB').save(TEMP, quality=90)
            temporary = Image.open(TEMP)
            diff = ImageChops.difference(original.convert('RGB'), temporary)

        extrema = diff.getextrema()

        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1

        scale = 255.0/max_diff
        ela_im = ImageEnhance.Brightness(diff).enhance(scale)

        ela_im.save(ela_key)

        d = diff.load()
        WIDTH, HEIGHT = diff.size

        for x in range(WIDTH):
            for y in range(HEIGHT):
                # print(d[x, y])
                d[x, y] = tuple(k * SCALE for k in d[x, y])

        return diff


def Tampering_Detection_Service():
    """
    """

    if _Tampering_Detection_Service._instance is None:
        _Tampering_Detection_Service._instance = _Tampering_Detection_Service()
        _Tampering_Detection_Service.model = load_model(SAVED_MODEL_PATH)

    return _Tampering_Detection_Service._instance


if __name__ == "__main__":

    tds = Tampering_Detection_Service()

    # make a prediction
    accurancy = tds.predict("test/cc.jpg", "cc")
    print(f"accurancy: {accurancy}")
