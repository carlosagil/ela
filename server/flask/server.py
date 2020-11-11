import os
import base64
import random
from flask import Flask, request, jsonify
from predict import Tampering_Detection_Service

ELA_EXT = ".ela.png"
TMP_EXT = ".temp.jpg"

# instantiate flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    """

    file_name = request.files['name']
    _, ext = os.path.splitext(file_name)

    name = str(random.randint(0, 100000))
    name_key = name + ext

    image = request.files['file']
    image = image[image.find(",")+1:]

    decode = base64.b64decode(image + "===")

    with open(name_key, "wb") as fh:
        fh.write(decode)

    # instantiate keyword spotting service singleton and get prediction
    tds = Tampering_Detection_Service()
    predicted = tds.predict(name_key, name)

    ela_key = name + ELA_EXT
    elab64 = ""
    with open(ela_key, "rb") as ela_img:
        elab64 = base64.b64encode(ela_img.read())

    # result json
    result = {"accurency": predicted, "ela": elab64}

    tmp_key = name + TMP_EXT

    # remove temps images
    os.remove(name_key)
    os.remove(ela_key)
    os.remove(tmp_key)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)
