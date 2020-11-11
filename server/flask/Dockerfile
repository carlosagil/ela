# Pull keras image with Python3
FROM gw000/keras:2.1.4-py3-tf-cpu

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install
RUN apt-get update \
    && apt-get install -y \
    python-pillow

# Install python packages
RUN pip install -r requirements.txt

# Download model
CMD [ "python", "./download_model.py" ]

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]