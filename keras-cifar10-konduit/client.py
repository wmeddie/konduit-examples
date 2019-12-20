import io
import logging
import time
from konduit.load import client_from_file

logging.basicConfig(level='DEBUG')
logging.info("Test")

def test_model():
    client = client_from_file("konduit.yml")

    image = open("1902_airplane.png", "rb").read()

    responses = []

    start = time.time()
    for i in range(10):
        response = client.predict({"default": image})
        print(response)
        responses.append(response)

    end = time.time()

    print("%f seconds elapsed for %d requests (%d RPS)" % (end - start, len(responses), (10.0 / (end - start))))


if __name__ == '__main__':
    test_model()
