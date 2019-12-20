import io
import time
from konduit.load import client_from_file
from konduit.client import Client


def test_model():
    request_count = 10

    client = Client(input_data_format="IMAGE", output_data_format="NUMPY", input_names=["default"], output_names=["output"], host="http://localhost", port=1337, prediction_type="NUMPY")
    #client = client_from_file("konduit.yml")
    with open("1902_airplane.png", "rb") as binary_file:
        data = binary_file.read()

    responses = []

    start = time.time()
    for _ in range(request_count):
            responses.append(client.predict({"default": data}))
    end = time.time()

    print(responses[0])
    print("%f seconds elapsed for %d requests (%f RPS)" % (end - start, request_count, (request_count / (end - start))))


if __name__ == '__main__':
    test_model()
