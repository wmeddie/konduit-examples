import unittest
import serve
import torch
import torchvision
import torchvision.transforms as transforms

class TestStringMethods(unittest.TestCase):

    def test_serving(self):
        serve.setup()

        testset = torchvision.datasets.CIFAR10(root='./model/data', train=False,
                                               download=True, transform=None)
        image, _ = testset[0]

        raw = transforms.ToTensor()(image)
        self.assertTrue(isinstance(raw, torch.Tensor))

        self.assertEqual(3, raw.shape[0])
        self.assertEqual(32, raw.shape[1])
        self.assertEqual(32, raw.shape[2])

        response = serve.run(image)
        print(response)
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 0)
        self.assertTrue(response.shape[0] == 1)
        self.assertTrue(response.shape[1] == 10)


if __name__ == '__main__':
    unittest.main()
