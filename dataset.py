import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def make_numpy_array_from_csv_line(line):
        pixels = [int(x) for x in line.strip().split(',')]
        pixels2d = []
        label = pixels[0]
        for i,x in enumerate(pixels[1:]):
            if i % 28 == 0:
                pixels2d.append([])
            pixels2d[-1].append(x)
        #print(pixels2d, len(pixels2d), len(pixels2d[0]))
        image = np.array(pixels2d, dtype=np.uint8)
        return label, image

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.images = []
        self.labels = []
        with open('MNIST_CSV/mnist_train.csv') as f:
            for line in f:
                label, image = make_numpy_array_from_csv_line(line)
                self.images.append(image)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
