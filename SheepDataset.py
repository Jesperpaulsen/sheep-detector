from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io, transform
import torchvision.transforms as transforms
from PIL import Image

transformer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


class SheepDataset(Dataset):
    def __init__(self, annot_path, images_path):
        self.sheep_frame = pd.read_csv(annot_path, header=None)
        self.images_path = images_path

    def __len__(self):
        return len(self.sheep_frame)

    def __getitem__(self, idx: torch.tensor):
        if torch.is_tensor(idx):
            idx = idx.toList()
        print('item: ' + str(idx))
        listing = self.sheep_frame.iloc[idx].tolist()

        image_path = listing[0]
        image = Image.open(image_path)
        rgb_image = image.convert('RGB')

        transformed_image = transformer(to_tensor(rgb_image))
        boxes = torch.tensor(listing[1:5])
        labels = torch.tensor(listing[5])

        return transformed_image, {'boxes': boxes, 'labels': labels}
