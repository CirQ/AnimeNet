import os

from PIL import Image
from torch import FloatTensor
import torch.utils.data as data
import torchvision.transforms as transform

class TagImageDataset(data.Dataset):
    def __init__(self, tag_path, img_path):
        super(TagImageDataset, self).__init__()
        _cwd = os.getcwd()
        self.tag = os.path.join(_cwd, tag_path)
        self.img = os.path.join(_cwd, img_path)
        self.attr_list = os.listdir(self.tag)
        self.transform = transform.Compose([
            transform.ToTensor()
        ])

    def __getitem__(self, index):
        tag = self.attr_list[index]
        img = tag.replace('txt', 'jpg')
        tag_path = os.path.join(self.tag, tag)
        img_path = os.path.join(self.img, img)

        with open(tag_path, 'r') as r:
            t = FloatTensor(eval(r.read()))
        i = self.transform(Image.open(img_path))
        return t, i

    def __len__(self):
        return len(self.attr_list)


def unit_test():
    dataset = TagImageDataset('small_tag', 'small_image')
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    for i, (tag, img) in enumerate(dataloader, start=1):
        print(i, tag.size(), img.size())


if __name__ == '__main__':
    unit_test()
