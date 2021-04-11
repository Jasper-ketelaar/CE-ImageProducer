import os
from typing import Dict, List, Callable, Union, Any, Tuple

import cv2
import numpy as np


class OGImage:

    def __init__(self, angle, res, path, sku, image=None):
        self.res = res
        self.angle = angle
        self.path = path
        self.sku = sku
        self.image = None

    def load_as_cv2(self):
        self.image = cv2.imread(self.path)

    def rescale_to_target(self, target=450, keep_instance=True):
        resized = cv2.resize(
            self.image,
            (target, target),
            interpolation=cv2.INTER_AREA if self.res < target else cv2.INTER_NEAREST
        )

        if keep_instance:
            self.image = resized
            return self
        else:
            og_alt = OGImage(self.angle, self.res, self.path, self.sku)
            og_alt.image = resized

    def show(self, *args, operation: Union[*Callable[[*Any], np.int8], None] = None):
        image = self.image
        if operation is not None:
            image = operation(*args)
            if isinstance(operation, Tuple):
                imgs = np.zeros(len(operation))
                for idx, op in enumerate(operation):
                    imgs += (op(*args[idx]))
                image = imgs.mean()

        cv2.imshow(f'{self.sku} - ({self.angle}, {self.res})', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def salt_pepper(self) -> np.int8:
        noise = np.zeros(self.image.shape, np.int8)
        cv2.randn(noise, np.zeros(3), np.ones(3) * 255 / (1.0 / (self.res / (self.res ** 1.5))))
        return cv2.add(noise, self.image, dtype=cv2.CV_8UC3)

    def smoothen(self) -> np.int8:
        kernel = \
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) * np.array([[0, 1, 0], [1, 1.25, 1], [1, 0, 1]])
        dst = cv2.filter2D(self.image, -1, kernel)
        return dst

    def sharpen(self, amount):
        img = self.image.copy()
        blurred = cv2.GaussianBlur(img, (5, 5), 1)
        sharpened = float(1 + amount) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        low_contrast_mask = np.absolute(img - blurred) < 50
        np.copyto(sharpened, img, where=low_contrast_mask)
        return sharpened


def load_image_dict(og_dir='../originals') -> Dict[str, List[OGImage]]:
    image_dict = dict()
    for sku in os.listdir(og_dir):
        image_dict[sku] = []

        for im_version in os.listdir(f'../originals/{sku}'):
            var_reqs, ext = im_version.split('.')
            angle, res = [int(x) for x in var_reqs.split('-')]
            image_dict[sku].append(OGImage(angle, res, f'../originals/{sku}/{angle}-{res}.{ext}', sku))

    return image_dict


if __name__ == '__main__':
    images: Dict[str, List[OGImage]] = load_image_dict()
    arna_imgs: List[OGImage] = images['ASTR-17005']
    for arna in arna_imgs:
        arna.load_as_cv2()
        arna.rescale_to_target()
        arna.show(operation=(arna.smoothen, arna.salt_pepper))
