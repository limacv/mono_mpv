import torch
import torch.nn.functional as torchf
import numpy as np
from torchvision.transforms.functional import _get_inverse_affine_matrix


class RandomAffine:
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):

        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
            "degrees should be a list or tuple and it must be of length 2."
        self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            assert isinstance(shear, (tuple, list)) and \
                   (len(shear) == 2 or len(shear) == 4), \
                "shear should be a list or tuple and it must be of length 2 or 4."
            # X-Axis shear with [min, max]
            if len(shear) == 2:
                self.shear = [shear[0], shear[1], 0., 0.]
            elif len(shear) == 4:
                self.shear = [s for s in shear]
        self.shear = shear

        self.grid = None

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        wid, hei = img_size
        angle = np.random.uniform(degrees[0], degrees[1])

        if translate is not None:
            max_dx = translate[0] * 2 / wid
            max_dy = translate[1] * 2 / hei
            translations = (np.random.uniform(-max_dx, max_dx),
                            np.random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [np.random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [np.random.uniform(shears[0], shears[1]),
                         np.random.uniform(shears[2], shears[3])]
            else:
                raise RuntimeError()
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def random_generate(self, img: torch.Tensor):
        batchsz, cnl, hei, wid = img.shape
        affine_mats = []
        for i in range(batchsz):
            angle, translations, scale, shear = self.get_params(self.degrees,
                                                                self.translate,
                                                                self.scale,
                                                                self.shear,
                                                                (wid, hei))
            affine_mat = _get_inverse_affine_matrix((0, 0), angle, translations, scale, shear)
            affine_mat = torch.tensor(affine_mat).type_as(img).reshape(1, 2, 3)
            affine_mats.append(affine_mat)
        affine_mats = torch.cat(affine_mats, dim=0)
        self.grid = torchf.affine_grid(affine_mats, [batchsz, cnl, hei, wid]).type_as(img)

    def apply(self, content: torch.Tensor):
        assert self.grid is not None, "Grid Not Initialized"
        img_warped = torchf.grid_sample(content, self.grid, mode="bilinear", padding_mode="border")
        return img_warped


if __name__ == "__main__":
    mat = _get_inverse_affine_matrix((0, 0), 90, (0, 0), 1, 0)
    img = torch.ones(1, 3, 100, 200)
    img[0, :, 21:41, :] = torch.tensor([1, 0, 0.]).reshape(3, 1, 1)

    # img_warp = RandomAffine((-1, 1), (-2, 2), (0.97, 1.03), (-1, 1, -1, 1))(img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img[0].permute(1, 2, 0))
    # plt.figure()
    # plt.imshow(img_warp[0].permute(1, 2, 0))
    # plt.show()
