import numbers


class Crop(object):
    """Crop the given PIL Image at a given location.

        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
            left (int): left bounding box
            top (int): top bounding box
        """

    def __init__(self, size, left, top):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.left = left
        self.top = top

    def __call__(self, imgs):
        """
        Args:
            imgs (list of PIL Image): Images to be cropped.

        Returns:
            list of PIL Image: Cropped images.
        """
        h, w = self.size

        return [img[:, self.top:self.top+h, self.left:self.left+w] for img in imgs]

