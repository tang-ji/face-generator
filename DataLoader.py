from cv2 import imread, resize
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, input_path='input/', output_path='output/', img_res=(218, 178)):
        self.input_path = input_path
        self.output_path = output_path
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        path = glob(self.input_path + '*')
        batch_images = np.random.choice(path, batch_size)
        imgs_in = []
        imgs_out = []
        for img_path in batch_images:
            name = img_path.split('/')[-1]
            if name[-3:] != 'jpg' and name[-3:] != 'png':
                continue
            img_in = imread(self.input_path + name)[:,:,::-1]
            img_out = imread(self.output_path + name)[:,:,::-1]

            img_in = resize(img_in, self.img_res)
            img_out = resize(img_out, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_in = np.fliplr(img_in)
                img_out = np.fliplr(img_out)

            imgs_in.append(img_in)
            imgs_out.append(img_out)

        imgs_in = np.array(imgs_in)/127.5 - 1.
        imgs_out = np.array(imgs_out)/127.5 - 1.

        return imgs_in, imgs_out
    
    def load_batch(self, batch_size=1, is_testing=False):
        path = glob(self.input_path + '*')
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_in = []
            imgs_out = []
            for img in batch:
                name = img.split('/')[-1]
                if name[-3:] != 'jpg' and name[-3:] != 'png':
                    continue
                img_in = imread(self.input_path + name)[:,:,::-1]
                img_out = imread(self.output_path + name)[:,:,::-1]

                img_in = resize(img_in, self.img_res)
                img_out = resize(img_out, self.img_res)

                # If training => do random flip
                if not is_testing and np.random.random() < 0.5:
                    img_in = np.fliplr(img_in)
                    img_out = np.fliplr(img_out)

                imgs_in.append(img_in)
                imgs_out.append(img_out)

            imgs_in = np.array(imgs_in)/127.5 - 1.
            imgs_out = np.array(imgs_out)/127.5 - 1.

            yield imgs_in, imgs_out
