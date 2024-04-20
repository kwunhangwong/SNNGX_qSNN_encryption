import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import copy
from torchvision import transforms


class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label=0, mode='train', epsilon=0.1, pos='top-left', attack_type='static', time_step=16,
                 trigger_size=0.1, dataname='mnist', polarity=0, n_masks=2, least=False, most_polarity=False):

        self.targets = dataset.targets
        # We need the images loaded instead of the paths
        self.data = np.array([np.array(x[0])
                                for x in dataset])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.time_step = time_step
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform
        self.trigger_label = trigger_label
        self.least = least
        self.most_polarity = most_polarity
        self.pos = pos
        self.polarity = polarity
        self.n_masks = n_masks

        self.data, self.targets = self.add_trigger(
            trigger_label, epsilon, mode, attack_type, trigger_size
        )

        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        # Retrieves the requested image applying the transformations and the label is one-hot encoded
        img = self.data[item]

        targets = self.targets[item]
        if self.transform:
            img = self.transform(img)

        return img, F.one_hot(targets.long(), self.class_num).float()

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[2:]

    def add_trigger(self, trigger_label, epsilon, mode, type, trigger_size):

        print("[!] Generating " + mode + " Bad Imgs")

        new_data = copy.deepcopy(self.data)
        new_targets = copy.deepcopy(self.targets)

        # Ensure that targets are tensors
        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)

        # Choose a random subset of samples to be poisoned
        perm = np.random.permutation(len(new_data))[
            0: int(len(new_data) * epsilon)]

        # Ensure that new_data is a np.array
        if torch.is_tensor(new_data):
            new_data = new_data.numpy()

        width, height = new_data.shape[3:]

        # Swap every samples to the target class
        new_targets[perm] = trigger_label

        size_width = int(trigger_size * width)
        size_height = int(trigger_size * height)

        if epsilon != 0.0:
            if size_width == 0:
                size_width = 1
                size_height = 1

        if len(perm) != 0:
            if type == 'static':
                new_data[perm] = self.create_static_trigger(
                    new_data[perm], size_width, size_height, width, height)
            elif type == 'moving':
                new_data[perm] = self.create_moving_trigger(
                    new_data[perm], size_width, size_height, height, width)
            elif type == 'smart':
                new_data[perm] = self.create_smart_trigger(
                    new_data[perm], size_height, height, width, new_data)
            else:
                raise Exception('Invalid Trigger Type')

            frame = torch.tensor(new_data[perm][0])

        print(
            f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(new_data)-len(perm)}. Epsilon: {epsilon}')

        return torch.Tensor(new_data), new_targets

    def create_static_trigger(self, data, size_width, size_height, width, height):
        pos = self.pos
        polarity = self.polarity

        if pos == 'top-left':
            x_begin = 0
            x_end = size_width
            y_begin = 0
            y_end = size_height

        elif pos == 'top-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = 0
            y_end = size_height

        elif pos == 'bottom-left':
            x_begin = 0
            x_end = size_width
            y_begin = int(height - size_height)
            y_end = height

        elif pos == 'bottom-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = int(height - size_height)
            y_end = height

        elif pos == 'middle':
            x_begin = int((width - size_width) / 2)
            x_end = int((width + size_width) / 2)
            y_begin = int((height - size_height) / 2)
            y_end = int((height + size_height) / 2)

        elif pos == 'random':
            x_begin = np.random.randint(0, int(width-size_width))
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, int(height - size_height))
            y_end = y_begin + size_height

        # The shape of the data is (N, T, C, H, W)
        if polarity == 0:
            data[:, :, :, y_begin:y_end, x_begin:x_end] = 0
        elif polarity == 1:
            data[:, :, 0, y_begin:y_end, x_begin:x_end] = 0
            data[:, :, 1, y_begin:y_end, x_begin:x_end] = 1
        elif polarity == 2:
            data[:, :, 0, y_begin:y_end, x_begin:x_end] = 1
            data[:, :, 1, y_begin:y_end, x_begin:x_end] = 0
        else:
            data[:, :, :, y_begin:y_end, x_begin:x_end] = 1

        return data

    def create_moving_trigger(self, data, size_x, size_y, height, width):
        pos = self.pos
        polarity = self.polarity
        time_step = self.time_step

        if pos == 'top-left':
            start_x = 0
            start_y = 0

            width_list = [start_x, start_x +
                          size_x + 2, start_x + size_x * 2 + 2]
            height_list = [start_y, start_y, start_y]
        elif pos == 'top-right':
            start_x = int(width - size_x)
            start_y = 0

            width_list = [start_x, start_x -
                          size_x - 2, start_x - size_x * 2 - 2]
            height_list = [start_y, start_y, start_y]
        elif pos == 'bottom-left':
            start_x = 0
            start_y = int(height - size_y)

            width_list = [start_x, start_x +
                          size_x + 2, start_x + size_x * 2 + 2]
            height_list = [start_y, start_y, start_y]
        elif pos == 'bottom-right':
            start_x = int(width - size_x)
            start_y = int(height - size_y)

            width_list = [start_x, start_x -
                          size_x - 2, start_x - size_x * 2 - 2]
            height_list = [start_y, start_y, start_y]
        elif pos == 'middle':
            start_x = int(width/2) - 2
            start_y = int(height/2) - 2

            width_list = [start_x, start_x +
                          size_x + 2, start_x + size_x * 2 + 2]
            height_list = [start_y, start_y, start_y]
        elif pos == 'random':
            start_x = np.random.randint(0, int(width - size_x * 2))
            start_y = np.random.randint(0, int(height - size_y * 2))

            width_list = [start_x, start_x +
                          size_x + 2, start_x + size_x * 2 + 2]
            height_list = [start_y, start_y, start_y]

        j = 0
        t = 0

        while t < time_step - 1:
            if j >= len(width_list):
                j = 0

            # The shape of the data is (N, T, C, H, W)
            if polarity == 0:
                data[:, t, :, height_list[j]:height_list[j] +
                     size_y, width_list[j]:width_list[j] + size_x] = 0
            elif polarity == 1:
                data[:, t, 0, height_list[j]:height_list[j] +
                     size_y, width_list[j]:width_list[j] + size_x] = 0
                data[:, t, 1, height_list[j]:height_list[j] +
                     size_y, width_list[j]:width_list[j] + size_x] = 1
            elif polarity == 2:
                data[:, t, 0, height_list[j]:height_list[j] +
                     size_y, width_list[j]:width_list[j] + size_x] = 1
                data[:, t, 1, height_list[j]:height_list[j] +
                     size_y, width_list[j]:width_list[j] + size_x] = 0

            else:
                data[:, t, :, height_list[j]:height_list[j] +
                     size_y, width_list[j]:width_list[j] + size_x] = 1

            j += 1
            t += 1

        return data

    def create_smart_trigger(self, data, size_height, H, W, new_data):

        t_size = size_height
        n = self.n_masks
        least = self.least
        most_polarity = self.most_polarity
        # Split the image into n masks
        n_masks = (n+1)**2
        pol_values = np.zeros((n_masks, 4),  dtype=int)

        masks = get_masks(H, W, n, n_masks)

        # Now, the boundaries of the masks are defined
        # We need to check which mask has the highest value
        # and then we will inject the trigger in that mask
        # if least is True, we will inject the trigger in the mask with the least value

        # Get the maximum value of each mask
        # Since the image is neuromorphic it shape is [batch, T, ...]
        # There we just iterate over each T, and calculate the sum of points with polarity 1

        # ALERT: We are calculating the sum of all the batch! So all the triggers will be in the same mask!
        # We have 4 different combinations for the polarity of the trigger
        # [0,0] Black (This usually is the background, i.e., no movement)
        # [0,1] Dark blue (On polarity)
        # [1,0] Green (Off polarity)
        # [1,1] Light blue (Mixed polarity)

        max_values = np.zeros((n_masks), dtype=int)
        for idx, mask in enumerate(masks):

            # [0,0] case
            p_0 = np.sum(
                new_data[:, :, :, mask[0]:mask[1], mask[2]:mask[3]] == 0
            )

            # [0,1] case
            p_1 = np.sum(np.logical_and(
                new_data[:, :, 0, mask[0]:mask[1], mask[2]:mask[3]] == 0,
                new_data[:, :, 1, mask[0]:mask[1], mask[2]:mask[3]] == 1
            ))

            # [1,0] case
            p_2 = np.sum(np.logical_and(
                new_data[:, :, 0, mask[0]:mask[1], mask[2]:mask[3]] == 1,
                new_data[:, :, 1, mask[0]:mask[1], mask[2]:mask[3]] == 0
            ))

            # [1,1] case
            p_3 = np.sum(
                new_data[:, :, :, mask[0]:mask[1], mask[2]:mask[3]] == 1
            )

            # print('Mask: {}, Polarity 0: {}, Polarity 1: {}, Polarity 2: {}, Polarity 3: {}'.format(
            #     mask, p_0, p_1, p_2, p_3))

            pol_values[idx, 0] = p_0
            pol_values[idx, 1] = p_1
            pol_values[idx, 2] = p_2
            pol_values[idx, 3] = p_3

            # Assuming that p_0 black, i.e., no movement
            # Here we are looking for the more "active" mask
            max_values[idx] = p_1 + p_2 + p_3

        # Get the index of the mask with the highest value
        if least:
            max_index = np.argmin(max_values)
        else:
            max_index = np.argmax(max_values)

        # Now we need to create the moving trigger in the mask with the highest value
        # Take into account the trigger size has to be inside the mask
        mask = masks[max_index]
        safe_h0 = mask[0] + t_size
        safe_h1 = mask[1] - t_size
        safe_w0 = mask[2] + t_size
        safe_w1 = mask[3] - t_size

        # The chosen polarity will be the one with the lowest value (in the chosen mask), so we can make contrast
        if most_polarity:
            polarity = np.argmax(pol_values[max_index])
        else:
            polarity = np.argmin(pol_values[max_index])

        # For each frame t, we will create a trigger in a random position (close the one before t-1) in between the safe boundaries
        # The trigger will be a square of size t_size

        # Get a random position for the trigger in the mask
        h0 = np.random.randint(safe_h0, safe_h1)
        w0 = np.random.randint(safe_w0, safe_w1)

        r = 2  # The closeness of the trigger to the previous one, in pixels
        for t in range(data.shape[1]):
            # Left or right
            while True:
                if np.random.randint(2) == 0:
                    w0 += r
                else:
                    w0 -= r
                if w0 >= safe_w0 and w0 <= safe_w1:
                    break

            # Up or down
            while True:
                if np.random.randint(2) == 0:
                    h0 += r
                else:
                    h0 -= r
                if h0 >= safe_h0 and h0 <= safe_h1:
                    break

            h1 = h0 + t_size
            w1 = w0 + t_size

            # [0, 0] Black
            if polarity == 0:
                data[:, t, :, h0:h1, w0:w1] = 0
            # [0, 1] Dark blue
            elif polarity == 1:
                data[:, t, 0, h0:h1, w0:w1] = 0
                data[:, t, 1, h0:h1, w0:w1] = 1
            # [1, 0] Green
            elif polarity == 2:
                data[:, t, 0, h0:h1, w0:w1] = 1
                data[:, t, 1, h0:h1, w0:w1] = 0
            # [1, 1] Light blue
            else:
                data[:, t, :, h0:h1, w0:w1] = 1

        return data


def create_backdoor_data_loader(args):

    # Get the dataset
    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    train_data = PoisonedDataset(train_data, args.trigger_label, mode='train', epsilon=args.epsilon,
                                 pos=args.pos, attack_type=args.type, time_step=args.T,
                                 trigger_size=args.trigger_size, dataname=args.dataset,
                                 polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity)

    test_data_ori = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=0,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity)

    test_data_tri = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=1,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity)

    frame, label = test_data_tri[0]

    train_data_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader



def get_masks(H, W, n, n_masks):

    masks = np.zeros((n_masks, 4), dtype=int)

    # Get the size of each mask
    size_h = H/(n+1)
    size_w = W/(n+1)
    x = 0
    y = 0
    for mask in masks:
        mask[0] = y
        mask[1] = y + size_h
        mask[2] = x
        mask[3] = x + size_w
        x += size_w
        if x >= W:
            x = 0
            y += size_h

    return masks


def get_most_active_mask(masks, data, least):
    # data can be a single image or a batch of images
    pol_values = np.zeros((masks.shape[0], 4),  dtype=int)
    max_values = np.zeros((masks.shape[0]), dtype=int)

    # Get the maximum value of each mask
    # Since the image is neuromorphic it shape is [batch, T, ...]

    # If data is a single image, we need to add a dimension for the batch
    # and another for the time
    if len(data.shape) == 3:
        data = data[np.newaxis, np.newaxis, :, :]

    for idx, mask in enumerate(masks):

        # [0,0] case
        p_0 = np.sum(
            data[:, :, :, mask[0]:mask[1], mask[2]:mask[3]] == 0
        )

        # [0,1] case
        p_1 = np.sum(np.logical_and(
            data[:, :, 0, mask[0]:mask[1], mask[2]:mask[3]] == 0,
            data[:, :, 1, mask[0]:mask[1], mask[2]:mask[3]] == 1
        ))

        # [1,0] case
        p_2 = np.sum(np.logical_and(
            data[:, :, 0, mask[0]:mask[1], mask[2]:mask[3]] == 1,
            data[:, :, 1, mask[0]:mask[1], mask[2]:mask[3]] == 0
        ))

        # [1,1] case
        p_3 = np.sum(
            data[:, :, :, mask[0]:mask[1], mask[2]:mask[3]] == 1
        )

        pol_values[idx, 0] = p_0
        pol_values[idx, 1] = p_1
        pol_values[idx, 2] = p_2
        pol_values[idx, 3] = p_3

        # Assuming that p_0 black, i.e., no movement
        # Here we are looking for the more "active" mask
        max_values[idx] = p_1 + p_2 + p_3

    if least:
        max_index = np.argmin(max_values)

    else:
        max_index = np.argmax(max_values)

    # Now we need to create the moving trigger in the mask with the highest value
    # Take into account the trigger size has to be inside the mask
    return masks[max_index]