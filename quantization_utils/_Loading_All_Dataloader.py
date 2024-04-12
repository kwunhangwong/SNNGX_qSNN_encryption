import tonic
import random
from torch.utils.data import DataLoader, Subset

# PLEASE Input your path to your dataset
# i.e. dataset_path = "./dataset/" #DEFAULT

def choose_dataset(target:str,batch_size:int,T_BIN:int = 15,dataset_path = 'Please input YOUR PATH TO DATASET DIRECTORY'):

    if (target == "NMNIST"):

        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=sensor_size, 
                                                                             n_time_bins=T_BIN)])

        trainset = tonic.datasets.NMNIST(save_to=dataset_path,transform=frame_transform, train=True)    
        testset = tonic.datasets.NMNIST(save_to=dataset_path, transform=frame_transform, train=False)

        train_loader = DataLoader(
            dataset = trainset,
            batch_size= batch_size,
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = True,
            # drop_last=True
        )

        test_loader = DataLoader(
            dataset = testset,
            batch_size= batch_size,
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = False,
            # drop_last=True
        )

        return train_loader, test_loader
    
    elif (target == "DVS128_Gesture"):

        sensor_size = tonic.datasets.DVSGesture.sensor_size
        # 128 x 128 -> 32 x 32
        reduction_factor = 4
        spatial_factor = 1/reduction_factor
        H, W, P = sensor_size
        sensor_size = int(H/reduction_factor),int(W/reduction_factor), P        

        frame_transform = tonic.transforms.Compose([tonic.transforms.Downsample(spatial_factor=spatial_factor),
                                                    tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T_BIN)])
            
        trainset = tonic.datasets.DVSGesture(save_to=dataset_path,transform=frame_transform, train=True)
        testset = tonic.datasets.DVSGesture(save_to=dataset_path, transform=frame_transform, train=False)

        train_loader = DataLoader(
            dataset = trainset,
            batch_size= batch_size,
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = True,
            # drop_last=True
        )

        test_loader = DataLoader(
            dataset = testset,
            batch_size= batch_size,
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = True,
            # drop_last=True
        )

        return train_loader, test_loader

    else:

        raise ValueError("choose_dataset: Target dataset not recognized. (NMNIST/DVS128)")
    

# Loader for SNNGX Encryption

def UNTARGETED_loader(target:str,num_images:int,batch_size:int,T_BIN:int=15,dataset_path = 'Please input YOUR PATH TO DATASET DIRECTORY'):

    if (target == "NMNIST"):
        #############################################
        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=sensor_size, 
                                                                             n_time_bins=T_BIN)])
        train_set = tonic.datasets.NMNIST(save_to=dataset_path, transform=frame_transform, train=True)

        #############################################
        num_samples = num_images
        num_total_samples = len(train_set)
        random_indices = random.sample(range(num_total_samples), num_samples)
        UNTARGETED_subset = Subset(train_set, random_indices)

        #############################################
        # Create a DataLoader for the subset
        UNTARGETED_loader = DataLoader(
            dataset = UNTARGETED_subset, 
            batch_size= batch_size, 
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = False,
            # drop_last=True
        )

        return UNTARGETED_loader
    
    elif (target == "DVS128_Gesture"):
        #############################################
        sensor_size = tonic.datasets.DVSGesture.sensor_size

        # 128 x 128 -> 32 x 32
        reduction_factor = 4
        spatial_factor = 1/reduction_factor
        H, W, P = sensor_size
        sensor_size = int(H/reduction_factor),int(W/reduction_factor), P        

        frame_transform = tonic.transforms.Compose([tonic.transforms.Downsample(spatial_factor=spatial_factor),
                                                    tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T_BIN)])
        train_set = tonic.datasets.DVSGesture(save_to=dataset_path, transform=frame_transform, train=True)
            
        #############################################
        num_samples = num_images
        num_total_samples = len(train_set)
        random_indices = random.sample(range(num_total_samples), num_samples)
        UNTARGETED_subset = Subset(train_set, random_indices)

        #############################################
        # Create a DataLoader for the subset
        UNTARGETED_loader = DataLoader(
            dataset = UNTARGETED_subset, 
            batch_size= batch_size, 
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = False,
            # drop_last=True
        )

        return UNTARGETED_loader
    
    else:

        raise ValueError("UNTARGETED_LOADER: Target dataset not recognized. (NMNIST/DVS128_Gesture)")

