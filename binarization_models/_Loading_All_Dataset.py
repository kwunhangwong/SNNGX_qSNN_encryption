import tonic
import torchvision

from torch.utils.data import DataLoader 

# PLEASE Input your path to your dataset
# dataset_path = "./dataset" #DEFAULT

def choose_dataset(target:str,batch_size:int,T_BIN:int = 15,dataset_path = "../../BSNN_Project/N-MNIST_TRAINING/dataset"):

    if (target == "NMNIST"):

        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tonic.transforms.Compose([tonic.transforms.Denoise(filter_time=10000),
                                                    tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T_BIN)])

        trainset = tonic.datasets.NMNIST(save_to=dataset_path,transform=frame_transform, train=True)    
        testset = tonic.datasets.NMNIST(save_to=dataset_path, transform=frame_transform, train=False)

        train_loader = DataLoader(
            dataset = trainset,
            batch_size= batch_size,
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = True,
            drop_last=True
        )

        test_loader = DataLoader(
            dataset = testset,
            batch_size= batch_size,
            collate_fn= tonic.collation.PadTensors(batch_first=False),
            shuffle = False,
            drop_last=True
        )

        return train_loader, test_loader

    elif (target == "MNIST"):

        transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),  # convert PIL image to PyTorch tensor
                    torchvision.transforms.Normalize((0.5,), (0.5,))
                    ])  

        trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)
        
        train_loader = DataLoader(
            dataset = trainset, 
            batch_size=batch_size, 
            shuffle=True)
        
        test_loader = DataLoader(
            dataset=testset, 
            batch_size=batch_size, 
            shuffle=False)
        
        return train_loader, test_loader

    else:

        raise ValueError("Target dataset not recognized. (NMNIST/MNIST)")
