import tonic
import torchvision

from torch.utils.data import DataLoader 


def choose_dataset(target:str,batch_size:int,T_bin:int):


    return

# Dataset (UNTARGETED and TEST Loader)
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = tonic.transforms.Compose([tonic.transforms.Denoise(filter_time=10000),
                                            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=15)])
    
testset = tonic.datasets.NMNIST(save_to="../dataset", transform=frame_transform, train=False)
test_loader = DataLoader(
    dataset = testset,
    batch_size= batch_size,
    collate_fn= tonic.collation.PadTensors(batch_first=False),
    shuffle = True,
    drop_last=True
)




transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # convert PIL image to PyTorch tensor
    torchvision.transforms.Normalize((0.5,), (0.5,))
    ])  # normalize the input images

# load the test data
testset = torchvision.datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
