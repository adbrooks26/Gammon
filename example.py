import numpy as np
import datagen

import torch 
from torch import nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt

# defining our dataset//construction of dataset

# the reason this is put here is for adaptability -- i.e. to have the ability to input more sources, could be move into datagen python file

def createdata(num):
    subdata = []
    key = []
    for n in tqdm(range(num)):
        
        peakHeight = np.random.uniform(50.,500.)
        compStr = np.random.uniform(1.,50.)
        xrayHeight = np.random.uniform(10.,500.)
        resVar = [np.random.uniform(0.,1.),np.random.normal(-0.5,0.1)]
        calibVar = [np.random.uniform(0.,20.),np.random.uniform(0.25,0.75),np.random.uniform(0.,1e-3)] 
        # need to be a bit careful with random generation of calibration coefficients as this can be problematic with interpolation 
        nStr = np.random.normal(7.,2.)
        bStr = np.random.uniform(0.,2.)


        class_select = np.random.randint(0,9) # to increase number of classes and hence sources the ML can identify then change this

        if class_select == 0:
            # Cs-137
            gEng = 662
            gIntensity = 0.8510
            xrayEng = 33.3
            xrayIntensity = 1.0

            label = np.asarray(0)

        elif class_select == 1:
            # Co-66
            gEng = np.asarray([1173.228,1332.492,2158.57])
            gIntensity = np.asarray([0.9985,0.999826,0.000012])
            xrayEng = 7.8
            xrayIntensity = 0.00001

            label = np.asarray(1)

        elif class_select == 2:
            # Na-22
            gEng = np.asarray([511.0,1274.537])
            gIntensity = np.asarray([1.7991,0.9994])
            xrayEng = 0.848
            xrayIntensity = 0.001

            label = np.asarray(2)

        elif class_select == 3:
            # Ti-44
            gEng = np.asarray([67.8679,78.3234,146.22,511.,1157.022,1499.449])
            gIntensity = np.asarray([0.93,0.968,0.01,1.88556*0.5,0.998867*0.5,0.00909*0.5])
            xrayEng = 4.1
            xrayIntensity = 0.11

            label = np.asarray(3)

        elif class_select == 4:
            # Ba-133
            gEng = np.asarray([53.1622,79.6142,80.9979,160.6120,223.2368,276.3989,302.8508,356.0129,383.8485])
            gIntensity = np.asarray([0.0214,0.0265,0.329,0.00638,0.00453,0.0716,0.1834,0.6205,0.0894])
            xrayEng = 34
            xrayIntensity = 0.6

            label = np.asarray(4)

        elif class_select == 5:
            # Am-241
            gEng = 59.5409
            gIntensity = 0.359
            xrayEng = 13.9
            xrayIntensity = 0.37

            label = np.asarray(5)

        elif class_select == 6:
            # Bi-207
            gEng = np.asarray([511,569.698,897.77,1063.656,1442.2,1770.228])
            gIntensity = np.asarray([0.00076,0.9775,0.00128,0.745,0.001310,0.0687])
            xrayEng = np.asarray([10.6,73,85])
            xrayIntensity = np.asarray([0.332,0.3,0.05])

            label = np.asarray(6)

        elif class_select == 7:
            # Ag-108m
            gEng = np.asarray([433.937,614.276,722.907])
            gIntensity = np.asarray([0.905,0.898,0.908])
            xrayEng = 22.
            xrayIntensity = 0.4

            label = np.asarray(7)

        elif class_select == 8:
            # Ge-68
            gEng = np.asarray([511.,1077.34,1115.539])
            gIntensity = np.asarray([1.8,0.0322,0.5004*0.2])
            xrayEng = np.asarray([9.25,10.26])
            xrayIntensity = np.asarray([0.3,0.03])

            label = np.asarray(8)


        # datagen.GammaSpecConstruct takes in the following inputs:
        # photopeaklist
        # photopeakscales
        # comptonstrength
        # xraylist
        # xrayscales
        # resolution_var -- p0,p1
        # calibration_var -- p0,p1,p2
        # noiseStr
        # backscatteron
        # bsStr
        # torchconvert

        _,geneddata = datagen.GammaSpecConstruct(gEng,gIntensity*peakHeight,compStr,xrayEng,xrayIntensity*xrayHeight,resVar,calibVar,nStr,True,bStr,True)

        subdata.append(geneddata)
        key.append(label)
    return torch.tensor(np.asarray(subdata).astype(np.float32)),torch.tensor(np.asarray(key))

print("--- Creating training data --- ")
data_train,keys_train = createdata(6000) # create both training data and testing data for ML traingin and validation per epoch
print("--- Creating testing data --- ")
data_test,keys_test = createdata(6000)


class CustomDataset(Dataset):   # used to make custom data into a pytorch dataset
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    

dataset_train = CustomDataset(data_train,keys_train)
dataset_test = CustomDataset(data_test,keys_test)

train_dataloader = DataLoader(dataset_train,batch_size=30,shuffle=True,num_workers=5)   # converts customdata into a DataLoader object for use with pytorch
test_dataloader = DataLoader(dataset_test,batch_size=30,shuffle=True,num_workers=5) # you can specify the batch size here and also the number of cpu workers
# probably important to bear in mind that the data set needs to be larger than the number of epochs*batchsize otherwise you will repeat training and test data 



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # used for gpu speed up
print(f"Using {device} device")

### machine learning class here ... 
