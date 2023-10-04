# Libraries
import os
import random
import shutil
from itertools import islice

# Folder
OutputFolderPath = 'CustomObjectDetect/SplitData'
InputFolderPath = 'CustomObjectDetect/All'
splitRatio = {"train":0.7, "val":0.2, "test":0.1}
classes = ["Gafas"]

try:
    shutil.rmtree(OutputFolderPath)
    print("Remove Directory")
except OSError as e:
    os.mkdir(OutputFolderPath)

# Folders
os.makedirs(f"{OutputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{OutputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{OutputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{OutputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{OutputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{OutputFolderPath}/test/labels", exist_ok=True)

# Img Names
listNames = os.listdir(InputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))
print(len(uniqueNames))

# Shuffle
random.shuffle(uniqueNames)

# Img Number Folders
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])
print(f'Total Images:{lenData} \nSplit Train: {lenTrain} Split Val: {lenVal} Split Test: {lenTest}')

# Img train
if lenData != lenTrain+lenVal+lenTest:
    remaining = lenData - (lenTrain+lenVal+lenTest)
    lenTrain += remaining

# Split
lengthSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem))for elem in lengthSplit]

# Copy
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{InputFolderPath}/{fileName}.jpg', f'{OutputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{InputFolderPath}/{fileName}.txt', f'{OutputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("PROCESS SPLIT COMPLETED")

# Data.yaml
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'



f = open(f"{OutputFolderPath}/Dataset.yaml", 'a')
f.write(dataYaml)
f.close()

print("DATASET.YAML COMPLETED")