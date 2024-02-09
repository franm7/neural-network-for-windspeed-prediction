import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

with open('data/wind_dataset.csv', 'r') as file:
    lines = file.readlines()
    windSpeeds = np.zeros(len(lines) - 1, np.float32)
    dates = []
    for i in range(1, len(lines)):
        line = lines[i].strip().split(",")
        windSpeeds[i-1] += np.float32(line[1])
        dates.append(line[0])

months = {'01': 0, '02': 31, '03': 59, '04': 90, '05': 120, '06' : 151, '07': 181, '08': 212, '09': 243, '10': 273, '11': 304, '12': 334}
inputs = []
for date in dates:
    date = date.split('-')
    if int(date[0]) % 4 != 0:
        dayOfYear = months[date[1]] + int(date[2])
        angle = 2 * np.pi * (dayOfYear - 1)/365
        tuple = [np.sin(angle), np.cos(angle)]
    else:
        if date[1] == '01' or date[1] == '02':
            dayOfYear = months[date[1]] + int(date[2])
        else:
            dayOfYear = months[date[1]] + int(date[2]) + 1
        angle = 2 * np.pi * (dayOfYear - 1)/366
        tuple = [np.sin(angle), np.cos(angle)]
    inputs.append(tuple)

trainFeatures = inputs[:4600]
trainTargets = windSpeeds[:4600]
validationFeatures = inputs[4600:5550]
validationTargets = windSpeeds[4600:5550]
testFeatures = inputs[5550:]
testTargets = windSpeeds[5550:]

trainFeaturesTensor = torch.tensor(trainFeatures, dtype=torch.float32)
trainFeaturesTensor = trainFeaturesTensor.unsqueeze(0)
trainTargetsTensor = torch.tensor(trainTargets, dtype=torch.float32)
trainTargetsTensor = trainTargetsTensor.unsqueeze(0).unsqueeze(2)

validationFeaturesTensor = torch.tensor(validationFeatures, dtype=torch.float32)
validationFeaturesTensor = validationFeaturesTensor.unsqueeze(0)
validationTargetsTensor = torch.tensor(validationTargets, dtype=torch.float32)
validationTargetsTensor = validationTargetsTensor.unsqueeze(0).unsqueeze(2)

testFeaturesTensor = torch.tensor(testFeatures, dtype=torch.float32)
testFeaturesTensor = testFeaturesTensor.unsqueeze(0)
testTargetsTensor = torch.tensor(testTargets, dtype=torch.float32)
testTargetsTensor = testTargetsTensor.unsqueeze(0).unsqueeze(2)



class WindSpeedLSTM(nn.Module):
    def __init__(self, hiddenSize):
        super(WindSpeedLSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.lstm1 = nn.LSTMCell(2, self.hiddenSize)
        self.lstm2 = nn.LSTMCell(self.hiddenSize, self.hiddenSize)
        self.linear = nn.Linear(self.hiddenSize, 1)
       
    def forward(self, input):
        batchSize = input.size(0)
        seqLen = input.size(1)

        h1 = torch.zeros(batchSize, self.hiddenSize)
        c1 = torch.zeros(batchSize, self.hiddenSize)
        h2 = torch.zeros(batchSize, self.hiddenSize)
        c2 = torch.zeros(batchSize, self.hiddenSize)

        outputs = []
        for t in range(seqLen):
            x = input[:, t, :]

            h1, c1 = self.lstm1(x, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            output = self.linear(h2)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1) 
        return outputs
    

model = WindSpeedLSTM(hiddenSize=10)
criterion = nn.L1Loss()
learningRate = 0.15
optimizer = optim.SGD(model.parameters(), lr=learningRate)

epochs = 201
outputs = []
lastValLoss = 1000

for epoch in range(epochs):

    model.train()
    optimizer.zero_grad()
    outputs = model(trainFeaturesTensor)
    loss = criterion(outputs, trainTargetsTensor)
    loss.backward()
    optimizer.step()


    model.eval()
    validationOutputs = model(validationFeaturesTensor)
    validationLoss = criterion(validationOutputs, validationTargetsTensor)

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs-1}], Loss: {loss.item()}, Validation Loss: {validationLoss.item()}")
        if validationLoss > lastValLoss:
            print(f'break, {epoch}')
            break
        lastValLoss = validationLoss



outputs = outputs.squeeze().detach().numpy()
validationOutputs = validationOutputs.squeeze().detach().numpy()
testOutputs = model(testFeaturesTensor)
testOutputs = testOutputs.squeeze().detach().numpy()

x = [i for i in range(1, 6575)]
startYear = 1961
endYear = 1979
step = 4 * 365
ticks = []
tickLabels = []
currentYear = startYear
currentIndex = 0

while currentYear <= endYear:
    ticks.append(currentIndex)
    tickLabels.append(currentYear)
    
    if currentYear % 4 == 0:
        currentIndex += 365
    else:
        currentIndex += 366
    currentYear += 1


plt.figure(figsize=(15, 10))
plt.plot(x, windSpeeds, 'ro')
plt.plot(x[:4600], outputs, 'b', linewidth=5.0)
plt.plot(x[4600:5550], validationOutputs, 'g', linewidth=5.0)
plt.plot(x[5550:], testOutputs, 'y', linewidth=5.0)

plt.xticks(ticks, tickLabels)
plt.savefig("diagrams/NNpredictions.png")

print(outputs[:15])
print(validationOutputs[:15])
print(testOutputs[:15])