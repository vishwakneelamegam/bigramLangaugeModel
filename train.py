# modules used
import torch
import torch.nn as nn
from torch.nn import functional as F

# device for training
print(torch.backends.mps.is_available())
device = torch.device("mps")

# read the dataset
with open("/Users/vishwakneelamegam/Desktop/mini-transformers/input.txt","r",encoding="utf-8") as f:
    text = f.read()

# characters in the dataset
characters = sorted(list(set(text)))

# size of the characters
characterSize = len(characters)

# mapping the characters
def characterMap():
    count = 0
    sToI = {}
    iToS = {}
    for char in characters:
        sToI[char] = count
        iToS[count] = char
        count += 1
    return (sToI, iToS)


# loading  the character map
charMap = characterMap()

# encode characters
def encodeCharacter(data):
    result = []
    for char in data:
        result.append(charMap[0][char])
    return result

# decode characters
def decodeCharacter(data):
    result = []
    for num in data:
        result.append(charMap[1][num])
    return "".join(result)

# encode all the text and make as torch tensors
data = torch.tensor(encodeCharacter(text), dtype=torch.long)

# used to split training and validation data
splitSize = int(1.0 * len(data))

# train data
trainData = data[:splitSize]

# validate data
validateData = data[splitSize:]

# learning rate
learningRate = 3e-4

# block size (maximum text length)
blockSize = 256

# batch size (the transformers can parallely process the text, so batch size is mentioned)
batchSize = 64

# embedding size
embeddingSize = 384

# number of layers
numberOfLayers = 6

# number of heads
numberOfHeads = 6

# drop out
dropOut = 0.2

# torch manual seed
#torch.manual_seed(40)

# used to get batch of data for training or validation
def getBatch(toTrain = True):
    if toTrain == True:
        data = trainData
    else:
        data = validateData
    # here comma is added (batchSize,)
    randomData = torch.randint(len(data) - blockSize, (batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in randomData])
    y = torch.stack([data[i+1:i+blockSize+1] for i in randomData])
    x = x.to(device)
    y = y.to(device)
    return x, y

# decoder block
class block(nn.Module):
    def __init__(self,embeddingSize,numberOfHeads):
        super().__init__()
        headSize = embeddingSize // numberOfHeads
        self.selfAttention = multiHeadAttention(numberOfHeads, headSize)
        self.feedForwardLayer = feedForward(embeddingSize)
        self.layerNormOne = nn.LayerNorm(embeddingSize)
        self.layerNormTwo = nn.LayerNorm(embeddingSize)
    
    def forward(self, input):
        input = input + self.selfAttention(self.layerNormOne(input))
        input = input + self.feedForwardLayer(self.layerNormTwo(input))
        return input

# self attention
class head(nn.Module):
    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(embeddingSize,headSize,bias = False)
        self.query = nn.Linear(embeddingSize,headSize,bias = False)
        self.value = nn.Linear(embeddingSize,headSize,bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(blockSize, blockSize)))
        self.dropout = nn.Dropout(dropOut)
    
    def forward(self, input):
        B, T, C = input.shape
        k = self.key(input)
        q = self.query(input)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        weight = q @ k.transpose(-2,-1) * C**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = F.softmax(weight, dim = 1)
        weight = self.dropout(weight)
        v = self.value(input)
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        output = weight @ v
        return output 

# multi head attention
class multiHeadAttention(nn.Module):
    def __init__(self, numberOfHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([head(headSize) for _ in range(numberOfHeads)])
        self.projections = nn.Linear(embeddingSize, embeddingSize)
        self.dropout = nn.Dropout(dropOut)
    
    def forward(self, input):
        output = torch.cat([h(input) for h in self.heads], dim=-1)
        output = self.dropout(self.projections(output))
        return output
    
# feed forward
class feedForward(nn.Module):
    def __init__(self, embeddingSize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embeddingSize, 4 * embeddingSize),
              nn.ReLU(),
              nn.Linear(4 * embeddingSize, embeddingSize),
              nn.Dropout(dropOut),
              )

    def forward(self, input):
        return self.net(input)

# bigram language model
class bigramLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(characterSize, embeddingSize)
        self.positionEmbeddingTable = nn.Embedding(blockSize, embeddingSize)
        self.blocks = nn.Sequential(*[block(embeddingSize,numberOfHeads=numberOfHeads) for _ in range(numberOfLayers)])
        self.layerNormalFunction = nn.LayerNorm(embeddingSize)
        # 4 heads of 8 dimension self attention
        self.languageModelHead = nn.Linear(embeddingSize, characterSize)
    
    def forward(self, input, target = None):
        B, T = input.shape
        # provides (batch, times, channel)
        tokenEmbed = self.tokenEmbeddingTable(input)
        # provides (times, channel)
        positionEmbed = self.positionEmbeddingTable(torch.arange(T, device=device))
        # provides (batch, times, channel)
        totalEmbed = tokenEmbed + positionEmbed
        totalEmbed = self.blocks(totalEmbed)
        totalEmbed = self.blocks(totalEmbed)
        # provides (batch, times, character size)
        logits = self.languageModelHead(totalEmbed)
        if target is  None:
            loss = None
        else:
            B, T, C = logits.shape
            # .view() is used to reshape the matrix
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            # calculate loss
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, input, maxTokens):
        for _ in range(maxTokens):
            # crop the input
            inputCropped = input[:,-blockSize:]
            # get loss and logits
            logits, loss = self(inputCropped)
            # focusing only on the last time step so removing T from B T C. we get B C
            logits = logits[:, -1, :] 
            # apply the logits to the softmax to get the probability. we get B C
            probability = F.softmax(logits, dim=1)
            # sample from distribution
            nextInput = torch.multinomial(probability,num_samples=1)
            # append the sample input or ids. we get B T + 1
            input = torch.cat((input, nextInput), dim=1)
        return input

# BLL model
bllModel = bigramLLM()
model = bllModel.to(device)

# creating pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

# train area
for steps in range(10000):
    xData, yData = getBatch()
    logits, loss = model(xData, yData)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"Loss : {loss.item()} Iteration : {steps + 1}")

# test area
print(decodeCharacter(model.generate(input=torch.zeros((1,1),dtype=torch.long, device=device),maxTokens=500)[0].tolist()))
