import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------------------------
# Encoder CNN using pre-trained ResNet
# -----------------------------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):  # ✅ Fixed __init__
        super(EncoderCNN, self).__init__()  # ✅ Fixed __init__
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# -----------------------------------------------
# Decoder RNN (LSTM)
# -----------------------------------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):  # ✅ Fixed __init__
        super(DecoderRNN, self).__init__()  # ✅ Fixed __init__
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20):
        "Generate caption from image feature (inference)"
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

# -----------------------------------------------
# Image preprocessing function
# -----------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------
# Example usage
# -----------------------------------------------
image_path = r"C:\Users\suren\Desktop\images.jpeg.jpg"
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension

# Hyperparameters (example values)
embed_size = 256
hidden_size = 512
vocab_size = 5000  # Replace with actual vocab size in real application

# Initialize models
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()

# Extract features
features = encoder(image)

# Dummy caption tensor for training example
dummy_captions = torch.randint(0, vocab_size, (1, 10))

# Forward pass (training)
outputs = decoder(features, dummy_captions)
print("Decoder outputs shape:", outputs.shape)

sampled_ids = decoder.sample(features)
print("Sampled IDs:", sampled_ids)

# -----------------------------------------------
# Example: Convert IDs to words (fake word_map)
# -----------------------------------------------
# Example word map
word_map = {0: '<pad>', 1: '<start>', 2: 'a', 3: 'dog', 4: 'on', 5: 'grass', 6: '<end>'}
sentence = []
for idx in sampled_ids[0]:
    word = word_map.get(idx.item(), '<unk>')
    if word == '<end>':
        break
    sentence.append(word)

print("Generated caption:", ' '.join(sentence))
