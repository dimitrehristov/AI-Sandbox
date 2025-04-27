# %% [markdown]
# 
# 
#  # Deep Learning: Mastering Neural Networks - Module 5 Assignment: LSTM Sentence Completion
# 
#  Now that we have a framework for working with sequential data in PyTorch - we would like to improve our sentence completion model by introducing a more sophisticated dataset encoding and neural network architecture.
# 
#  In this assignment, we would like you to implement an LSTM model that contains 2 hidden layers and completes sentences at a word level encoding instead of character. We will provide code for cleaning and preparing the data as well as some helper functions so that you can complete the task.
# 
#  Note: This LSTM can take a long time to train. Try using a small number of epochs or a small dataset(~10 samples) to verify your network can train properly before using the full dataset and a larger number of Epochs!

# %% [markdown]
# 
# 
#  ## Dataset and Encoding
# %%

from io import open
import unicodedata
import string
import random
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
PYTORCH_ENABLE_MPS_FALLBACK=1 # Enable MPS fallback for Apple Silicon devices
import time, copy
import matplotlib.pyplot as plt
# import PIL and skimage for image processing and metrics
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu') # Force to use CPU for this assignment
# %%
# %%

# Here we download and unzip the text file that contains all of our translated phrases
import os
import subprocess

# Download the file if it does not exist
if not os.path.exists("spa-eng.zip"):
    subprocess.run(["wget", "https://www.manythings.org/anki/spa-eng.zip"], check=True)
# Unzip the file if it does not exist
if not os.path.exists("spa.txt"):
    subprocess.run(["unzip", "spa-eng.zip"], check=True)
# List the files in the current directory
subprocess.run(["ls"], check=True)

# %%
# Helper functions combined from PyTorch tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
# This is important because we want all words to be formatted the same similar
# to our image normalization
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!'?]+", r" ", s)
    return s

def parse_data(filename):
    # Read the file and split into lines
    lines = open(filename, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Throw out the attribution as it is not a part of the data
    pairs = [[pair[0], pair[1]] for pair in pairs]

    return pairs

# %%

pairs = parse_data("spa.txt")
# We only want the english sentences because we aren't translating
english_sentences = [pair[0] for pair in pairs]
# Shuffle our dataset
random.shuffle(english_sentences)
print("Number of English sentences:", len(english_sentences))

# %%

# %%

# Pick up a sample soze of 6000 sentences
sample_size = 2000
sentences = english_sentences[:sample_size]
print("Sample size:", len(sentences))
print("First 5 sentences:", sentences[:5])
# Here we are using a small number of Sentences to ease training time. Feel free to use more
# train_size = 6000
# train_sentences = english_sentences[:train_size]
# val_sentences = english_sentences[train_size:train_size+1000]
# test_sentences = english_sentences[train_size+1000:train_size+2000]

# Using this function we will create a dictionary to use for our one hot encoding vectors
def add_words_to_dict(word_dictionary, word_list, sentences):
    for sentence in sentences:
        for word in sentence.split(" "):
            if word in word_dictionary:
                continue
            else:
                word_list.append(word)
                word_dictionary[word] = len(word_list)-1

english_dictionary = {}
english_list = []
add_words_to_dict(english_dictionary, english_list, sentences)

# add_words_to_dict(english_dictionary, english_list, train_sentences)
# add_words_to_dict(english_dictionary, english_list, val_sentences)
# add_words_to_dict(english_dictionary, english_list, test_sentences)


# %% [markdown]
# Create a dateset class that will allow us to use the dataloader to create batches of data for training and testing
# 
class SentenceDataset(Dataset):
    def __init__(self, sentences, word_dictionary):
        self.sentences = sentences
        self.word_dictionary = word_dictionary

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_tensor = create_input_tensor(sentence, self.word_dictionary)
        target_tensor = create_target_tensor(sentence, self.word_dictionary)
        return input_tensor, target_tensor

# %% [markdown]
#  ### Encoding
# 
#  We encode the sequence simply by the index of a given word in the dictionary. For the target tensors 
#  End-Of-Sentence (End-Of-Sequence, EOS) equal to negative one (-1) is created

# %%
#  %%
EOS = len(english_dictionary) 
# End of Sentence token
# We will use this token to mark the end of a sentence in our dataset
# Now make tensors that reflect a sentence and its version shifted to the right adding 
# end of sequences token EOS

def create_input_tensor(sentence, word_dictionary): 
    word_to_index = {word: index for index, word in enumerate(word_dictionary.keys())}
    sentence_indices = [word_to_index[word] if word in word_to_index else EOS for word in sentence.split(" ")] 
    input_tensor = torch.tensor(sentence_indices, dtype=torch.long)
    return input_tensor

def create_target_tensor(sentence, word_dictionary):
    word_to_index = {word: index for index, word in enumerate(word_dictionary.keys())}
    sentence_indices = [word_to_index[word] for word in sentence.split(" ")] 
    sentence_indices.append(EOS) # Add EOS token (End of Sentence)
    sentence_indices.pop(0) # Remove the first word to create the target sequence
    
    # Create a tensor for the target sequence
    # The target tensor will be the same as the input but with an EOS token at the end
    # Note: The EOS token is represented by the length of the word_dictionary

    target_seq_tensor = torch.tensor(sentence_indices, dtype=torch.long)
    return target_seq_tensor

# train_tensors = [(create_input_tensor(sentence, english_dictionary), create_target_tensor(sentence, english_dictionary)) for sentence in train_sentences]
# val_tensors = [(create_input_tensor(sentence, english_dictionary), create_target_tensor(sentence, english_dictionary)) for sentence in val_sentences]
# test_tensors = [(create_input_tensor(sentence, english_dictionary), create_target_tensor(sentence, english_dictionary)) for sentence in test_sentences]

# %%

def tensor_to_sentence(dictionary, input_tensor):
    sentence = ""
    sentence_words_indices = input_tensor.tolist() # Convert the tensor to a list for easier processing
    # given a dictionary with words as keys and indices as values create a reverse dictionary
    index_to_word = {index: word for word, index in dictionary.items()}
    # Given the dictionary from index to word, we can create a list of words from the indices

    word_list = []
    for index in sentence_words_indices:
        if index == EOS:
            word_list.append("<EOS>")
        elif index in index_to_word:
            word_list.append(index_to_word[index])

    sentence = " ".join(word_list) # Join the words together to form a sentence
    if (sentence_words_indices[-1] == [EOS]):
        sentence += " <EOS>"
    
    return sentence

print("This code helps visualize which words represent an input_tensor and its corresponding target_tensor!")
examples_to_show = 6

for count in range (examples_to_show):
    tensor = create_input_tensor(sentences[count], english_dictionary)
    target_tensor = create_target_tensor(sentences[count], english_dictionary)
    print("Input tensor: ", tensor.tolist())
    print("Target tensor: ", target_tensor.tolist())
    print("Input sentence: ", sentences[count])
    print("Input decoded sentence: ", tensor_to_sentence(english_dictionary, tensor))
    print("Target sentence: ", tensor_to_sentence(english_dictionary, target_tensor))   

# %%
batch_size = 1
# Create the dataset and dataloaders for training, validation, and testing
train_dataset = SentenceDataset(sentences, english_dictionary)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.1), int(len(train_dataset)*0.1)])

dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size),
               'val': DataLoader(val_dataset, batch_size=batch_size), 
               'test': DataLoader(test_dataset, batch_size=batch_size)}

dataset_sizes = {'train': len(train_dataset),
                 'val': len(val_dataset),
                 'test': len(test_dataset)}

print(f'dataset_sizes = {dataset_sizes}')

# %%
def integer_to_one_hot(indices, vocab_size):
    """
    Convert integer encoding to one-hot encoding.

    Args:
        indices (torch.Tensor): Tensor of integer indices (shape: [batch_size, seq_length]).
        vocab_size (int): Size of the vocabulary.

    Returns:
        torch.Tensor: One-hot encoded tensor (shape: [batch_size, seq_length, vocab_size]).
    """
    # Create a long integet tensor of zeros with shape [batch_size, seq_length, vocab_size]
    one_hot = torch.zeros(indices.size(0), indices.size(1), vocab_size, dtype = torch.float32).to(device)
    
    # Scatter 1s at the appropriate positions
    one_hot.scatter_(2, indices.unsqueeze(-1), 1)
    
    return one_hot

# %% [markdown]
#  ### LSTM Definition
# 
# %%
# %%

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # Embedding layer to convert word indices to vectors
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, dropout=0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input) # Pass the input through the embedding layer
        output, hidden = self.lstm(embedded) # Pass the embedded input through the LSTM layer)
        output = self.fc(output) # Pass the output of the LSTM to the fully connected layer
        return output, hidden

    def initHidden(self):
        # We need two hidden layers because of our two layered lstm!
        # Your model should be able to use this implementation of initHidden()
        return (torch.zeros(2, self.hidden_size).to(device), torch.zeros(2, self.hidden_size).to(device))

# %%
# %%

# TODO
# lstm = LSTM(...)
lstm = LSTM(len(english_dictionary)+1, 64, 256, len(english_dictionary)+1) # +1 for EOS token
lstm.to(device)

# %%

def train_lstm(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
    best_loss = np.inf
    best_epoch = 0

    # Each epoch has a training, validation, and test phase
   # phases = ['train', 'val', 'test']
   # phases = ['train', 'val'] # We will not use the test set for training
    phases = ['train'] # We will only train the model on the training set

    # Keep track of how loss evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for input_sequence, target_sequence in dataloaders[phase]:
                # Now Iterate through each sequence here:

                hidden = model.initHidden() # Start with a fresh hidden state

                current_input_sequence = input_sequence.to(device)


                # convert the intger encodingof the target sequence to a tensor
                # with one hot encoding over the vocabulary size
                current_target_sequence = integer_to_one_hot(target_sequence.to(device), len(english_dictionary)+1) # +1 for EOS token
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0
                    # Make a prediction for each element in the sequence,
                    # keeping track of the hidden state along the way
                    for i in range(current_input_sequence.size(1)):
                        # Need to be clever with how we transfer our hidden layers to the device
                        current_hidden = (hidden[0].to(device), hidden[1].to(device))
                        # Pass the current input through the model
                        # Note: We are passing the i-th element of the sequence to the model
                        output, hidden = model(current_input_sequence[:, i], current_hidden)
                        # reshape the output to batch, 
                        l = criterion(output, current_target_sequence[:, i, :])
                        loss += l
    
                        # Normalize the loss by dividing it by the sequence length
                        # loss = loss / current_input_sequence.size(1)

                    # backward + update weights only if in training phase at the end of a sequence
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() / current_input_sequence.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)

            print(f'{phase:5} Loss: {epoch_loss:.4f}')

            # deep copy the model if it's the best loss
            # Note: We are using the train loss here to determine our best model
            if phase == 'train' and epoch_loss < best_loss:
              best_epoch = epoch
              best_loss = epoch_loss
              best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f} at epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, training_curves


# %%

# We define our predict function here so that we can run some predictions in the same cell as our training!
def predict(model, word_dictionary, word_list, input_sentence, max_length = 20):
    output_sentence = input_sentence + " "
    tensor = create_input_tensor(input_sentence, word_dictionary)
    hidden = model.initHidden()
    # First unsqueze the tensor to make it a batch of size 1
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    current_input_sequence = tensor.to(device)
    input = None

    for i in range(current_input_sequence.size(1)):
        current_hidden = (hidden[0].to(device), hidden[1].to(device))
        output, hidden = model(current_input_sequence[:,i], current_hidden)

    topv, topi = output.topk(1)
    topi = topi[0][0]
    if topi ==  len(word_dictionary):
        topv, topi = output.topk(2)
        topi = topi[0][1]
    word = word_list[topi]
    output_sentence += word
    output_sentence += " "
    input = create_input_tensor(word, word_dictionary).to(device)
    input = input.unsqueeze(0)  # Add batch dimension  of 1
    for i in range(len(input_sentence.split(" ")), max_length):
        current_hidden = (hidden[0].to(device), hidden[1].to(device))
        current_input = input.to(device)
        output, hidden = model(current_input, current_hidden)
        topv, topi = output.topk(1)
        topi = topi[0][0]
        if topi == len(word_dictionary):
            print("Hit the EOS")
            break
        word = word_list[topi]
        output_sentence += word
        output_sentence += " "
        input = create_input_tensor(word, word_dictionary)
        input = input.unsqueeze(0)  # Add batch dimension of 1
    return output_sentence

# %%
# %%

# TODO: Fill in the necessary code to execute the training function
learning_rate = 0.01
num_epochs = 50
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss for classification!
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=0.001) # Use Adam for optimization
# Note: You can use other optimizers as well, but Adam is a good default choice for this task
# Note: You can also experiment with other learning rates and optimizers if you would like
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


lstm, training_curves = train_lstm(lstm, dataloaders, dataset_sizes,
                                     criterion, optimizer, scheduler, num_epochs=num_epochs)

print(predict(lstm, english_dictionary, english_list, "what is"))
print(predict(lstm, english_dictionary, english_list, "my name"))
print(predict(lstm, english_dictionary, english_list, "how are"))
#print(predict(lstm, english_dictionary, english_list, "hi"))
print(predict(lstm, english_dictionary, english_list, "choose"))

# %%
# %%

def plot_training_curves(training_curves,
                         phases=['train', 'val', 'test'],
                         metrics=['loss']):
    epochs = list(range(len(training_curves['train_loss'])))
    for metric in metrics:
        plt.figure()
        plt.title(f'Training curves - {metric}')
        for phase in phases:
            key = phase+'_'+metric
            if key in training_curves:
                plt.plot(epochs, training_curves[key])
        plt.xlabel('epoch')
        plt.legend(labels=phases)
        plt.show()



# %%
# %
plot_training_curves(training_curves, phases=['train'])


