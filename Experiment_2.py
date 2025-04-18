'''
CSCI 5832 Assignment 2
Spring 2025
The following sample code was taken from a tutorial by PyTorch and modified for our assignment.
Source: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
'''
import torch
import random
from tqdm import tqdm
from util import *
import matplotlib.pyplot as plt

class SentimentClassifier(torch.nn.Module):

    def __init__(self, input_dim: int = 6, output_size: int = 1):
        super(SentimentClassifier, self).__init__()

        # Define the parameters that we will need.
        # Torch defines nn.Linear(), which gives the linear function z = Xw + b.
        self.linear = torch.nn.Linear(input_dim, output_size)

    def forward(self, feature_vec):
        # Pass the input through the linear layer,
        # then pass that through sigmoid to get a probability.
        z = self.linear(feature_vec)
        return torch.sigmoid(z)

model = SentimentClassifier()

# the model knows its parameters.  The first output below is X, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, SentimentClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    print(param)


def logprob2label(log_prob):
    # This helper function converts the probability output of the model
    # into a binary label. Use it for the evaluation metrics.
    return log_prob.item() > 0.5

# To run the model, pass in a feature vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sample_feature_vector = torch.tensor([[3.0, 2.0, 1.0, 3.0, 0.0, 4.18965482711792]])
    log_prob = model(sample_feature_vector)
    print('Log probability from the untrained model:', log_prob)
    print('Label based on the log probability:', logprob2label(log_prob))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Sample training loop below. Because it uses functions that you are asked to write for the assignment,     #
# it will not run as is, and is not guaranteed to work with your existing code. You may need to modify it.  #
#                                                                                                           #
# No need to use this code if you have a better way,                                                        #
# or if you can't figure out how to make it run with your existing code.                                    #
# It is only provided here to give you an idea of how we expect you to train the model.                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
num_epochs = 100
batch_size = 16

########################


from collections import Counter
all_texts, all_labels = load_train_data('hotelPosT-train.txt', 'hotelNegT-train.txt')
count_labels = Counter(all_labels)
print(count_labels)



train_texts, train_labels, dev_texts, dev_labels = split_data(all_texts, all_labels)




# Featurize and normalize
train_vectors = [featurize_text(text) for text in train_texts]


dev_vectors = [featurize_text(text) for text in dev_texts]

   
all_dev_preds = []
all_dev_labels = []
dev_losses = []

epoch_i_train_losses = []
dev_epoch_precision = []
dev_epoch_recall = []
dev_epoch_f1 = []
dev_epoch_accuracy = []
dev_epoch_losses = []

for epoch in range(num_epochs):
    # Aggregate data into batches
    samples = list(zip(train_vectors, train_labels))
    random.shuffle(samples)
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

    

    for batch in tqdm(batches):
        feature_vectors, labels = zip(*batch)
        # Step 1. PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run the forward pass.
        log_probs = model(torch.tensor(feature_vectors)).squeeze()
        labels = torch.tensor(labels, dtype=torch.float32)

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, labels)
        loss.backward()
        optimizer.step()

        # (For logging purposes, we will store the loss for this instance)
        epoch_i_train_losses.append(loss.item())
    
    # Print the average loss for this epoch
    print('Epoch:', epoch)
    print('Avg train loss:', sum(epoch_i_train_losses) / len(epoch_i_train_losses))

    model.eval() 
    dev_samples = list(zip(dev_vectors, dev_labels))
    dev_batches = [dev_samples[i:i + batch_size] for i in range(0, len(dev_samples), batch_size)]

    
    
    with torch.no_grad(): 
        for dev_batch in dev_batches:
            dev_features, dev_labels_batch = zip(*dev_batch)

            dev_labels_batch = torch.tensor(dev_labels_batch, dtype=torch.float32)
            dev_log_probs = model(torch.tensor(dev_features)).squeeze()

            dev_loss = loss_function(dev_log_probs, dev_labels_batch)
            dev_losses.append(dev_loss.item())

            dev_preds_batch =   (dev_log_probs > 0.5).float()


            all_dev_preds.extend(dev_preds_batch.cpu().numpy())
            all_dev_labels.extend(dev_labels_batch.cpu().numpy())


    avg_dev_loss = sum(dev_losses) / len(dev_losses)
    dev_epoch_losses.append(avg_dev_loss)

    dev_precision = precision(all_dev_preds, all_dev_labels)
    dev_epoch_precision.append(dev_precision)
    dev_recall = recall(all_dev_preds, all_dev_labels)
    dev_epoch_recall.append(dev_recall)
    dev_f1 = f1(all_dev_preds, all_dev_labels)
    dev_epoch_f1.append(dev_f1)
    dev_accuracy = accuracy(all_dev_preds, all_dev_labels)
    dev_epoch_accuracy.append(dev_accuracy)

    print(f"Epoch: {epoch} |Dev Loss: {avg_dev_loss:.4f} | Dev Precision: {dev_precision:.4f} | Dev Recall: {dev_recall:.4f} | Dev F1: {dev_f1:.4f} | Dev Accuracy: {dev_accuracy:.4f}")

    model.train()  




test_data, test_labels = load_test_data('HW2-testset.txt')
test_vectors = [featurize_text(text) for text in test_data]




with torch.no_grad():
    test_log_probs = model(torch.tensor(test_vectors)).squeeze()
    test_pred=(test_log_probs > 0.5).float()

    test_precision = precision(test_pred,test_labels)
    test_recall = recall(test_pred,test_labels)
    test_f1 = f1(test_pred,test_labels)
    test_accuracy=accuracy(test_pred,test_labels)

    print(f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}  | Test F1: {test_f1:.4f} | Test Accuracy :{test_accuracy:.4f}")


epochs = list(range(1, num_epochs + 1))


plt.plot(epochs, dev_epoch_f1, label='F1 Score')
plt.title('F1 Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.show()

plt.plot(epochs, dev_epoch_accuracy, label='Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(epochs, dev_epoch_losses, label='Dev Loss')
plt.title('Dev Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()