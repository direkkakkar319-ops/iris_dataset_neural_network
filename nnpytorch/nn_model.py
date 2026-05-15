# %% [markdown]
# ## Builidng a nn model

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# Creating a model class that inherits nn.module
class Model(nn.Module):
  # Input layer (4features of flowers)->
  # hidden layer 1->
  # hidden layer 2->
  # output(3 classes of irir flowers)
  def __init__(self, inFeatures=4, h1=8, h2=9, outFeature=3):
    super().__init__()
    self.fc1=nn.Linear(inFeatures, h1)
    self.fc2=nn.Linear(h1, h2)
    self.out=nn.Linear(h2, outFeature)

  def forward(self, x):
    x=F.relu(self.fc1(x))#Rectified linear unit
    # if output less than zero considered as zero
    x=F.relu(self.fc2(x))
    x=self.out(x)
    return x

# %%
# Picking a random seed for randomization
torch.manual_seed(32)
# creating an instance of our model
model=Model()

# %%
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# %%
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_dataframe = pd.read_csv(url)

# %%
my_dataframe['species'] = my_dataframe['species'].replace("setosa", 0.0)
my_dataframe['species'] = my_dataframe['species'].replace("versicolor", 1.0)
my_dataframe['species'] = my_dataframe['species'].replace("virginica", 2.0)

# %%
# train test and split:
X=my_dataframe.drop('species', axis=1)
y=my_dataframe['species']

# %%
# Converting it to numpy strings
X=X.values
y=y.values

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# %%
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

# %%
# LOng tesnor are 64but integers
if isinstance(y_train, torch.Tensor):
    y_train = y_train.long()
else:
    y_train = torch.LongTensor(y_train)

if isinstance(y_test, torch.Tensor):
    y_test = y_test.long()
else:
    y_test = torch.LongTensor(y_test)

# %%
# Set criterion of model to measure the error, how far off the prediction are form the data
criterion = nn.CrossEntropyLoss()

# %%
# Choose Adam optimizer, lr= set our learniing rate(if eror dosent go down after a bunch of iteration (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %% [markdown]
# ### traning our model

# %%
# Train our model
# Epoch? (one run thru all the traning data in our network)
epochs = 100
losses = []
for i in range(epochs):
  # Go forward
  y_pred = model.forward(X_train) # Getting our predictted results
  # MEasure the loss or error -going to be high first
  loss = criterion(y_pred, y_train)#predicted values vs the y_train
  # Keep trac of our losses
  losses.append(loss.detach().numpy())
  # Print every ten epoch
  if i %10==0:
    print(f"Epoch: {i} and loss:{loss}")
  # Do some back propagation take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# %%
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("epoch")

# %% [markdown]
# #### evaluating the model

# %%
# Evaluate the model on the test dataset
with torch.no_grad():# basically turning of the backpropogation
  y_predict = model.forward(X_test) # they are the features form our test set
  loss = criterion(y_predict, y_test) #Find the loss or error

# %%
loss

# %%
correct = 0
print("TENSORS                                          PREDICTED         ACTUAL")
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)
    if y_test[i]==0:
      x="Setosa"
    elif y_test[i]==1:
      x="Versicolor"
    else:
      x="Virginica"
    if y_val.argmax().item()==0:
      y="Setosa"
    elif y_val.argmax().item()==1:
      y="Versicolor"
    else:
      y="Virginica"
    # will tell us what type of flower class oir
    print(f"{i+1}.) {str(y_val)} \t {x} \t {y}")
    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct+=1
print(f"\n we got correct:{correct}")
# Higher number indicates that which flower will be choosesn

# %% [markdown]
# ### feeding data to the model

# %%
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])


# %%
with torch.no_grad():
    y_val = model.forward(new_iris)
    pred = y_val.argmax().item()

    if pred == 0:
        print("It is a Setosa")
    elif pred == 1:
        print("It is a Versicolor")
    else:
        print("It is a Virginica")

# %% [markdown]
# ### saving and loading our model

# %%
# save our model's weight and biases that we have trained in form of dictionary
torch.save(model.state_dict(), "iris_model.pt")

# %%
# Loading the saved model
new_model = Model()
new_model.load_state_dict(torch.load("iris_model.pt"))

# %%
# Mkaing sure the model is loaded
new_model.eval()

# %%
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])

# %%
with torch.no_grad():
  y_eval = new_model.forward(new_iris)
  pred = y_eval.argmax().item()
  if pred == 0:
    print("It is a Setosa")
  elif pred == 1:
    print("It is a Versicolor")
  else:
    print("It is a Virginica")


