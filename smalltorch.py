import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def process_numerical(col_name, advanced_mean=True):
    global selected_data
    if advanced_mean:
        means = {x: selected_data.query('AGE >= %1d and AGE <= %2d and %3s > 0' % (x - 1, x + 1, col_name))[col_name].mean() for x in selected_data['AGE'].unique()}
        print(col_name)
        print(means)
        for i, row in selected_data.iterrows():
            age = row['AGE']
            selected_data.loc[i, col_name] = means[age]
    else:
        mean = selected_data.query('%s > 0' % col_name)[col_name].mean()
        selected_data.loc[selected_data[col_name] < 0, col_name] = mean

# let's grab our data
data14 = pd.read_csv('ed14-known.csv')
data15 = pd.read_csv('ed15-known.csv')
data16 = pd.read_csv('ed16-known.csv')
numerical_cols = ['POPCT', 'BPSYS', 'PULSE', 'BPDIAS', 'RESPR', 'AGE', 'PAINSCALE', 'TEMPF']
categorical_cols = ['SEX', 'ASTHMA', 'COPD']
extras = ['IMMEDR', 'RFV13D', 'RFV23D', 'RFV33D']
selected14 = data14[numerical_cols + categorical_cols + extras]
selected15 = data15[numerical_cols + categorical_cols + extras]
selected16 = data16[numerical_cols + categorical_cols + extras]

selected_data = pd.concat([selected14, selected15, selected16])

# pre-processing
selected_data = selected_data.query('IMMEDR > 0 and IMMEDR <= 5')
# binary
selected_data.loc[:, 'IMMEDR'] = selected_data['IMMEDR'].apply(lambda x: 2 if x <= 2 else 1 )
selected_data.loc[:,'IMMEDR'] = selected_data['IMMEDR'].apply(lambda x: x-1)

# undersampling
sd_sizes = []
for i in range(2):
    sd_sizes.append(selected_data.query('IMMEDR == %1d' % i).shape[0])
smallest = min(sd_sizes)
print('Class size:', smallest)
sd_chunks = []
for i in range(2):
    sd_chunks.append(selected_data.query('IMMEDR == %1d' % i).sample(frac = smallest/sd_sizes[i]))
selected_data = pd.concat([chunk for chunk in sd_chunks])
selected_data = selected_data.sample(frac=1)

# guess unknowns and normalize
for col in numerical_cols:
    if col == 'AGE':
        process_numerical(col, False)
    else:
        process_numerical(col, True)
    selected_data.loc[:, col] = (selected_data[col] - selected_data[col].mean()) / selected_data[col].std()

# process any categoricals
selected_data.loc[:, 'SEX'] = selected_data['SEX'] - 1

# add RFV's
rfv = (pd.get_dummies(selected_data['RFV13D'], prefix='RFV') | pd.get_dummies(selected_data['RFV23D'], prefix='RFV') | pd.get_dummies(selected_data['RFV33D'], prefix='RFV'))
rfv = rfv.astype('int32')
print(rfv.columns)
rfv = rfv.drop(['RFV_0'], axis='columns')
selected_data = selected_data.drop(['RFV13D', 'RFV23D', 'RFV33D'], axis = 'columns')
selected_data = selected_data.join(rfv)

print(selected_data.sample(10))

train_data = selected_data[:int(0.8 * len(selected_data))]
train_labels = train_data.pop('IMMEDR')
test_data = selected_data[int(0.8 * len(selected_data)):]
test_labels = test_data.pop('IMMEDR')

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        num_features = len(numerical_cols + categorical_cols + list(rfv.columns))
        self.layer1 = nn.Linear(num_features, 200)
        #self.layer2 = nn.Linear(25, 25)
        self.logits = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        x = self.logits(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = MyNet()
net = net.double()
print(net)

batch_size = 10
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)
optimizer = torch.optim.Adam(net.parameters())
for epoch in range(5):
    running_loss = 0.0
    train_data, train_labels = shuffle(train_data, train_labels)
    for i in range(len(train_data) // batch_size):
        np_inputs = train_data.iloc[batch_size * i : batch_size*(i+1)].values
        np_labels = train_labels.iloc[batch_size * i : batch_size * (i+1)].values

        inputs = torch.from_numpy(np_inputs)
        #inputs = torch.randn(batch_size, len(numerical_cols + categorical_cols)).double()
        labels = torch.tensor(np_labels, dtype=torch.long)
        
        net.zero_grad()  
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #if i == 0:
        #    print('Example Loss:')
        #    print(outputs)
        #    print(labels)
        loss.backward()
        optimizer.step()
        
        # print stats
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %3d] loss: %.3f' % (epoch, i, running_loss))
        running_loss = 0.0

print('Finished training')

# test
total = 0
correct = 0
with torch.no_grad():
    # DEBUG
    outputs = net(torch.from_numpy(test_data.iloc[10:20].values))
    prob, predicted = torch.max(outputs.data, 1)
    
    print(test_labels.iloc[10:20].values)
    print(predicted)
    
    # The real stuff
    outputs = net(torch.from_numpy(test_data.values))
    prob, predicted = torch.max(outputs.data, 1)
    
    total += predicted.size(0)
    correct += (predicted == torch.from_numpy(test_labels.values)).sum().item()
    

print('Final Accuracy: %.2f' % (correct / total))
