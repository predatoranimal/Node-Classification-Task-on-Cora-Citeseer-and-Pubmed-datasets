import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
dataset = Planetoid(root='C:/torch_geometric_datasets/',name='Pubmed')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data)
_,pred2 = model(data).max(dim=1)
pred = pred.detach()
correct = int(pred2[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
f1 = f1_score(pred2[data.test_mask], data.y[data.test_mask], average='macro')
print('Accuracy:{:.4f}'.format(acc))
print('F1_score:{:.4f}'.format(f1))

y_test = label_binarize(data.y[data.test_mask],classes=[0,1,2])
y_score = pred[data.test_mask]
n_classes = y_test.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])
fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(),y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])
plt.figure()
lw=2
color = ['green','yellow','blue','orange','red','darkorange','navy']
for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],color=color[i],lw=lw,label='ROC curve of single class(area=%0.2f)'%roc_auc[i])
plt.plot(fpr['micro'],tpr['micro'],label='micro-average ROC curve(area={0:0.2f})'.format(roc_auc['micro']),color='black',linestyle=':',linewidth=4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.legend(loc='lower right')
plt.show()