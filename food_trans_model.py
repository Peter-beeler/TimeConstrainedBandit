import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torchmetrics import AUROC
import matplotlib.pyplot as plt

epsilon = 1e-12
SPLIT_BY_IDs = False
batch_size = 1
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
 
        return x

# Define the combined model with two MLPs
class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.mlp1 = MLP(input_dim, hidden_dim, 2)
        self.mlp2 = MLP(input_dim, hidden_dim, 2)
        self.m = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        p1 = self.mlp1(x)
        p1 = self.m(p1)
        p2 = self.mlp2(x)
        p2 = self.m(p2)
        return torch.cat((p1,p2), 0)

class LikelihoodLoss(nn.Module):
    def __init__(self):
        super(LikelihoodLoss, self).__init__()

    def forward(self, predictions, targets):
        pred1, pred2 = predictions
        target1, target2 = targets
        loss1 = torch.mean((pred1 - target1)**2)
        loss2 = torch.mean((pred2 - target2)**2)
        return loss1 + loss2

def my_loss(output, Y_last, Y, T):
    p = output

    if Y_last == 0:
        prev = torch.tensor([[0.99 , 0.01]])
    else:
        prev = torch.tensor([[0.01 , 0.99] ])
    # prev = prev.to(device)
    loss =  torch.mm(prev , torch.linalg.matrix_power(p, T.int().item()))

    loss =  - torch.log(loss + epsilon)

    return (1-Y)*loss[0][0] + Y * loss[0][1]
    # return loss

def prob_pred(output, Y_last, Y, T):
    p = output
    # print(p)
    if Y_last == 0:
        prev = torch.tensor([[0.99 , 0.01]])
    else:
        prev = torch.tensor([[0.01 , 0.99] ])
    # print(p)
    loss =  torch.mm(prev , torch.linalg.matrix_power(p, T.int().item()))
   

    return loss[0][0]

def prepare_data(df):
    xg_df = pd.read_csv("./Chicago_data/real_raw_data.csv", header=0)
    df['Facility_Type'] = pd.factorize(df['Facility_Type'])[0]
    df['Inspector_Assigned'] = pd.factorize(df['Inspector_Assigned'])[0]
    df = df.apply(pd.to_numeric)
    df = df.drop(columns = ["Unnamed: 0"])
    
    # X = df.drop(columns = ["License", "criticalFound", "last_check", "timeSinceLast"]).values
    xgboost_score = []
    for index, row in df.iterrows():
        ins_id = row['Inspection_ID']
        score = xg_df[xg_df['Inspection_ID'] == ins_id]['score']
        xgboost_score.append(score)
    
    if SPLIT_BY_IDs:
        df["xg_score"] = xgboost_score
        unique_ids = df['License'].unique()
        np.random.seed(42) 
        test_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * 0.2), replace=False)
        train_ids = np.setdiff1d(unique_ids, test_ids)
        train_df = df[df['License'].isin(train_ids)]
        test_df = df[df['License'].isin(test_ids)]
        Y_train, Y_last_train, T_train, train_scores = train_df["criticalFound"].values, train_df["last_check"].values, train_df["timeSinceLast"].values, train_df["xg_score"].values 
        X_train = train_df.drop( columns = ["License", "criticalFound", "last_check", "timeSinceLast", "Inspection_ID", "xg_score"]).values
        Y_test, Y_last_test, T_test, test_score = test_df["criticalFound"].values, test_df["last_check"].values, test_df["timeSinceLast"].values, test_df["xg_score"].values 
        X_test = test_df.drop( columns = ["License", "criticalFound", "last_check", "timeSinceLast", "Inspection_ID", "xg_score"]).values
        return X_train, X_test, Y_train, Y_test, Y_last_train, Y_last_test, T_train, T_test, train_scores, test_score
    

    else:
        Y = df["criticalFound"].values
        Y_last = df["last_check"].values
        T = df["timeSinceLast"].values 
        X = df.drop(columns = ["License", "criticalFound", "last_check", "timeSinceLast", "Inspection_ID"]).values
        X_train, X_test, Y_train, Y_test, Y_last_train, Y_last_test, T_train, T_test = train_test_split(X, Y, Y_last, T, test_size=0.2, random_state=42)
        train_scores, test_score = train_test_split(xgboost_score, test_size=0.2, random_state=42)
        return X_train, X_test, Y_train, Y_test, Y_last_train, Y_last_test, T_train, T_test, train_scores, test_score



df = pd.read_csv("./Chicago_data/transition_data_all.csv", header=0)
X_train, X_test, Y_train, Y_test, Y_last_train, Y_last_test, T_train, T_test, train_scores, test_score = prepare_data(df)


# loader data prep
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
Y_last_train_tensor = torch.tensor(Y_last_train, dtype=torch.float32).view(-1, 1)
Y_last_test_tensor = torch.tensor(Y_last_test, dtype=torch.float32).view(-1, 1)
T_train_tensor = torch.tensor(T_train, dtype=torch.float32).view(-1, 1)
T_test_tensor = torch.tensor(T_test, dtype=torch.float32).view(-1, 1)
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)
# Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(device)
# Y_last_train_tensor = torch.tensor(Y_last_train, dtype=torch.float32).view(-1, 1).to(device)
# Y_last_test_tensor = torch.tensor(Y_last_test, dtype=torch.float32).view(-1, 1).to(device)
# T_train_tensor = torch.tensor(T_train, dtype=torch.float32).view(-1, 1).to(device)
# T_test_tensor = torch.tensor(T_test, dtype=torch.float32).view(-1, 1).to(device)
# Create a dataset and data loader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, Y_last_train_tensor, T_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor, Y_last_test_tensor, T_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 1

model = CombinedModel(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10000
best_model_path = ("split_by_licenses" if SPLIT_BY_IDs else "split_by_records") + '_time0_model.pth'

# model.load_state_dict(torch.load(best_model_path))
def train():
    best_loss = float('inf')
    # print(torch.cuda.get_device_name(0))
    model.train()
    # model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (X_batch, y_batch, y_last_batch, T_batch) in enumerate(train_dataloader):
            # Forward pass
            # X_batch = X_batch.to(device)
            # y_batch = y_batch.to(device)
            # y_last_batch = y_last_batch.to(device)
            # T_batch = T_batch.to(device)
            outputs = model(X_batch)
            loss = my_loss(outputs, y_last_batch, y_batch, T_batch)
            # print(X_batch, outputs, y_last_batch, y_batch, T_batch, loss)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # for param in model.parameters():
            #     print(param.grad)
            
            optimizer.step()
            
            
            running_loss += loss.item()
            # Log batch loss
            # if i % 100 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                
            # break
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (X_batch, y_batch, y_last_batch, T_batch) in enumerate(test_dataloader):
                outputs = model(X_batch)
                loss = my_loss(outputs, y_last_batch, y_batch, T_batch)
                val_loss += loss.item()

        val_loss /= len(test_dataloader)

        # Log average loss for the epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model with Validation Loss: {val_loss:.4f}')
        if epoch % 10 == 0:
            auc()

def eval():
    model = CombinedModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(best_model_path))
    model.eval() 
    print(X_test_tensor.shape)
    rel = []
    with torch.no_grad():  # Disable gradient calculation for inference
    
        for i, (X_batch, y_batch, y_last_batch, T_batch) in enumerate(test_dataloader):
            tmp = []
            tmp.append(model(X_batch).detach().numpy())
            tmp.append(np.array([[1,0], [1, 0]]))
            rel.append(tmp)
    rel = np.array(rel)
    print(rel.shape)

    file = "./trans_real.npy"
    with open(file, "wb") as fp:
        np.save(fp, rel)

def auc():
    print("AUC")
    model = CombinedModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(best_model_path))
    model.eval() 
    preds = []
    labels = []
    with torch.no_grad():  # Disable gradient calculation for inference
    
        for i, (X_batch, y_batch, y_last_batch, T_batch) in enumerate(test_dataloader):
            outputs = model(X_batch)
            prob = prob_pred(outputs, y_last_batch, y_batch, T_batch)
            preds.append(prob.item())
            labels.append(1 - y_batch.item())
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    # for i in range(len(preds)):
    #     print(preds[i], labels[i])
    # # print(preds)
    # # print(labels)
    auroc = AUROC(task="binary")
    auc = auroc(preds, labels)
    N = len(preds)
    incorrect_cnt = 0
    for i in range(N):
        if round(preds[i].item()) != round(labels[i].item()):
            incorrect_cnt += 1
    # print(incorrect_cnt)
    print("AUC:", auc)

# print histgrams of model parameters
def hist():
    print("Hist")
    model = CombinedModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(best_model_path))
    model.eval() 
    preds = []
    labels = []
    matrices = []
    with torch.no_grad():  # Disable gradient calculation for inference
    
        for i, (X_batch, y_batch, y_last_batch, T_batch) in enumerate(test_dataloader):
            outputs = model(X_batch)
            matrices.append(outputs)
        
    # Step 1: Extract each element across all matrices
    elements = {
        'Good to Good': [matrix[0, 0] for matrix in matrices],
        'Good to Bad': [matrix[0, 1] for matrix in matrices],
        'Bad to Good': [matrix[1, 0] for matrix in matrices],
        'Bad to Bad': [matrix[1, 1] for matrix in matrices]
    }

    # Step 2: Plot the histograms
    plt.figure(figsize=(20, 16))
    for i, (title, values) in enumerate(elements.items(), start=1):
        plt.subplot(2, 2, i)  # Create a 2x2 grid of subplots
        plt.hist(values, bins=50, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("hist_probs.png")


train()
auc()
# input_names = ["Features"]
# output_names = ["Transitions"]
# print(X_train_tensor.shape)
# x = torch.randn(batch_size, 32, requires_grad=True)
# # torch.onnx.export(model, x, "model.onnx", input_names=input_names, output_names=output_names)
# test_score = np.array(test_score)  
# print(test_score)
# file = "./xg_scores_1801.npy"
# with open(file, "wb") as fp:
#     np.save(fp, test_score)
    