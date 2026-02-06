# SeqHAR model by A.Baraka

import torch
import torch.nn as nn
import torch.nn.functional as F


w_feat=512
hidden_size=128

class PreModule(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(PreModule, self).__init__()
        input_size=input_shape[2]
        window_size=input_shape[1]  
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)     
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(w_feat, hidden_size)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)      
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CurModule(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CurModule, self).__init__()
        input_size=input_shape[2]  
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten=nn.Flatten() 
        self.fc1 = nn.Linear(w_feat, hidden_size)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x,x1

    
class NxtModule(nn.Module):
    def __init__(self, num_classes):
        super(NxtModule, self).__init__()
        hidden_size=num_classes*2
        self.fc1 = nn.Linear(num_classes, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, y_prev):
        x=self.fc1(y_prev)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SeqHAR(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SeqHAR, self).__init__()
        self.pre_module = PreModule(input_shape,  num_classes)
        self.cur_module = CurModule(input_shape,  num_classes)
        self.nxt_module = NxtModule(num_classes)
        hidden_size=w_feat
        # Cross-Attention for NxtModule output
        self.query_cur = nn.Linear(w_feat, w_feat) # Query from LSTM output
        self.key_cur = nn.Linear(num_classes, w_feat)   # Key from NxtModule output
        self.value_cur = nn.Linear(num_classes, w_feat) # Value from NxtModule output
        
        self.query_pre = nn.Linear(w_feat, w_feat) # Query from LSTM output
        self.key_pre = nn.Linear(num_classes, w_feat)   # Key from NxtModule output
        self.value_pre = nn.Linear(num_classes, w_feat) # Value from NxtModule output

        self.softmax = nn.Softmax(dim=-1)
        self.d_k = torch.tensor(w_feat**0.5) # for scaled dot-product attention
        # Final FC layer after concatenating LSTM output and attention outputs
        self.norm = nn.LayerNorm(w_feat)
        self.fc = nn.Linear(w_feat, num_classes)
        
    def forward(self, x_pre, x_cur):
        pre_output = self.pre_module(x_pre)
        cur_feat,cur_output = self.cur_module(x_cur) 
        next_activity_probs = self.nxt_module(pre_output)
        
        # Attention for current
        q_cur = self.query_cur(cur_feat).unsqueeze(1)
        k_cur = self.key_cur(cur_output).unsqueeze(1)
        v_cur = self.value_cur(cur_output).unsqueeze(1)

        scores_cur = torch.bmm(q_cur, k_cur.transpose(1, 2)) / self.d_k
        attention_weights_cur = self.softmax(scores_cur)
        attended_cur_output = torch.bmm(attention_weights_cur, v_cur).squeeze(1)

        # Attention for pre_output
        q_pre = self.query_pre(cur_feat).unsqueeze(1)
        k_pre = self.key_pre(next_activity_probs).unsqueeze(1)
        v_pre = self.value_pre(next_activity_probs).unsqueeze(1)

        scores_pre = torch.bmm(q_pre, k_pre.transpose(1, 2)) / self.d_k
        attention_weights_pre = self.softmax(scores_pre)
        attended_pre_output = torch.bmm(attention_weights_pre, v_pre).squeeze(1)

        # Concatenate original LSTM output with attention outputs
        result=cur_feat*attended_pre_output*attended_cur_output
        result=self.norm(result)
        result = self.fc(result)    
        return pre_output, cur_output, next_activity_probs, result
    