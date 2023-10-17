import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]
        return y_pred, h_2
    

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10

model = MLP(INPUT_DIM, OUTPUT_DIM)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load('../oneDNN/build/tut1-model.pt'))

# enable oneDNN graph fusion globally
torch.jit.enable_onednn_fusion(True)
import os
os.environ["ONEDNN_GRAPH_DUMP"] = "graph"
# os.environ["DNNL_VERBOSE"]="1" 
# os.environ["PYTORCH_JIT_LOG_LEVEL"]=">>graph_helper:>>graph_fuser:>>kernel:>>interface"

rand_inp = torch.rand(1, 1, 28, 28)
# construct the model
with torch.no_grad():
    model.eval()
    fused_model = torch.jit.trace(model, rand_inp)

# print(fused_model)
fused_model(rand_inp)

# run the model
with torch.no_grad():
    # oneDNN graph fusion will be triggered during runtime
    output = fused_model(rand_inp)
print(output)