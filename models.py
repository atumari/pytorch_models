import torch
from torch import nn
from torch.nn import functional as F

class FactorizationMachine(nn.Module):
    def __init__(self, feature_num, latent_num):
        super(FactorizationMachine, self).__init__() 
        self.feature_num = feature_num
        self.latent_num = latent_num

        self.emb_bias = nn.Embedding(feature_num, 1)
        self.emb_factor = nn.Embedding(feature_num, latent_num)
   
    def forward(self, x):
        bias = torch.sum(self.emb_bias(x), (-1,-2)).view(-1, 1)
        factor = self.emb_factor(x)
        diff = torch.sum(factor, 1, True) - factor
        dot = torch.sum((factor * diff), -1)
        out = bias + torch.sum(dot, -1, keepdim=True)
        return out


class NeuralFactorizationMachine(nn.Module):
    def __init__(self, feature_num, field_num, latent_num, hidden_layers):
        super(NeuralFactorizationMachine, self).__init__() 
        self.feature_num = feature_num
        self.latent_num = latent_num
        self.field_num = field_num
        self.hidden_layers = [field_num] + hidden_layers

        self.emb_bias = nn.Embedding(feature_num, 1)
        self.emb_factor = nn.Embedding(feature_num, latent_num)
        layers_list = []
        for i in range(len(hidden_layers)):
            layers_list.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            layers_list.append(nn.BatchNorm1d(self.hidden_layers[i+1]))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(self.hidden_layers[-1], 1))
        self.dense_part = nn.Sequential(*layers_list)
   
    def forward(self, x):
        bias = torch.sum(self.emb_bias(x), (-1,-2)).view(-1, 1)
        factor = self.emb_factor(x)
        diff = torch.sum(factor, 1, True) - factor
        dot = torch.sum((factor * diff), -1)
        out = bias + self.dense_part(dot)
        return out


class Embedding3d(nn.Module):
    def __init__(self, feature_num, field_num, latent_num):
        super(Embedding3d, self).__init__()
        self.weights = nn.Parameter(torch.rand((feature_num, field_num, latent_num), dtype=torch.float32))
        self.feature_num = feature_num
        self.field_num = field_num
        self.latent_num = latent_num

    def forward(self, x):
        return torch.index_select(self.weights, 0, x.flatten()).view(-1, self.field_num, self.field_num, self.latent_num)


class FieldFactorizationMachine(nn.Module):
    def __init__(self, feature_num, field_num, latent_num):
        super(FieldFactorizationMachine, self).__init__()
        self.feature_num = feature_num
        self.field_num = field_num
        self.latent_num = latent_num
        self.factor_size = (field_num * (field_num - 1)) // 2

        self.emb_bias = nn.Embedding(feature_num, 1)
        self.emb_factor = Embedding3d(feature_num, field_num, latent_num)
        
    def forward(self, x):
        bias = torch.sum(self.emb_bias(x), (-1,-2)).view(-1, 1)
        
        factor = self.emb_factor(x)

        second = factor * factor.permute(0, 2, 1, 3)
        second = torch.sum(second, -1)
        mask = torch.triu(torch.ones_like(second[0], dtype=torch.uint8), diagonal=1)
        second = torch.masked_select(second, mask).view(-1, self.factor_size)
        out = bias + torch.sum(second, -1, keepdim=True)
        return out


class NeuralFieldFactorizationMachine(nn.Module):
    def __init__(self, feature_num, field_num, latent_num, hidden_layers):
        super(NeuralFieldFactorizationMachine, self).__init__()
        self.feature_num = feature_num
        self.field_num = field_num
        self.latent_num = latent_num
        self.factor_size = (field_num * (field_num - 1)) // 2
        self.hidden_layers = [self.factor_size] + hidden_layers

        self.emb_bias = nn.Embedding(feature_num, 1)
        self.emb_factor = Embedding3d(feature_num, field_num, latent_num)

        layers_list = []
        for i in range(len(hidden_layers)):
            layers_list.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            layers_list.append(nn.BatchNorm1d(self.hidden_layers[i+1]))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(self.hidden_layers[-1], 1))
        self.dense_part = nn.Sequential(*layers_list)
        
    def forward(self, x):
        bias = torch.sum(self.emb_bias(x), (-1,-2)).view(-1, 1)

        factor = self.emb_factor(x)

        second = factor * factor.permute(0, 2, 1, 3)
        second = torch.sum(second, -1)
        mask = torch.triu(torch.ones_like(second[0], dtype=torch.uint8), diagonal=1)
        second = torch.masked_select(second, mask).view(-1, self.factor_size)
        out = bias + self.dense_part(second)
        return out


class xDeepFM(nn.Module):
    def __init__(self, feature_num, field_num, latent_num, cin_layers, cin_res_layers, dnn_layers):
        super(xDeepFM, self).__init__()
        
        self.feature_num = feature_num
        self.field_num = field_num
        self.latent_num = latent_num

        self.dnn_size = field_num * latent_num

        self.cin_layers = [field_num] + cin_layers

        self.emb_bias = nn.Embedding(feature_num, 1)
        self.emb = nn.Embedding(feature_num, latent_num)

        conv_list = []
        cin_len = 0
        for i in range(1, len(self.cin_layers)):
            conv_list.append(torch.nn.Conv1d(self.cin_layers[i-1]*field_num, self.cin_layers[i], 1, bias=False))
            cin_len += self.cin_layers[i]

        self.conv_list = nn.ModuleList(conv_list)


        self.cin_res_layers = [cin_len] + cin_res_layers
        cin_res_layers_list = []
        for i in range(len(cin_res_layers)):
            cin_res_layers_list.append(nn.Linear(self.cin_res_layers[i], self.cin_res_layers[i+1]))
            cin_res_layers_list.append(nn.BatchNorm1d(self.cin_res_layers[i+1]))
            cin_res_layers_list.append(nn.ReLU())

        self.cin_res_part = nn.Sequential(*cin_res_layers_list)

        self.cin_dense = nn.Linear(cin_len + cin_res_layers[-1], 1)

        self.dense_size = cin_len + self.field_num + 256

        self.dnn_layers = [self.dnn_size] + dnn_layers
        dnn_layers_list = []
        for i in range(len(dnn_layers)):
            dnn_layers_list.append(nn.Linear(self.dnn_layers[i], self.dnn_layers[i+1]))
            dnn_layers_list.append(nn.BatchNorm1d(self.dnn_layers[i+1]))
            dnn_layers_list.append(nn.ReLU())
        dnn_layers_list.append(nn.Linear(self.dnn_layers[-1], 1))
        self.dnn_part = nn.Sequential(*dnn_layers_list)
  
    def forward(self, x):

        bias = self.emb_bias(x).view(-1, self.field_num)
        factor = self.emb(x)

        # linear part
        linear_result = torch.sum(bias, -1, keepdim=True)

        # cin part
        final_result = []
        hidden_layers = []
        
        X0 = torch.transpose(factor, 1, 2) 
        hidden_layers.append(X0)
        X0 = X0.unsqueeze(-1)
        for idx, layer_size in enumerate(self.cin_layers[:-1]):
            Xk_1 = hidden_layers[-1].unsqueeze(2)
            out_product = torch.matmul(X0,Xk_1)#[N,K,H0,H_(k-1)]
            out_product = out_product.view(-1, self.latent_num, layer_size*self.field_num)
            out_product = out_product.transpose(1,2)#[N,H0XH_(k-1),K]
            zk = self.conv_list[idx](out_product)#[N,Hk,K]
            next_hidden = zk.transpose(1, 2)
            final_result.append(zk)
            hidden_layers.append(next_hidden)
            
        cin_result = torch.cat(final_result,1) #[N,H1+...+Hk,K]
        cin_result = torch.sum(cin_result,-1) #[N,H1+...+Hk]
        cin_res = self.cin_res_part(cin_result)
        cin_res = F.relu(cin_res)
        cin_result = torch.cat([cin_result, cin_res], -1)
        cin_result = self.cin_dense(cin_result)

        # dnn part
        dnn_in = factor.view(-1, self.dnn_size)
        dnn_result = self.dnn_part(dnn_in)

        out = linear_result + cin_result + dnn_result

        return out



def sub_func(x):
    if x < 4:
        return 1
    elif x < 10:
        return 2
    elif x < 30:
        return 3
    else:
        return 4

class EntityEmbedding(nn.Module):
    def __init__(self, unique_list, layers, drop=True, latent_list=None):
        # unique_listは列ごとのuniqueな特徴の個数

        super(EntityEmbedding, self).__init__()

        if latent_list:
            self.emb_list = nn.ModuleList(
                [nn.Embedding(unique_list[i], latent_list[i]) for i in range(len(unique_list))]
            )
            self.size = sum(latent_list)

        else:
            self.emb_list = nn.ModuleList(
                [nn.Embedding(unique_list[i], sub_func(unique_list[i])) for i in range(len(unique_list))]
            )
            self.size = sum(list(map(sub_func, unique_list)))

        self.layers = [self.size] + layers
        layers_list = []
        for i in range(len(layers)):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i+1]))
            layers_list.append(nn.BatchNorm1d(self.layers[i+1]))
            layers_list.append(nn.ReLU())
            if drop:
               layers_list.append(nn.Dropout2d(p=0.2, inplace=False))
        layers_list.append(nn.Linear(self.layers[-1], 1))
        self.dense_part = nn.Sequential(*layers_list)
        
    def forward(self, x):
        embs = []
        for i in range(len(self.emb_list)):
            embs.append(self.emb_list[i](x[:, i]))
        
        embs = torch.cat(embs, 1)

        x = self.dense_part(embs)
        return x


class DAE(nn.Module):
    def __init__(self, unique_list, layers, drop=True, latent_list=None):
        super(DAE, self).__init__()

        if latent_list:
            self.emb_list = nn.ModuleList(
                [nn.Embedding(unique_list[i], latent_list[i]) for i in range(len(unique_list))]
            )
            self.size = sum(latent_list)

        else:
            self.emb_list = nn.ModuleList(
                [nn.Embedding(unique_list[i], sub_func(unique_list[i])) for i in range(len(unique_list))]
            )
            self.size = sum(list(map(sub_func, unique_list)))
        
        self.layers = [self.size] + layers
        layers_list = []
        for i in range(len(layers)):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i+1]))
            layers_list.append(nn.BatchNorm1d(self.layers[i+1]))
            layers_list.append(nn.ReLU())
            if drop:
               layers_list.append(nn.Dropout2d(p=0.2, inplace=False))

        self.dense_part = nn.Sequential(*layers_list)
        
        self.dec_list = nn.ModuleList(
            [nn.Linear(layers[-1], unique_list[i], bias=False) for i in range(len(unique_list))]
        )
    
    def forward(self, x):
        embs = []
        for i in range(len(self.emb_list)):
            tmp = self.emb_list[i](x[:, i])
            embs.append(tmp)
        
        embs = torch.cat(embs, 1)

        x = self.dense_part(embs)

        decs = []
        for i in range(len(self.dec_list)):
            decs.append(F.log_softmax(self.dec_list[i](x), dim=1))

        return x, decs