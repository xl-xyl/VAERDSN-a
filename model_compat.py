import torch.nn as nn
import torch
from functions import ReverseLayerF
import torch.nn.functional as F


class DSN(nn.Module):
    def __init__(self, field_size,code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################
        self.s_encoder=nn.Sequential()
        self.s_encoder.add_module('sencoder',nn.Linear(300,200))
        self.s_encoder.add_module('srelu',nn.ReLU(True))
        # self.fc1 = nn.Linear(image_size, h_dim)
        self.fc1 = nn.Linear(200,100)
        self.fc2 = nn.Linear(200,100)


        #########################################
        # private target encoder
        #########################################
        self.t_encoder=nn.Sequential()
        self.t_encoder.add_module('sencoder',nn.Linear(300,200))
        self.t_encoder.add_module('srelu',nn.ReLU(True))
        # self.fc1 = nn.Linear(image_size, h_dim)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(200,100)



        ################################
        # shared encoder (dann_mnist)
        ################################
        self.sh_encoder=nn.Sequential()
        self.sh_encoder.add_module('sencoder',nn.Linear(300,200))
        self.sh_encoder.add_module('srelu',nn.ReLU(True))
        # self.fc1 = nn.Linear(image_size, h_dim)
        self.fc5 = nn.Linear(200,100)
        self.fc6 = nn.Linear(200,100)





        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=field_size*100, out_features=300))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=300, out_features=2))
        self.shared_encoder_pred_class.add_module('d_softmax',nn.LogSoftmax(dim=1))

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################
        self.sh_decoder=nn.Sequential()
        self.sh_decoder.add_module('shdecoder',nn.Linear(in_features=100, out_features=200))
        self.sh_decoder.add_module('shdecoder2',nn.ReLU(True))
        self.sh_decoder.add_module('shdecoder3',nn.Linear(in_features=200, out_features=300))
        # self.sh_decoder.add_module('shdecoder4',nn.Sigmoid())

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_data, mode, rec_scheme,field_size,p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.s_encoder(input_data)
            mu1,var1=self.fc1(private_feat),self.fc2(private_feat)
            private_code = self.reparameterize(mu1, var1)

        elif mode == 'target':

            # target private encoder
            private_feat = self.t_encoder(input_data)
            mu1,var1=self.fc3(private_feat),self.fc4(private_feat)
            private_code = self.reparameterize(mu1,var1)

        result.append(private_code)

        # shared encoder
        shared_feat = self.sh_encoder(input_data)
        mu2,var2=self.fc5(shared_feat),self.fc6(shared_feat)
        shared_code = self.reparameterize(mu2, var2)
        result.append(shared_code)
        # print(shared_code)
        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        # print(domain_label.shape)
        result.append(domain_label)

        if mode == 'source':
            j=0
            aa=torch.zeros(1,field_size*100).to('cuda')
            ee=torch.zeros(1,field_size*100).to('cuda')
            h=torch.zeros(100).to('cuda')
            for i in shared_code:
                if(j!=field_size):
                    h=torch.cat((h,i),dim=0)
                    j=j+1
                    if(j==field_size):
                        c = torch.unsqueeze(h[100:], 0).to('cuda')
                        aa=torch.cat((aa,c),dim=0)
                        h=torch.zeros(100).to('cuda')
                        j=0
            # print(aa[1:].shape)
            a=0.3
            b=0.5
            d=0.1
            e=0.1
            for ii,jj in enumerate(aa[1:]):
                cc=torch.zeros(100).to('cuda')
                cc=torch.cat((cc,a*jj[0:100]),dim=0)
                cc=torch.cat((cc,b*jj[100:200]),dim=0)
                cc=torch.cat((cc,d*jj[200:300]),dim=0)
                cc=torch.cat((cc,e*jj[300:400]),dim=0)
                cc=torch.cat((cc,a*jj[400:500]),dim=0)
                cc=torch.cat((cc,b*jj[500:600]),dim=0)
                cc=torch.cat((cc,d*jj[600:700]),dim=0)
                cc=torch.cat((cc,e*jj[700:800]),dim=0)
                cc = torch.unsqueeze(cc[100:], 0).to('cuda')              #增加一维
                ee=torch.cat((ee,cc),dim=0)
            # print(ee[1:].shape)
            class_label = self.shared_encoder_pred_class(ee[1:])
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        # print(union_code.shape)
        rec_code = self.sh_decoder(union_code)

        result.append(rec_code)
        result.append(mu1)
        result.append(var1)
        result.append(mu2)
        result.append(var2)

        return result



