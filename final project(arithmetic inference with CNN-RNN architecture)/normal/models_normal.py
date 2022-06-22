import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def _make_layer(input_dim, output_dim, device):
        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, (3, 3), stride=1, padding=1, padding_mode='zeros', bias=False).to(device))
        layers.append(torch.nn.BatchNorm2d(output_dim).to(device))
        return nn.Sequential(*layers)

class CustomCNN(nn.Module):
    
    def __init__(self, input_dim, hidden_size, output_dim):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-1: define cnn model        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        self.layer1 = _make_layer(1, hidden_size, device)
        self.layer2 = _make_layer(hidden_size, hidden_size, device)
        self.layer3 = _make_layer(hidden_size, hidden_size, device)
        self.layer4 = _make_layer(hidden_size, hidden_size, device)
        
        self.layer5 = _make_layer(hidden_size, 2 * hidden_size, device)
        self.layer6 = _make_layer(2 * hidden_size, 2 * hidden_size, device)
        self.layer7 = _make_layer(2 * hidden_size, 2 * hidden_size, device)
        self.layer8 = _make_layer(2 * hidden_size, 2 * hidden_size, device)
        
        self.relu = nn.ReLU().to(device)
        self.max_pool = nn.MaxPool2d((2, 2), 2).to(device)
        self.flatten = nn.Flatten(start_dim=1, end_dim=- 1).to(device)
        self.linear = nn.Linear(2 * hidden_size * 7 * 7, output_dim, bias=True).to(device)
        self.bn = torch.nn.BatchNorm1d(output_dim).to(device)
              
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: Batch size X (Sequence_length, Channel=1, Height, Width)
        outputs: (Sequence_length X Batch_size, Hidden_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-2: code CNN forward path
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        
        # input B T C H W -> B T 1 28 28 
        outputs = self.layer1(inputs)
        
        identity = outputs
        
        outputs = self.relu(outputs) # B T hidden_size 28 28
        outputs = self.layer2(outputs)
        outputs = self.relu(outputs) # B T hidden_size 28 28
        outputs = self.layer3(outputs)
        outputs = self.relu(outputs) # B T hidden_size 28 28
        outputs = self.layer4(outputs)
        
        outputs += identity
        
        outputs = self.relu(outputs) # B T hidden_size 28 28        
        outputs = self.max_pool(outputs) # B T hidden_size 14 14
        
        ###
        
        outputs = self.layer5(outputs)
        
        identity = outputs
        
        outputs = self.relu(outputs) # B T hidden_size 28 28
        outputs = self.layer6(outputs)
        outputs = self.relu(outputs) # B T hidden_size 28 28
        outputs = self.layer7(outputs)
        outputs = self.relu(outputs) # B T hidden_size 28 28
        outputs = self.layer8(outputs)
        
        outputs += identity
        
        outputs = self.relu(outputs) # B T hidden_size 28 28  
        outputs = self.max_pool(outputs) # B T hidden_size 14 14
        
        outputs = self.flatten(outputs) # B T hidden_size*7*7 
        outputs = self.linear(outputs) # B T output_dim
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super(GRU, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-1: Define lstm and input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        # you can either use torch LSTM or manually define it
        
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=False, bidirectional=True).to(self.device)
        
        self.fc_in = hidden_size * 2
        self.fc_out = vocab_size
        self.linear = nn.Linear(self.fc_in, self.fc_out, bias=True).to(self.device)
        self.softmax = nn.Softmax(dim=-1).to(self.device)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, feature, h):
        """
        For reference (shape example)
        feature: (Sequence_length, Batch_size, Input_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-2: Design LSTM model for letter sorting
        # NOTE: sequence length of feature can be various        
        
        output, h_next = self.gru(feature, h)
        output = self.linear(output)
        output = self.softmax(output)  
        
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        
        # (sequence_lenth, batch, num_classes), (num_rnn_layers, batch, hidden_dim), (num_rnn_layers, batch, hidden_dim)
        return output, h_next


class ConvGRU(nn.Module):
    def __init__(self, sequence_length=5, num_classes=26, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=256,
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1, rnn_dropout=0.0):
        # NOTE: you can freely add hyperparameters argument
        super(ConvGRU, self).__init__()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv = CustomCNN(cnn_input_dim, cnn_hidden_size, rnn_input_dim)
        self.gru1 = GRU(rnn_input_dim, rnn_hidden_size, num_classes, rnn_num_layers, rnn_dropout)
        self.gru2 = GRU(num_classes, rnn_hidden_size, num_classes, rnn_num_layers, rnn_dropout)
        
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        input is (imgaes, labels) (training phase) or images (test phase)
        images: sequential features of Batch size X (Sequence_length, Channel=1, Height, Width)
        labels: Batch size X (Sequence_length)
        outputs should be a size of Batch size X (1, Num_classes) or Batch size X (Sequence_length, Num_classes)
        """

        # for teacher-forcing
        have_labels = False
        if len(inputs) == 2:
            have_labels = True
            images, labels = inputs
        else:
            images = inputs

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem3: input image into CNN and RNN sequentially.
        # NOTE: you can use teacher-forcing using labels or not
        # NOTE: you can modify below hint code 
        
        batch_size = len(images)
        hidden_state = torch.zeros(self.rnn_num_layers * 2, self.rnn_hidden_size, requires_grad=True).to(self.device)
        
        mask = torch.zeros(batch_size).type(torch.long)
        for i in range(batch_size):
            mask[i] = images[i].shape[0] # img T_max 
            
        images = torch.nn.utils.rnn.pad_sequence(images, batch_first = True, padding_value=0).to(self.device) # B T 1 28 28
        
        max_sequence_length = images.shape[1]
        images = images.reshape(-1, 1, 28, 28) # B*T 1 28 28
        
        features = self.conv(images) # B*T rnn_input_dim
        features = features.reshape(batch_size, max_sequence_length, -1) # B T rnn_input_dim
        features = torch.transpose(features, 0, 1).to(self.device) # T B rnn_input_dim      

        if have_labels:
            # training code ...
            # teacher forcing by concatenating ()
            output1 = []
            output2 = []
            for i in range(batch_size):
                y = torch.zeros(mask[i], self.num_classes).to(self.device)
                for j in range(mask[i]):
                    y[j][labels[i][j]] = 1
                    # one_hot = nn.functional.one_hot(torch.LongTensor(labels[i][j]), num_classes = 26)  
                    # soft_label = one_hot * (1 - 0.1) + (0.1 / 26)
                    # y[j] = soft_label

                out1, h_next = self.gru1(features[range(mask[i]), i, :], hidden_state) # T output_dim
                out2, h_next = self.gru2(y, hidden_state) # T output_dim / T wordsize
                output1.append(out1)  
                output2.append(out2)  
            outputs = output1, output2
        else:
            # evaluation code ...
            outputs = torch.zeros(batch_size, self.num_classes).to(self.device)

            for i in range(batch_size):
                out1, h_next = self.gru1(features[range(mask[i]), i, :], hidden_state) # T output_dim
                for j in range(mask[i]):
                    out1[j] = nn.functional.one_hot(torch.argmax(out1[j]), self.num_classes)
                    #out1[j] = out1[j] * (1 - 0.1) + (0.1 / 26)
                out2, h_next = self.gru2(out1, hidden_state) # T output_dim                
                outputs[i] = out2[mask[i] - 1]
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

