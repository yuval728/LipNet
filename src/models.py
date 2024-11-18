import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Reshape input: (batch_size, seq_length, ...)
        if self.batch_first:
            x_reshape = x.contiguous().view(-1, *x.size()[2:])  # (batch_size * seq_length, ...)
        else:
            x_reshape = x.contiguous().view(-1, *x.size()[1:])  # (seq_length * batch_size, ...)

        y = self.module(x_reshape)

        # Reshape output back to (batch_size, seq_length, ...)
        if self.batch_first:
            return y.contiguous().view(x.size(0), -1, y.size(-1))  # (batch_size, seq_length, output_size)
        else:
            return y.view(-1, x.size(1), y.size(-1))  # (seq_length, batch_size, output_size)
        
        
class LipNet(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, dropout=0.4, input_channels=3):
        super().__init__()
                
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3,5,5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            # nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,5,5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3,3,3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(True),
            # nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
        )
        
        self.rnn1 = nn.GRU(input_size= 96 * 3 * 6 , hidden_size=hidden_size,
                             num_layers=1, batch_first=False, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.rnn2 = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size,
                             num_layers=1, batch_first=False, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, vocab_size+1)
        
        self.initialize_weights()
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)
        # print(x.shape)
        
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        
        x, _ = self.rnn2(x)
        x = self.dropout2(x)
        
        x = self.fc(x)
        x = x.permute(1, 0, 2).contiguous()
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        print('Model weights initialized.')
        
        
if __name__ == '__main__':
    import utils
    import constants
    
    word2idx, idx2word = utils.get_word2idx_idx2word(constants.vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    temp_input = torch.randn(4, 75, 1, 46, 140).permute(0, 2, 1, 3, 4).to(device) # (B, C, T, H, W)
    
    model = LipNet(vocab_size=len(word2idx), input_size=75).to(device)
    
    output = model(temp_input)
    print(output.shape)
    
    print('Model architecture:')
    print(model)
    