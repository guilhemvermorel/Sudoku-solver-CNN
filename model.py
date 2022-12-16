import pandas as pd
import numpy as np



class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
      
        return torch.cat([x[..., ::3, ::3], x[..., 1::3, ::3], x[..., 2::3, ::3], 
                          x[..., 2::3, 1::3],x[..., 1::3, 1::3], x[..., ::3, 1::3],
                          x[..., ::3, 2::3], x[..., 1::3, 2::3],x[..., 2::3, 2::3]], 1)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True),
        ]

        if stride ==3:


            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= 1, kernel_size=3, padding=1, bias= False),
            space_to_depth(),   # the output of this will result in 4*out_channels
            nn.BatchNorm2d(9*out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(9*out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        else:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= stride, kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        layers.extend(layers2)

        self.residual_function = torch.nn.Sequential(*layers)


        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, channel_init,num_classes=81*9):#9
        super().__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_init, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.in_channels = 64
		
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)    # Here in_channels = 64, and num_block[0] = 64 and s = 1 
        self.conv3_x = self._make_layer(block, 128, num_block[1], 3)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 3)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        output = self.conv1(x)

        output = self.conv2_x(output)

        output = self.conv3_x(output)

        output = self.conv4_x(output)

        output = output.view(output.size(0), -1)

        output = self.fc(output)

        return output

def resnet1(channel_init):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [1, 1, 1, 1],channel_init)


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


def resnet18():
    """
    return a ResNet 18 object
    """
    return ResNet(BottleNeck,[3,4,6,3])

  
  
  
  
def save_model():
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_function
            }, './model.tar')


  df_save = pd.DataFrame({'epochs':[i for i in range(len(train_loss))],'train_loss': train_loss,
                        'valid_loss': valid_loss, 'train_accuracy1': train_accuracy1,
                        'train_accuracy2' : train_accuracy2, 'train_accuracy3' : train_accuracy3,
                        'valid_accuracy1' : valid_accuracy1,'valid_accuracy2' : valid_accuracy2,
                        'valid_accuracy3' : valid_accuracy3})
  
  
  df_save.to_csv('./data.csv')


def load_model(): 
  df_load = pd.read_csv('./data.csv')

  epochs=[i for i in range(n_epochs)]
  train_loss = df_load['train_loss'].tolist()
  valid_loss = df_load['valid_loss'].tolist()
  train_accuracy1 = df_load['train_accuracy1'].tolist()
  train_accuracy2 = df_load['train_accuracy2'].tolist()
  train_accuracy3 = df_load['train_accuracy3'].tolist()

  valid_accuracy1 = df_load['valid_accuracy1'].tolist()
  valid_accuracy2 = df_load['valid_accuracy2'].tolist()
  valid_accuracy3 = df_load['valid_accuracy3'].tolist()

  del(df_load)

  checkpoint = torch.load('./model.tar')

  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  depoch = checkpoint['epoch']
  loss_function = checkpoint['loss']
