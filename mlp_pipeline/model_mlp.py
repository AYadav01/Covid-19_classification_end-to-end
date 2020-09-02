from torch import nn

class MLP(nn.Module):
    def __init__(self, input_num_features, neurons):
        super().__init__()
        self.layers = self._create_layers(input_num_features, neurons)

    def _create_layers(self, input_features, neurons):
        task_layers = []
        if isinstance(neurons, list):
            for index, arg in enumerate(neurons):
                if index == len(neurons) - 1:
                    task_layers += [nn.Linear(input_features, arg)]
                else:
                    task_layers += [nn.Linear(input_features, arg), nn.ReLU(), nn.BatchNorm1d(arg), nn.Dropout(p=0.2)]
                input_features = arg
            return nn.Sequential(*task_layers)
        else:
            print("Input neurons must be list")

    def forward(self, x):
        x = self.layers(x)
        return x


