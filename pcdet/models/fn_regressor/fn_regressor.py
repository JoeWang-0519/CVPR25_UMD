import torch.nn as nn
import torch.nn.functional as F

class FNRegressor(nn.Module):
    def __init__(self):
        super(FNRegressor, self).__init__()
      
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(True),
            # nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Softplus()
            )

        self._initialize_weights()

    def forward(self, feature):
        feature = self.regressor(feature)

        return feature

    def _initialize_weights(self):
        # Apply Xavier initialization to the weights of each linear layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)        # Initialize gamma (scale) to 1
                nn.init.constant_(m.bias, 0)   