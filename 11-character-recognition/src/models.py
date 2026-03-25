import torch.nn as nn


class CannotParseModelName(ValueError):
    def __init__(self, str_name: str):
        super().__init__(f"Unable to parse name of model: {str_name}")


class EMNISTLogisticRegression(nn.Module):
    """EMNIST logistic regression model"""

    def __init__(self, num_classes):
        super().__init__()
        self.slug = "logreg"
        self.fc = nn.Linear(in_features=28 * 28, out_features=num_classes)

    def forward(self, x):
        return self.fc(x.flatten(start_dim=1))

    @staticmethod
    def from_slug(str_name: str, num_classes: int) -> "EMNISTLogisticRegression":
        if str_name == "logreg":
            return EMNISTLogisticRegression(num_classes)
        else:
            raise CannotParseModelName(str_name)


class EMNISTMultiLayerPerceptron(nn.Module):
    """EMNIST multi-layer perceptron"""

    def __init__(self, num_classes: int, depth: int = 1, width: int = 128):
        super().__init__()
        self.slug = f"mlp_{width}_{depth}"
        self.layers = nn.ModuleList([nn.Flatten()])

        in_features = 28 * 28
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(in_features, width))
            self.layers.append(nn.ReLU())
            in_features = width
        self.layers.append(nn.Linear(in_features, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def from_slug(str_name: str, num_classes: int) -> "EMNISTMultiLayerPerceptron":
        parts = str_name.split("_")
        if str_name.startswith("mlp") and len(parts) == 3:
            width = int(parts[1])
            depth = int(parts[2])
            return EMNISTMultiLayerPerceptron(num_classes, depth, width)
        else:
            raise CannotParseModelName(str_name)


class EMNISTConvNet(nn.Module):
    """EMNIST ConvNet model"""

    def __init__(
        self,
        num_classes: int,
        conv_layers: int = 5,
        conv_channels: int = 32,
        readout_width: int = 128,
        readout_depth: int = 2,
    ):
        super().__init__()
        self.slug = f"cnn_{conv_channels}_{conv_layers}_{readout_width}_{readout_depth}"
        self.layers = nn.ModuleList()

        in_channels = 1
        for _ in range(conv_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.layers.append(nn.ReLU())
            in_channels = conv_channels

        self.layers.append(nn.Flatten())
        in_features = 28 * 28 * conv_channels
        for _ in range(readout_depth - 1):
            self.layers.append(nn.Linear(in_features, readout_width))
            self.layers.append(nn.ReLU())
            in_features = readout_width
        self.layers.append(nn.Linear(readout_width, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def from_slug(str_name: str, num_classes: int) -> "EMNISTConvNet":
        parts = str_name.split("_")
        if str_name.startswith("cnn") and len(parts) == 5:
            conv_width = int(parts[1])
            conv_depth = int(parts[2])
            mlp_width = int(parts[3])
            mlp_depth = int(parts[4])
            return EMNISTConvNet(
                num_classes,
                conv_layers=conv_depth,
                conv_channels=conv_width,
                readout_width=mlp_width,
                readout_depth=mlp_depth,
            )
        else:
            raise CannotParseModelName(str_name)


EMNIST_MODELS = [EMNISTLogisticRegression, EMNISTMultiLayerPerceptron, EMNISTConvNet]


def get_model(str_name: str, num_classes: int) -> nn.Module:
    """Construct a model from an underscore separated name.

    Examples:
        'logreg' creates a logistic regression model, equivalent to 'mlp_0_1'
        'mlp_128_2' creates a multi layer perceptron with a width of 128 and 1 hidden layer
        'cnn_32_5_128_2' creates a convnet with 5 conv layers of 32 features each followed by a
            MLP readout which has one 128-unit hidden layer.
    """
    for cls in EMNIST_MODELS:
        try:
            model = cls.from_slug(str_name, num_classes)
            return model
        except CannotParseModelName:
            continue
    raise CannotParseModelName(str_name)
