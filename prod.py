import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # input [3, 640, 640]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),  # [64, 640, 640]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),      # [64, 320, 320]

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), # [128, 320, 320]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),      # [128, 160, 160]

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), # [256, 160, 160]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),      # [256, 80, 80]

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # [512, 80, 80]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),       # [512, 40, 40]

            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # [512, 40, 40]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),       # [512, 20, 20]
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features = 512*20*20, out_features = 1024),
            nn.ReLU(),
            nn.Linear(in_features = 1024, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 2)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def predict_image(image_path, model, device):
    image = Image.open(image_path)

    transform = transforms.Compose(
    [transforms.Resize((640, 640)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    image_tensor = transform(image).float().unsqueeze(0)

    input_tensor = image_tensor.to(device)
    input = Variable(input_tensor, requires_grad=True)

    output = model(input)

    #plt.imshow(image)

    #label = output.data.cpu().numpy()
    #return label[0][0]

    label = nn.functional.softmax(output.data.cpu())
    label = label.numpy()
    return label[0][0].round(2)

def format_string(input_string):
    if input_string.startswith('"') and input_string.endswith('"') or input_string.startswith("'") and input_string.endswith("'"):
        input_string = input_string.strip('"').strip("'")
    return input_string

def main():
    print("Starting...")
    net = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    #print('Using: ',device)
    net.to(device)

    model_name = input("Write name of model(without .pt) or just press Enter to use default model: ")
    if model_name == "":
        model_name = "2 model lr0.001 e15"
    #print(model_name)
    print("Loading a model...")
    if str(device) == "cpu":
        net.load_state_dict(torch.load(f"{model_name}.pt", map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(f"{model_name}.pt"))

    import os

    while True:
        file_path = input("Enter a file path of image or enter 'Stop' to exit: ")
        file_path = file_path.lower()
        file_path = format_string(file_path)

        if file_path == "stop":
            break

        if not os.path.exists(file_path):
            print("File does not exist. Please try another path.")
            continue
        elif not os.access(file_path, os.R_OK):
            print("You don't have permission to read this file. Please try another path.")
            continue

        try:
            result = predict_image(file_path, net, device)
            print(result)
            print("I think photo blurred on ", (100 * result).round(2), "%")
        except Exception as e:
            print("Unexpected error:", e)

if __name__ == "__main__":
    main()
