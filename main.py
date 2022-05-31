import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchbearer
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchbearer import Trial
from torchvision.datasets import MNIST
from Model import MNIST_CNN


def train_network():
    # define the loss function and the optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    trial.run(epochs=15)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)
    torch.save(model.state_dict(), "./CNNvisualisation.weights")


def load_image(name):
    im = transform(Image.open(name)).unsqueeze(0)

    image = torch.squeeze(im)[0:3, :].permute(1, 2, 0)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return im[:, 0:1, :, :]


def simple_activation():
    # plot the first layer features
    for i in range(0, 30): # number of feature maps
        plt.subplot(5, 6, i + 1)
        plt.axis('off')
        plt.imshow(weights[i, 0, :, :], cmap=plt.get_cmap('gray'))
    plt.show()


def hook_function(module, grad_in, grad_out):
    for i in range(grad_out.shape[1]):
        conv_output = grad_out.data[0, i]
        plt.subplot(5, int(1 + grad_out.shape[1] / 5), i + 1)
        plt.axis('off')
        plt.imshow(conv_output, cmap=plt.get_cmap('gray'))
    plt.show()


def visualise_maximum_activation(model, target, num=10, alpha=1.0):
    plt.suptitle('Maximal activations of classes')
    for selected in range(num):
        input_img = torch.randn(1, 1, 28, 28, requires_grad=True)
        # we're interested in maximising outputs of the 3rd layer:
        conv_output = None
        def hook_function(module, grad_in, grad_out):
            nonlocal conv_output
            # Gets the conv output of the selected filter/feature (from selected layer)
            conv_output = grad_out[0, selected]
        hook = target.register_forward_hook(hook_function)
        for i in range(40): # feature maps
            model(input_img)
            loss = torch.mean(conv_output)
            loss.backward()
            norm = input_img.grad.std() + 1e-5
            input_img.grad /= norm
            input_img.data = input_img + alpha * input_img.grad
        hook.remove()
        input_img = input_img.detach()
        plt.subplot(2, int(num / 2), selected + 1)
        plt.axis('off')
        plt.title(f'number {selected}')
        plt.imshow(input_img[0, 0], cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # convert each image to tensor format
    transform = transforms.Compose([
        transforms.ToTensor()  # convert to tensor
    ])

    # load data
    trainset = MNIST(".", train=True, download=True, transform=transform)
    testset = MNIST(".", train=False, download=True, transform=transform)


    # reset the data loaders
    torch.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=True)

    # build the model
    model = MNIST_CNN()


    train_network()


    #to visualise the network
    model = MNIST_CNN()
    model.load_state_dict(torch.load('CNNvisualisation.weights'))

    #extraction of first layer weights

    weights = model.conv1.weight.data.cpu()

    simple_activation()

    #take an image as an input and propegate it forward to get the response of the network at that layer
    from PIL import Image

    # transform = torchvision.transforms.ToTensor()
    im = load_image("1.PNG")


    hook = model.conv1.register_forward_hook(hook_function)  # register the hook
    model(im)  # forward pass
    hook.remove()  # Tidy up

    hook = model.conv2.register_forward_hook(hook_function)  # register the hook
    model(im)  # forward pass
    hook.remove()  # Tidy up

    # make some images to show the maximal response of the filter based on classes

    visualise_maximum_activation(model, model.fc1)
    visualise_maximum_activation(model, model.fc2)
    visualise_maximum_activation(model, model.fc3)
    visualise_maximum_activation(model, model.conv1)
    visualise_maximum_activation(model, model.conv3)
