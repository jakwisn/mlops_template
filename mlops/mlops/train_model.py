import torch
from models import MyNeuralNet
from torch.utils.data import Dataset, TensorDataset
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """Train a model on MNIST."""

    lr = 0.001
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")

    print("Training day and night")
    print(lr)

    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")

    train_set = TensorDataset(train_images, train_target)

    model = MyNeuralNet(1, 10 ).to(device)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            history.append(loss.item())
        print(f"Epoch {epoch} - Training loss: {train_loss/len(trainloader)}")

    if not os.path.exists("models/saved_models"):
        os.makedirs("models/saved_models")

    torch.save(model, "models/saved_models/model.pt")

    plt.plot(range(len(history)), history, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig('reports/figures/training_curve.png')

#
# @click.command()
# @click.argument("model_checkpoint")
# def evaluate(model_checkpoint):
#     """Evaluate a trained model."""
#     print("Evaluating like my life dependends on it")
#     print(model_checkpoint)
#
#     test_images = torch.load("mlops/data/processed/test_images.pt")
#     test_target = torch.load("mlops/data/processed/test_target.pt")
#     test_set = TensorDataset(test_images, test_target)
#
#     model = torch.load(model_checkpoint)
#     testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
#     accuracy = 0
#     with torch.no_grad():
#         for images, labels in testloader:
#             output = model(images)
#             _, predicted = torch.max(output.data, 1)
#             accuracy += (predicted == labels).sum().item()
#     print(f"Accuracy: {accuracy/len(test_set)}")


# cli.add_command(train)
# cli.add_command(evaluate)



