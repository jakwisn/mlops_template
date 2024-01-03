import torch
import sys

def predict(model, images) -> None:
    """Predict on images using a pretrained model."""
    print("Predicting")

    model = torch.load(model).to("cpu")
    images = torch.load(images)
    output = model(images)
    _, predicted = torch.max(output.data, 1)

    return predicted

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <model_path> <images_path>")
    else:
        model_path = sys.argv[1]
        images_path = sys.argv[2]
        predicted = predict(model_path, images_path)
        print(predicted)







