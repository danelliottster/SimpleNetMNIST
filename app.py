import gradio as gr
import torch

from torchvision.transforms.functional import pad, resize, to_tensor

from simpnet import SimpnetSlim310K

CLASSES = (
    '0 - zero',
    '1 - one',
    '2 - two',
    '3 - three',
    '4 - four',
    '5 - five',
    '6 - six',
    '7 - seven',
    '8 - eight',
    '9 - nine'
)

MODEL_PATH = "models/simpnet_slim_310k.pt"

simpnet_slim = SimpnetSlim310K(1, len(CLASSES), 64)
simpnet_slim.load_state_dict(torch.load(MODEL_PATH))

def predict(image):
    if image is None:
        return None

    simpnet_slim.eval()
    with torch.inference_mode():
        X = pad(resize(to_tensor(image), [28, 28], antialias=True), [2])
        logits = simpnet_slim(X.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)

    return dict(zip(CLASSES, map(torch.Tensor.item, probs.squeeze())))

app = gr.Interface(
    fn=predict,
    inputs="sketchpad",
    outputs="label",
    live=True
)

if __name__ == "__main__":
    app.launch()
