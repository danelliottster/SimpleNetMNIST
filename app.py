import gradio as gr
import torch

from torchvision.transforms.functional import resize, to_tensor

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

simpnet_slim = SimpnetSlim310K(1, len(CLASSES))
simpnet_slim.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

def predict(image):
    if image is None:
        return None

    simpnet_slim.eval()
    with torch.inference_mode():
        x = to_tensor(resize(image, [32, 32])).unsqueeze(0)
        logits = simpnet_slim(x)
        probs = torch.softmax(logits, dim=1)

    return dict(zip(CLASSES, map(torch.Tensor.item, probs.squeeze())))

app = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(image_mode="L", type="pil"),
    outputs=gr.Label(),
    live=True
)

if __name__ == "__main__":
    app.launch()
