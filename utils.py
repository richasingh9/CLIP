import torch
from transformers import CLIPProcessor
from transformers import CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.to(device)


def predict_label(image, labels):

    prompts = [
        f"a photo of a {label}"
        for label in labels
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probs = outputs.logits_per_image.softmax(
            dim=1
        )

    idx = probs.argmax().item()

    return (
        labels[idx],
        probs[0][idx].item()
    )
