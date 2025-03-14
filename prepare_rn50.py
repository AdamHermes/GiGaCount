import os
import json
import urllib.request
import torch
from torch import nn, Tensor
from tqdm import tqdm

# RN50 Model URL (CLIP)
MODEL_URL = "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
MODEL_NAME = "RN50"
MODEL_SAVE_NAME = "resnet50"

# Paths
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(CURR_DIR, "weights")
CONFIG_DIR = os.path.join(CURR_DIR, "configs")
MODEL_PATH = os.path.join(WEIGHT_DIR, "RN50.pt")

# Ensure directories exist
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Function to download model if not present
def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading {MODEL_URL} to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, open(MODEL_PATH, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        chunk_size = 8192
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as progress:
            while True:
                buffer = response.read(chunk_size)
                if not buffer:
                    break
                out_file.write(buffer)
                progress.update(len(buffer))
    print("Download complete!")
    return MODEL_PATH

# Load model function (based on CLIP)
def load_clip_rn50(device="cpu"):
    """Loads the RN50 model and returns it"""
    model_path = download_model()
    model = torch.jit.load(model_path, map_location=device).eval()
    return model

# CLIP Text Encoder (Temporary Wrapper)
class CLIPTextEncoderTemp(nn.Module):
    def __init__(self, clip: nn.Module):
        super().__init__()
        self.context_length = clip.context_length
        self.vocab_size = clip.vocab_size
        self.dtype = clip.dtype
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

    def forward(self, text: Tensor) -> None:
        pass

# Prepare RN50 Model
def prepare_rn50():
    print("Preparing CLIP RN50 model...")
    device = torch.device("cpu")
    
    # Load the full model
    model = load_clip_rn50(device=device).to(device)

    # Extract encoders
    image_encoder = model.visual.to(device)
    text_encoder = CLIPTextEncoderTemp(model).to(device)

    # Save weights separately
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, f"clip_{MODEL_SAVE_NAME}.pth"))
    torch.save(image_encoder.state_dict(), os.path.join(WEIGHT_DIR, f"clip_image_encoder_{MODEL_SAVE_NAME}.pth"))
    torch.save(text_encoder.state_dict(), os.path.join(WEIGHT_DIR, f"clip_text_encoder_{MODEL_SAVE_NAME}.pth"))

    # Save configuration files
    model_config = {
        "embed_dim": model.embed_dim,
        "image_resolution": model.image_resolution,
        "vision_layers": model.vision_layers,
        "vision_width": model.vision_width,
        "vision_patch_size": model.vision_patch_size,
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
        "transformer_width": model.transformer_width,
        "transformer_heads": model.transformer_heads,
        "transformer_layers": model.transformer_layers,
    }
    image_encoder_config = {
        "embed_dim": model.embed_dim,
        "image_resolution": model.image_resolution,
        "vision_layers": model.vision_layers,
        "vision_width": model.vision_width,
        "vision_patch_size": model.vision_patch_size,
        "vision_heads": model.vision_heads,
    }
    text_encoder_config = {
        "embed_dim": model.embed_dim,
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
        "transformer_width": model.transformer_width,
        "transformer_heads": model.transformer_heads,
        "transformer_layers": model.transformer_layers,
    }

    # Write configs
    with open(os.path.join(CONFIG_DIR, f"clip_{MODEL_SAVE_NAME}.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(CONFIG_DIR, f"clip_image_encoder_{MODEL_SAVE_NAME}.json"), "w") as f:
        json.dump(image_encoder_config, f, indent=4)
    with open(os.path.join(CONFIG_DIR, f"clip_text_encoder_{MODEL_SAVE_NAME}.json"), "w") as f:
        json.dump(text_encoder_config, f, indent=4)

    print("RN50 model preparation complete!")

# Run the preparation function
if __name__ == "__main__":
    prepare_rn50()
