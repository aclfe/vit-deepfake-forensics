import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import cv2
import numpy as np
import skimage.feature
from scipy.ndimage import gaussian_filter
import tempfile
import os

repo_name = "kairavaclfe/deepfake-verifier-model"
model = AutoModelForImageClassification.from_pretrained(repo_name)
processor = AutoImageProcessor.from_pretrained(repo_name)
device = torch.device("cpu")

class HFModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

wrapped_model = HFModelWrapper(model)
cam = GradCAM(
    model=wrapped_model,
    target_layers=[wrapped_model.model.vit.encoder.layer[-1].layernorm_before],
    reshape_transform=vit_reshape_transform
)

def get_noise_map(image_path, window_size=8, threshold=1.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not loaded")
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    noise_var = np.abs(laplacian)
    h, w = img.shape
    noise_map = np.zeros((h, w))
    for i in range(0, h - window_size + 1, window_size // 2):
        for j in range(0, w - window_size + 1, window_size // 2):
            patch = noise_var[i:i+window_size, j:j+window_size]
            local_std = np.std(patch)
            noise_map[i:i+window_size, j:j+window_size] = local_std
    noise_map_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min() + 1e-8)
    anomaly_score = np.mean(noise_map_norm > threshold)
    return noise_map_norm, anomaly_score

def get_texture_map(image_path, radius=3, n_points=24):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not loaded")
    lbp = skimage.feature.local_binary_pattern(img, n_points, radius, method="uniform")
    h, w = lbp.shape
    texture_map = np.zeros((h, w))
    window_size = 16
    for i in range(0, h - window_size + 1, window_size // 2):
        for j in range(0, w - window_size + 1, window_size // 2):
            patch = lbp[i:i+window_size, j:j+window_size]
            hist, _ = np.histogram(patch.ravel(), bins=np.arange(0, n_points + 3), density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            texture_map[i:i+window_size, j:j+window_size] = entropy
    texture_map_norm = (texture_map.max() - texture_map) / (texture_map.max() - texture_map.min() + 1e-8)
    inconsistency_score = np.mean(texture_map_norm)
    return texture_map_norm, inconsistency_score

def get_ela(image_path, quality=90):
    img = cv2.imread(image_path)
    cv2.imwrite('temp.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    temp_img = cv2.imread('temp.jpg')
    ela = cv2.absdiff(img, temp_img)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela_norm = ela_gray / 255.0
    return ela_norm

def verify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        fake_prob = torch.softmax(logits, dim=-1)[0][1].item()
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=targets)[0]
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    ela_map = get_ela(image_path)
    ela_score = ela_map.mean()
    ela_norm = max(0, min(1, (ela_score - 0.5) * 4))
    noise_map, noise_score = get_noise_map(image_path)
    noise_norm = max(0, min(1, (noise_score - 0.7) * 5))
    texture_map, texture_score = get_texture_map(image_path)
    texture_norm = max(0, min(1, (texture_score - 0.3) * 3))
    hybrid_conf = max(0, min(1, (
        0.94 * fake_prob +
        0.02 * ela_norm +
        0.02 * noise_norm +
        0.02 * texture_norm
    )))
    if hybrid_conf > 0.5:
        label = "FAKE"
        conf_pct = hybrid_conf * 100
    else:
        label = "REAL"
        conf_pct = (1 - hybrid_conf) * 100
    explanation = f"""{label} — {conf_pct:.1f}% Forensic Confidence
- ELA detected compression artifacts: {ela_score:.3f} (uniform suggests generation)
- Noise analysis: inconsistency score {noise_score:.3f} (low entropy at blends indicates swaps)
- Texture analysis: edge inconsistency {texture_score:.3f} (non-uniform noise suggests manipulation)
- Hybrid score fuses neural DL (primary) + calibrated forensics (2025 best practice for robustness)"""
    return label, conf_pct, explanation, Image.fromarray(cam_overlay), Image.fromarray(ela_map * 255).convert("RGB"), Image.fromarray(noise_map * 255).convert("RGB"), Image.fromarray(texture_map * 255).convert("RGB")

def detect(image_path):
    if image_path is None:
        return "Upload an image", 0, "", None, None, None, None
    return verify_image(image_path)

iface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Label"),
        gr.Textbox(label="Confidence (%)"),
        gr.Textbox(label="Explanation"),
        gr.Image(label="Grad-CAM (XAI)"),
        gr.Image(label="ELA Map"),
        gr.Image(label="Noise Map"),
        gr.Image(label="Texture Map")
    ],
    title="Robust Synthetic Media Authenticity Verifier with OpenFake",
    description="Upload an image for hybrid deepfake detection (ViT DL + forensics). Trained on 160k+ images (140k + 2025 OpenFake).",
    examples=[
        "https://raw.githubusercontent.com/aclfe/vit-deepfake-forensics/main/images/fake_00020.jpg",
        "https://raw.githubusercontent.com/aclfe/vit-deepfake-forensics/main/images/fake_00174.jpg",
        "https://raw.githubusercontent.com/aclfe/vit-deepfake-forensics/main/images/real_00001.jpg",
        "https://raw.githubusercontent.com/aclfe/vit-deepfake-forensics/main/images/real_00013.jpg"
    ],
    cache_examples=True
)

iface.launch(share=True)