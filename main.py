import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image
from collections import Counter
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# COVID-19 X-RAY DETECTION USING RESNET50 + GRAD-CAM XAI
# ============================================================
# This project detects COVID-19 from chest X-ray images using:
# 1. ResNet50 deep learning model (Transfer Learning)
# 2. Grad-CAM (Gradient-weighted Class Activation Mapping)
#    for Explainable AI - shows WHERE the model looks
# 4 Classes: COVID, Normal, Lung Opacity, Viral Pneumonia
# Dataset: COVID-19 Radiography Database (21,165 images)
# Performance: 88.33% Test Accuracy, 91% COVID Recall
# ============================================================

# Configuration
MODEL_PATH = 'best_model.pth'
IMG_SIZE = 224
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
CLASS_COLORS = ['#FF4444', '#FF9800', '#4CAF50', '#2196F3']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# MODEL DEFINITION
# ============================================================

def build_model():
    """Build ResNet50 model with custom classifier"""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 4)
    )
    return model

def load_model(model_path):
    """Load trained model"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please ensure best_model.pth is in the same directory")
        return None
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully!")
    return model

# ============================================================
# GRAD-CAM IMPLEMENTATION (EXPLAINABLE AI)
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which regions of the X-ray the model focuses on
    for making its diagnosis decision.
    This is Explainable AI (XAI) - making AI decisions transparent.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """Generate Grad-CAM heatmap for given class"""
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def create_gradcam_overlay(image_array, cam_map, alpha=0.4):
    """Create colored overlay of Grad-CAM on original image"""
    heatmap = mpl_cm.jet(cam_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image_array, 1-alpha, heatmap, alpha, 0)
    return overlay, heatmap

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_input):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            image_input = image_input[:, :, :3]
        pil_img = Image.fromarray(image_input.astype(np.uint8))
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    else:
        pil_img = image_input.convert('RGB')

    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(pil_img)
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    return tensor, img_array

# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_covid(model, grad_cam, image_input):
    """
    Detect COVID-19 from chest X-ray image
    Returns: prediction, confidence, Grad-CAM overlay
    """
    input_tensor, img_array = preprocess_image(image_input)

    # Get prediction
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    # Generate Grad-CAM
    cam_map = grad_cam.generate(input_tensor, pred_idx)
    overlay, heatmap = create_gradcam_overlay(img_array, cam_map)

    return {
        'prediction': CLASSES[pred_idx],
        'confidence': confidence,
        'probabilities': probs.detach().cpu().numpy(),
        'cam_map': cam_map,
        'overlay': overlay,
        'heatmap': heatmap,
        'original': img_array
    }

# ============================================================
# VISUALIZATION DASHBOARD
# ============================================================

def create_dashboard(result, save_path='results_dashboard.png'):
    """Create comprehensive results dashboard"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('COVID-19 X-Ray Detection + Explainable AI (Grad-CAM)\nResNet50 Transfer Learning',
                 fontsize=16, fontweight='bold')

    # Row 1: Images
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(result['original'])
    ax1.set_title('Original X-Ray', fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(result['cam_map'], cmap='jet')
    ax2.set_title('Grad-CAM Heatmap\n(AI Focus Region)', fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(result['heatmap'])
    ax3.set_title('Pure Heatmap', fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(result['overlay'])
    pred = result['prediction']
    conf = result['confidence']
    ax4.set_title(f'Overlay Result\nPrediction: {pred} ({conf*100:.1f}%)',
                 fontweight='bold',
                 color='red' if pred == 'COVID' else 'green')
    ax4.axis('off')

    # Row 2: Charts
    ax5 = fig.add_subplot(2, 4, 5)
    probs = result['probabilities'] * 100
    bars = ax5.bar(CLASSES, probs, color=CLASS_COLORS, edgecolor='black')
    ax5.set_title('Class Probabilities', fontweight='bold')
    ax5.set_ylabel('Probability (%)')
    ax5.set_ylim(0, 105)
    ax5.tick_params(axis='x', rotation=15)
    for bar, prob in zip(bars, probs):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Model performance metrics
    ax6 = fig.add_subplot(2, 4, 6)
    metrics = ['Test\nAccuracy', 'COVID\nPrecision', 'COVID\nRecall', 'Viral Pneumonia\nPrecision']
    values = [88.33, 89.0, 91.0, 99.0]
    metric_colors = ['#2196F3', '#FF4444', '#FF4444', '#4CAF50']
    bars = ax6.bar(metrics, values, color=metric_colors, edgecolor='black', width=0.5)
    ax6.set_title('Model Performance\nMetrics', fontweight='bold')
    ax6.set_ylabel('Score (%)')
    ax6.set_ylim(0, 105)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Class distribution
    ax7 = fig.add_subplot(2, 4, 7)
    dataset_counts = [3616, 6012, 10192, 1345]
    wedges, texts, autotexts = ax7.pie(
        dataset_counts,
        labels=CLASSES,
        colors=CLASS_COLORS,
        autopct='%1.1f%%',
        startangle=90
    )
    ax7.set_title('Dataset Distribution\n(21,165 X-rays)', fontweight='bold')

    # Summary box
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    summary = "DETECTION SUMMARY\n"
    summary += "="*32 + "\n\n"
    summary += f"Prediction: {result['prediction']}\n"
    summary += f"Confidence: {result['confidence']*100:.1f}%\n\n"
    summary += "CLASS PROBABILITIES:\n"
    for cls, prob in zip(CLASSES, result['probabilities']):
        summary += f"  {cls:<18}: {prob*100:.1f}%\n"
    summary += "\n" + "="*32 + "\n"
    summary += "MODEL INFO\n"
    summary += "="*32 + "\n"
    summary += "Architecture: ResNet50\n"
    summary += "Method: Transfer Learning\n"
    summary += "XAI: Grad-CAM\n"
    summary += "Test Accuracy: 88.33%\n"
    summary += "COVID Recall: 91%\n"
    summary += "Dataset: 21,165 X-rays\n"

    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Dashboard saved as {save_path}")

# ============================================================
# GRADIO INTERFACE
# ============================================================

def gradio_detect(image, model, grad_cam):
    """Gradio prediction function"""
    if image is None:
        return None, None, None, "Please upload an X-ray image"

    result = detect_covid(model, grad_cam, image)

    # Create output images
    fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(result['original'])
    ax[0].set_title('Original X-Ray')
    ax[0].axis('off')
    ax[1].imshow(result['cam_map'], cmap='jet')
    ax[1].set_title('Grad-CAM Focus')
    ax[1].axis('off')
    ax[2].imshow(result['overlay'])
    ax[2].set_title(f"Result: {result['prediction']}")
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig('gradio_output.png', dpi=100)
    plt.close()

    output_img = plt.imread('gradio_output.png')

    result_text = f"DIAGNOSIS RESULT\n"
    result_text += "="*35 + "\n\n"
    result_text += f"Prediction: {result['prediction']}\n"
    result_text += f"Confidence: {result['confidence']*100:.1f}%\n\n"
    result_text += "All Class Probabilities:\n"
    for cls, prob in zip(CLASSES, result['probabilities']):
        bar = '█' * int(prob * 20)
        result_text += f"  {cls:<18}: {prob*100:.1f}% {bar}\n"
    result_text += f"\nModel: ResNet50 | XAI: Grad-CAM\n"
    result_text += f"Test Accuracy: 88.33%"

    return output_img, result_text

def launch_gradio(model, grad_cam):
    """Launch Gradio web interface"""
    def predict(image):
        return gradio_detect(image, model, grad_cam)

    app = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Upload Chest X-Ray Image"),
        outputs=[
            gr.Image(type="numpy", label="Grad-CAM Visualization"),
            gr.Textbox(label="Diagnosis Result", lines=15)
        ],
        title="COVID-19 X-Ray Detection with Explainable AI",
        description="""
        Upload a chest X-ray image to detect COVID-19 using ResNet50 deep learning.
        The system uses Grad-CAM (Explainable AI) to show WHERE in the X-ray 
        the model looks to make its decision - making AI transparent and trustworthy.
        
        Classes: COVID | Normal | Lung Opacity | Viral Pneumonia
        Model: ResNet50 (Transfer Learning) | Accuracy: 88.33%
        """,
        theme=gr.themes.Soft()
    )
    app.launch(share=True)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("   COVID-19 X-RAY DETECTION + EXPLAINABLE AI (Grad-CAM)")
    print("   Using ResNet50 Transfer Learning")
    print("="*60)

    # Load model
    print("\n[1] Loading ResNet50 model...")
    model = load_model(MODEL_PATH)

    if model is None:
        print("\nERROR: Model file (best_model.pth) not found!")
        print("Please place best_model.pth in the same folder as main.py")
        exit()

    # Setup Grad-CAM
    print("\n[2] Setting up Grad-CAM XAI...")
    grad_cam = GradCAM(model, model.layer4[-1])
    print("Grad-CAM ready!")

    # Test on sample images
    import glob
    test_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        test_images.extend(glob.glob(ext))

    if test_images:
        print(f"\n[3] Running detection on {test_images[0]}...")
        result = detect_covid(model, grad_cam, test_images[0])

        print("\n" + "="*50)
        print("   DETECTION RESULTS")
        print("="*50)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print("\nAll probabilities:")
        for cls, prob in zip(CLASSES, result['probabilities']):
            print(f"  {cls}: {prob*100:.1f}%")

        print("\n[4] Creating results dashboard...")
        create_dashboard(result)
    else:
        print("\n[3] No test images found")
        print("    Place chest X-ray images (.jpg/.png) in this folder")

    # Launch Gradio
    print("\n[5] Launching Gradio Web Interface...")
    launch_gradio(model, grad_cam)