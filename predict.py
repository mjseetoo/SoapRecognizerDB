import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io

# Function to resize image
def resize_image(uploaded_image):
    max_size=(400, 400)
    image = Image.open(uploaded_image)
    
    image.thumbnail(max_size)
    
    img = image.convert('RGB')
    # Convert PIL image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    
    return img_bytes

# Function to predict class of image
def predict_image(image, model):
    # Open the image and apply the same transformations used during training
    
    transform = ToTensor()
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        # Map predicted label index to class name
        class_names = ["soap", "not soap"]
        predicted_class = class_names[predicted.item()]
        
        return predicted_class