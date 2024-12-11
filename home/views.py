from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.conf import settings
import pandas as pd
import numpy as np
import os
import io
import time
import base64
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths to model and dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "ml_model", "prices_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "fruit_veg_model.h5")
IMAGE_SIZE = (224, 224)

# Load the model and CSV
model = load_model(MODEL_PATH)
price_data = pd.read_csv(CSV_PATH)
price_data['Product_Name'] = price_data['Product_Name'].str.strip().str.lower()
price_data['Market_Price'] = price_data['Market_Price'].astype(float)
price_data['Price_Date'] = pd.to_datetime(price_data['Price_Date'])
price_mapping = price_data.groupby('Product_Name')['Market_Price'].mean().to_dict()

# Mapping classes
class_mapping = {idx: product_name for idx, product_name in enumerate(price_mapping.keys())}

def home(request):
    return render(request, 'home/index.html')

def camera(request):
    return render(request, 'home/camera.html')

def camera_and_predict(request):
    context = {}
    if request.method == 'POST' and request.POST.get('imageData'):
        image_data = request.POST.get('imageData')  # Get base64 image string

        try:
            # Decode the base64 image
            if ';base64,' not in image_data:
                raise ValueError("Invalid image data format.")
            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[1]  # Get file extension (e.g., png, jpeg)
            image = ContentFile(base64.b64decode(imgstr), name=f"captured_image.{ext}")

            # Save the image to a temporary location
            fs = FileSystemStorage()
            file_path = fs.save(image.name, image)
            uploaded_image_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # Preprocess the image for prediction
            img = load_img(uploaded_image_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class and price
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class_name = class_mapping.get(predicted_class_idx, "Unknown")
            price = price_mapping.get(predicted_class_name, "Price not available")
            if isinstance(price, float):  # Check kung numeric ang price
               price = round(price, 2)  # Round off to 2 decimal places

            # Generate price history plot
            graphic_visualization = plot_price_history(predicted_class_name)

            # Update context for rendering the result
            context.update({
                'image_name': image.name,
                'predicted_class_name': predicted_class_name,
                'predicted_price': price,
                'graphic_visualization': graphic_visualization,
                'MEDIA_URL': settings.MEDIA_URL,
                'timestamp': int(time.time()),  # Prevent caching
            })
        except Exception as e:
            context['error'] = f"An error occurred: {str(e)}"
            print(f"Error: {e}")  # Log the error for debugging

    return render(request, 'home/camera.html', context)

def upload_and_predict(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        image = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(image.name, image)
        uploaded_image_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            # Preprocess the image for prediction
            img = load_img(uploaded_image_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0  # Normalize if the model expects this
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class and price
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class_name = class_mapping[predicted_class_idx]
            price = price_mapping.get(predicted_class_name, "Price not available")
            if isinstance(price, float):  # Check kung numeric ang price
              price = round(price, 2)  # Round off to 2 decimal places

            # Generate price history plot
            graphic_visualization = plot_price_history(predicted_class_name)

            context = {
                'image_name': image.name,
                'predicted_class_name': predicted_class_name,
                'predicted_price': price,
                'graphic_visualization': graphic_visualization,
                'MEDIA_URL': settings.MEDIA_URL,
                'timestamp': int(time.time()),  # Added timestamp to prevent caching
            }
        except Exception as e:
            context['error'] = f"An error occurred: {str(e)}"
            print(f"Error: {str(e)}")  # Log error for debugging

    return render(request, 'home/upload.html', context)

def plot_price_history(product_name):
    product_data = price_data[price_data['Product_Name'] == product_name]
    if product_data.empty:
        return None

    plt.figure(figsize=(10, 6))
    plt.plot(product_data['Price_Date'], product_data['Market_Price'], marker='o', color='b')
    plt.title(f'Price History of {product_name.capitalize()}')
    plt.xlabel('Date')
    plt.ylabel('Price (per kg)')
    plt.grid(True)
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return base64.b64encode(image_png).decode('utf-8')
