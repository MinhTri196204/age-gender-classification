import streamlit as st
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
import os

# Định nghĩa lớp LightweightCNN
class LightweightCNN(torch.nn.Module):
    def __init__(self, num_gender_classes=2, num_age_classes=4):
        super(LightweightCNN, self).__init__()
        
        # Convolutional layers with BN, ReLU, MaxPool, Dropout
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_cnn = torch.nn.Dropout(0.25)
        
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.dropout_fc = torch.nn.Dropout(0.25)
        self.fc_gender = torch.nn.Linear(128, num_gender_classes)
        self.fc_age = torch.nn.Linear(128, num_age_classes)
        
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.dropout_cnn(x)
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout_cnn(x)
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.dropout_cnn(x)
        
        x = x.view(-1, 64 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        gender_out = self.fc_gender(x)
        age_out = self.fc_age(x)
        
        return gender_out, age_out

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình
def load_model():
    try:
        model = LightweightCNN(num_gender_classes=2, num_age_classes=4).to(device)
        model.load_state_dict(torch.load('lightweight_cnn_64x64.pth', map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("File 'lightweight_cnn_64x64.pth' không tồn tại. Vui lòng kiểm tra đường dẫn!")
        return None

# Tiền xử lý ảnh cho một khuôn mặt
def get_image_features(face_img):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize về 64x64 như lúc huấn luyện
        transforms.ToTensor(),  # Giữ ảnh RGB (3 kênh)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chuẩn hóa
    ])
    img = transform(face_img).unsqueeze(0).to(device)  # Thêm batch dimension
    return img

# Phát hiện và dự đoán giới tính/độ tuổi cho tất cả khuôn mặt
def detect_and_predict_faces(image, model):
    img_array = np.array(image)
    img_rgb = img_array if img_array.shape[-1] == 3 else cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    detector = MTCNN()
    faces = detector.detect_faces(img_rgb)
    
    if not faces:
        return [], [], []
    
    face_images, face_boxes, predictions = [], [], []
    
    for face in faces:
        x, y, w, h = face['box']
        face_img = img_rgb[y:y+h, x:x+w]
        face_img = Image.fromarray(face_img)
        
        processed_image = get_image_features(face_img)
        with torch.no_grad():
            gender_out, age_out = model(processed_image)
            _, pred_gender = torch.max(gender_out, 1)
            _, pred_age = torch.max(age_out, 1)
        
        gender_mapping = {0: 'Male', 1: 'Female'}
        age_mapping = {0: '0-18', 1: '19-30', 2: '31-50', 3: '>50'}
        
        gender = gender_mapping[pred_gender.item()]
        age = age_mapping[pred_age.item()]
        
        face_images.append(face_img)
        face_boxes.append((x, y, w, h))
        predictions.append((gender, age))
    
    return face_images, face_boxes, predictions

# Giao diện Streamlit
def main():
    st.title("Age and Gender Prediction for All Faces")
    st.write("Upload an image to predict the age and gender of all detected faces.")

    img_to_test = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if img_to_test is not None:
        image = Image.open(img_to_test)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        model = load_model()
        if model is None:
            return
        
        face_images, face_boxes, predictions = detect_and_predict_faces(image, model)
        
        if not face_images:
            st.error("Không tìm thấy khuôn mặt nào trong ảnh!")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(np.array(image))
            
            for (x, y, w, h), (gender, age) in zip(face_boxes, predictions):
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y-10, f'{gender}, {age}', color='red', fontsize=10, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8))
            
            ax.axis('off')
            st.pyplot(fig)
            
            st.subheader("Detected Faces")
            for i, (face_img, (gender, age)) in enumerate(zip(face_images, predictions)):
                st.image(face_img, caption=f"Face {i+1}: {gender}, Age: {age}", width=150)

if __name__ == "__main__":
    main()


    #cd "c:\Users\This PC\Downloads\Computer Vision"
    # streamlit run app1.py