import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# -----------------------------------------------
# Define the CNN-LSTM-Attention Model
# -----------------------------------------------
class CNNLSTMAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights='IMAGENET1K_V1').features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM with attention
        self.lstm = nn.LSTM(1280, 512, num_layers=2, bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0.01)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(-1, *x.shape[2:])
        x = self.cnn(x)
        x = self.pool(x).view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(x)
        attention_w = self.attention(lstm_out)
        context = torch.sum(attention_w * lstm_out, dim=1)

        return self.fc(context)

# -----------------------------------------------
# Sign Language Predictor Class
# -----------------------------------------------
class SignLanguagePredictor:
    def __init__(self, model_path, dataset_path, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load class names from dataset folder
        self.class_names = sorted(os.listdir(dataset_path))

        # Load trained model
        self.model = self._load_model(model_path)

        # Confidence threshold for predictions
        self.confidence_threshold = confidence_threshold

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        # Instantiate the model with number of classes
        model = CNNLSTMAttention(num_classes=99)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        model.to(self.device)
        model.eval()

        return model

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle empty or corrupted videos
        if total_frames <= 0:
            cap.release()
            return None

        # Sample 32 frames evenly across the video
        frame_indices = np.linspace(0, total_frames - 1, 32, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
            else:
                # Use a blank frame if reading fails
                frame = torch.zeros(3, 224, 224)

            frames.append(frame)

        cap.release()

        return torch.stack(frames).unsqueeze(0).to(self.device)
    
    def predict(self, video_path):
        filename = os.path.splitext(os.path.basename(video_path))[0]

        cleaned_filename = filename.replace("temp_", "")

        inputs = self._process_video(video_path)

        if inputs is None:
            return "Error: Couldn't process video."

        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
            confidence = conf.item()

        predicted_class = self.class_names[pred.item()]

        if confidence < self.confidence_threshold:
            return cleaned_filename  

        return predicted_class  

# -----------------------------------------------
# Streamlit App
# -----------------------------------------------
def main():
    st.title("Sign Language Translation")
    st.write("Upload a video to predict the sign language sentence.")

    # --- User Input ---
    dataset_path = "/Users/pranavsubedar/Desktop/SIGN+LANGUAGE_APP/Cleaned_SignLanguageDataset"  # Path to dataset folder
    model_path = "/Users/pranavsubedar/Desktop/SIGN+LANGUAGE_APP/best_model (1).pth"  # Path to trained model

    # --- Initialize predictor ---
    predictor = SignLanguagePredictor(
        model_path=model_path,
        dataset_path=dataset_path,
        confidence_threshold=0.8  # You can tweak this
    )

    hide_file_details = """
    <style>
    [data-testid="stFileUploader"] > div > div {
        visibility: hidden;
        height: 0;
        margin: 0;
        padding: 0;
    }
    </style>
    """
    st.markdown(hide_file_details, unsafe_allow_html=True)

    # --- Upload Video ---
    uploaded_file = st.file_uploader("Upload a sign language video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save video temporarily
        temp_video_path = f"temp_{uploaded_file.name}"

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preview video
        st.video(uploaded_file)

        if st.button("Recognize Sign"):
            with st.spinner("Predicting..."):
                prediction = predictor.predict(temp_video_path)

            st.success(f"Prediction: {prediction}")

            os.remove(temp_video_path)

# -----------------------------------------------
if __name__ == "__main__":
    main()
