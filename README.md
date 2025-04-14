🤟 Indian Sign Language (ISL) Sentence-Level Translator using Deep Learning
This project presents a deep learning-based Sign Language Translation system designed to recognize and translate Indian Sign Language (ISL) sentence-level gestures into text using a CNN-RNN hybrid model with MobileNetV2 + Bi-LSTM + Attention, deployed through a user-friendly Streamlit web app.

📚 Dataset: ISL_CSLRT_CORPUS
We use the publicly available ISL CSLRT CORPUS which contains:

Videos_Sentence_Level: 685 videos across 99 sentence classes.

Each video contains a signer demonstrating a full ISL sentence.

🛠️ Preprocessing:
Each video is sampled into exactly 32 frames using OpenCV.

These frames are converted and saved as .npy arrays.

Total processed frames: 685 videos × 32 frames = 21,920 frames.

🧠 Model Architecture
A. Feature Extraction with MobileNetV2
Each frame is passed through MobileNetV2, extracting 1280-dimensional spatial features.

Adaptive Average Pooling is applied to ensure dimensional consistency.

B. Temporal Modeling with Bi-LSTM + Attention
Sequence of 32 feature vectors is passed into a Bidirectional LSTM.

An attention mechanism is added to focus on keyframes that contribute most to the sentence meaning.

The final context vector is passed through a dense classification head to predict the sentence class.

🏋️ Model Training & Evaluation
Loss Function: Cross-Entropy Loss

Optimizer: Adam

Learning Rate: 0.0001

Batch Size: 8

Epochs: 50+

Data Augmentation: Rotation, cropping, brightness adjustment

📊 Performance:
Training Accuracy: 87.35%

Validation Accuracy: 83.64%

🌐 Streamlit Web App
An elegant and minimal Streamlit interface is built for easy interaction.

Features:
Upload any ISL sentence-level video

View video preview

Get real-time predicted ISL sentence class

Clean layout with confidence filtering

How It Works:
Video is preprocessed (frame sampling)

Model inference is run on processed frames

Predicted class is displayed in UI
