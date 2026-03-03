🌾 Rice Leaf Disease Detection System
Deep Learning Based Smart Agricultural Diagnostic Tool
📌 Project Overview
The Rice Leaf Disease Detection System is a deep learning–powered web application designed to identify and classify rice leaf diseases using image input.
The system uses EfficientNet-B3 with transfer learning to classify rice leaf images into 8 disease categories. The trained model is integrated with a Flask-based web application, allowing farmers to upload leaf images and receive:

✅ Disease Name
✅ Confidence Score
✅ Organic Treatment Suggestions
✅ Chemical Treatment Recommendations
✅ Downloadable PDF Report
This project bridges the gap between AI research and real-world agricultural application.

🎯 Problem Statement
Rice crops are highly vulnerable to fungal, bacterial, and viral infections. Manual diagnosis:
Requires expert knowledge
Is time-consuming
Is often inaccurate in early stages
This system automates disease detection using computer vision and deep learning to enable early intervention.

🧠 Model Architecture
📦 Base Model: EfficientNet-B3
🔄 Transfer Learning with Fine-Tuning
🧪 Test-Time Augmentation (TTA)
🎯 Loss Function: Categorical Cross-Entropy
⚙️ Optimizer: Adam
📊 Validation Accuracy: 91.8%

🦠 Disease Classes
The model classifies images into the following 8 categories:
Rice Hispa
Sheath Blight
Leaf Blast
Leaf Scald
Narrow Brown Leaf Spot
Bacterial Leaf Blight
Brown Spot
Healthy Rice Leaf

🏗️ System Architecture

The application follows a Three-Tier Architecture:

1️⃣ Presentation Layer (Frontend)
HTML5
CSS3
JavaScript

2️⃣ Application Layer (Backend)
Python
Flask
TensorFlow / Keras

3️⃣ Data Layer
SQLite Database
Disease Knowledge Base (JSON Dictionary)

🔍 Key Features
📷 Image Upload & Validation
🧪 Automatic Image Preprocessing (300×300 Resize + Normalization)
🤖 Deep Learning Inference
📊 Confidence Score Calculation
📄 PDF Report Generation
📚 History Tracking (SQLite Logging)
🔥 Grad-CAM Explainability Support
