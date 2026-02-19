"""
Machine Learning Pose Classification System
Train custom model based on pose images
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class PoseFeatureExtractor:
    """Extract pose features from images"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        
    def extract_landmarks(self, image_path):
        """Extract pose landmarks from image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(landmarks)
        
        return None
    
    def calculate_angles(self, landmarks):
        """Calculate key angles from landmarks"""
        angles = []
        
        # Reshape landmarks
        lm = landmarks.reshape(-1, 4)
        
        def get_angle(p1_idx, p2_idx, p3_idx):
            p1 = lm[p1_idx][:2]
            p2 = lm[p2_idx][:2]
            p3 = lm[p3_idx][:2]
            
            radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - \
                      np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            
            return angle
        
        # Right arm angles
        angles.append(get_angle(12, 14, 16))  # Right shoulder-elbow-wrist
        angles.append(get_angle(11, 12, 14))  # Right shoulder angle
        
        # Left arm angles  
        angles.append(get_angle(11, 13, 15))  # Left shoulder-elbow-wrist
        angles.append(get_angle(12, 11, 13))  # Left shoulder angle
        
        # Right leg angles
        angles.append(get_angle(24, 26, 28))  # Right hip-knee-ankle
        
        # Left leg angles
        angles.append(get_angle(23, 25, 27))  # Left hip-knee-ankle
        
        # Body angles
        angles.append(get_angle(11, 23, 25))  # Left torso
        angles.append(get_angle(12, 24, 26))  # Right torso
        
        return np.array(angles)
    
    def calculate_distances(self, landmarks):
        """Calculate key distances"""
        distances = []
        lm = landmarks.reshape(-1, 4)
        
        def get_distance(idx1, idx2):
            p1 = lm[idx1][:2]
            p2 = lm[idx2][:2]
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Hand to face distances
        distances.append(get_distance(0, 15))   # Nose to left wrist
        distances.append(get_distance(0, 16))   # Nose to right wrist
        
        # Hand positions relative to shoulders
        distances.append(get_distance(11, 15))  # Left shoulder to left wrist
        distances.append(get_distance(12, 16))  # Right shoulder to right wrist
        
        # Hand positions relative to hips
        distances.append(get_distance(23, 15))  # Left hip to left wrist
        distances.append(get_distance(24, 16))  # Right hip to right wrist
        
        # Shoulder width
        distances.append(get_distance(11, 12))
        
        # Hip width
        distances.append(get_distance(23, 24))
        
        return np.array(distances)
    
    def extract_features(self, image_path):
        """Extract all features from image"""
        landmarks = self.extract_landmarks(image_path)
        
        if landmarks is None:
            return None
        
        # Get angles and distances
        angles = self.calculate_angles(landmarks)
        distances = self.calculate_distances(landmarks)
        
        # Combine all features
        features = np.concatenate([landmarks, angles, distances])
        
        return features


class PoseDatasetBuilder:
    """Build dataset from images"""
    
    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        self.extractor = PoseFeatureExtractor()
        self.features = []
        self.labels = []
        
    def create_directory_structure(self):
        """Create dataset directory structure"""
        poses = ['thinking', 'pointing_up', 'relaxed', 'other']
        
        for pose in poses:
            pose_dir = os.path.join(self.data_dir, pose)
            os.makedirs(pose_dir, exist_ok=True)
        
        print("📁 Dataset structure created:")
        print(f"   {self.data_dir}/")
        for pose in poses:
            print(f"   ├── {pose}/")
        print("\n💡 Place your images in the respective folders!")
        
    def load_dataset(self):
        """Load dataset from directory"""
        print("🔄 Loading dataset...")
        
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            
            if not os.path.isdir(label_dir):
                continue
            
            print(f"\n📂 Processing {label}...")
            
            for image_file in os.listdir(label_dir):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                image_path = os.path.join(label_dir, image_file)
                features = self.extractor.extract_features(image_path)
                
                if features is not None:
                    self.features.append(features)
                    self.labels.append(label)
                    print(f"   ✓ {image_file}")
                else:
                    print(f"   ✗ {image_file} - No pose detected")
        
        print(f"\n✅ Dataset loaded: {len(self.features)} samples")
        
        return np.array(self.features), np.array(self.labels)
    
    def save_dataset(self, filename='pose_dataset.pkl'):
        """Save extracted features"""
        data = {
            'features': np.array(self.features),
            'labels': np.array(self.labels)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Dataset saved to {filename}")
    
    def load_saved_dataset(self, filename='pose_dataset.pkl'):
        """Load saved dataset"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.features = data['features'].tolist()
        self.labels = data['labels'].tolist()
        
        print(f"📂 Dataset loaded from {filename}")
        return np.array(self.features), np.array(self.labels)


class PoseClassifierTrainer:
    """Train pose classification models"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.best_model = None
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, features, labels):
        """Prepare data for training"""
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        results = {}
        
        print("\n" + "="*60)
        print("🎯 TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"   Accuracy: {accuracy*100:.2f}%")
            
            # Classification report
            print("\n   Classification Report:")
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
            print("   " + report.replace("\n", "\n   "))
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_name]['model']
        
        print("\n" + "="*60)
        print(f"🏆 Best Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")
        print("="*60)
        
        return results, X_test, y_test
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_model_comparison(self, results, save_path='model_comparison.png'):
        """Plot model comparison"""
        models = list(results.keys())
        accuracies = [results[m]['accuracy']*100 for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylim(0, 105)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison saved to {save_path}")
        plt.close()
    
    def save_model(self, filename='pose_classifier.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.best_model,
            'label_encoder': self.label_encoder
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename='pose_classifier.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        
        print(f"📂 Model loaded from {filename}")


class PosePredictor:
    """Make predictions on new images"""
    
    def __init__(self, model_path='pose_classifier.pkl'):
        self.extractor = PoseFeatureExtractor()
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        
        print(f"✅ Model loaded from {model_path}")
    
    def predict(self, image_path):
        """Predict pose from image"""
        features = self.extractor.extract_features(image_path)
        
        if features is None:
            return None, 0.0
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
        else:
            confidence = 100.0
        
        # Decode label
        pose_name = self.label_encoder.inverse_transform([prediction])[0]
        
        return pose_name, confidence
    
    def predict_webcam(self):
        """Real-time prediction from webcam"""
        cap = cv2.VideoCapture(0)
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(min_detection_confidence=0.5)
        
        print("🎥 Webcam started. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save temporary frame
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Predict
            pose_name, confidence = self.predict(temp_path)
            
            # Draw landmarks
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
            
            # Display prediction
            if pose_name:
                text = f"{pose_name.upper()} ({confidence:.1f}%)"
                cv2.putText(frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Pose Classification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    """Main training pipeline"""
    print("="*60)
    print("🤖 MACHINE LEARNING POSE CLASSIFICATION")
    print("="*60)
    
    print("\nWhat would you like to do?")
    print("1. Create dataset structure")
    print("2. Build dataset from images")
    print("3. Train models")
    print("4. Test on webcam")
    print("5. Predict single image")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        builder = PoseDatasetBuilder()
        builder.create_directory_structure()
        
    elif choice == '2':
        builder = PoseDatasetBuilder()
        features, labels = builder.load_dataset()
        builder.save_dataset()
        
    elif choice == '3':
        # Load dataset
        builder = PoseDatasetBuilder()
        
        if os.path.exists('pose_dataset.pkl'):
            features, labels = builder.load_saved_dataset()
        else:
            features, labels = builder.load_dataset()
            builder.save_dataset()
        
        # Train models
        trainer = PoseClassifierTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)
        results, X_test, y_test = trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        # Get predictions from best model
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        y_pred = results[best_name]['predictions']
        
        # Plot results
        trainer.plot_confusion_matrix(y_test, y_pred)
        trainer.plot_model_comparison(results)
        
        # Save model
        trainer.save_model()
        
    elif choice == '4':
        if not os.path.exists('pose_classifier.pkl'):
            print("❌ Model not found! Train a model first.")
            return
        
        predictor = PosePredictor()
        predictor.predict_webcam()
        
    elif choice == '5':
        if not os.path.exists('pose_classifier.pkl'):
            print("❌ Model not found! Train a model first.")
            return
        
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print("❌ Image not found!")
            return
        
        predictor = PosePredictor()
        pose_name, confidence = predictor.predict(image_path)
        
        if pose_name:
            print(f"\n🎯 Prediction: {pose_name.upper()}")
            print(f"📊 Confidence: {confidence:.1f}%")
        else:
            print("❌ No pose detected in image")
    
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
