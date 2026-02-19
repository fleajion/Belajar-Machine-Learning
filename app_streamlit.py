import streamlit as st
import cv2
import mediapipe as mp
from mediapipe import solutions
import numpy as np
from PIL import Image
import tempfile

class PoseDetectorWeb:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
    
    def calculate_angle(self, point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def detect_poses(self, landmarks):
        poses_detected = []
        
        # Thinking Pose
        nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
        
        left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        left_distance = np.sqrt((nose[0] - left_wrist[0])**2 + (nose[1] - left_wrist[1])**2)
        right_distance = np.sqrt((nose[0] - right_wrist[0])**2 + (nose[1] - right_wrist[1])**2)
        
        if left_distance < 0.15 or right_distance < 0.15:
            poses_detected.append("🤔 Thinking Pose")
        
        # Pointing Up Pose
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        if right_wrist[1] < right_shoulder[1] - 0.1:
            angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            if angle > 140:
                poses_detected.append("☝️ Pointing Up Pose")
        
        if left_wrist[1] < left_shoulder[1] - 0.1:
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            if angle > 140:
                poses_detected.append("☝️ Pointing Up Pose")
        
        # Relaxed Pose
        if len(poses_detected) == 0:
            left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            
            if abs(left_shoulder_y - right_shoulder_y) < 0.05:
                poses_detected.append("😌 Relaxed Pose")
        
        return poses_detected if poses_detected else ["❌ No Specific Pose Detected"]
    
    def process_image(self, image):
        # Convert PIL to OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        poses = []
        annotated_image = image_cv.copy()
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            poses = self.detect_poses(results.pose_landmarks.landmark)
        
        # Convert back to RGB for display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return annotated_image, poses


def main():
    st.set_page_config(
        page_title="Machine Pose Detection System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    st.title("🤖 Machine Pose Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        Sistem ini menggunakan MediaPipe untuk mendeteksi pose manusia.
        
        **Pose yang dapat dideteksi:**
        - 🤔 Thinking Pose
        - ☝️ Pointing Up Pose  
        - 😌 Relaxed Pose
        """)
        
        st.header("🎯 Referensi Pose")
        st.markdown("""
        **Thinking Pose**: Tangan di area wajah/dagu
        
        **Pointing Up**: Jari menunjuk ke atas dengan lengan terangkat
        
        **Relaxed**: Pose santai dengan bahu sejajar
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using MediaPipe & OpenCV")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar yang ingin dianalisis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
    
    with col2:
        st.header("🎯 Detection Result")
        
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing pose..."):
                detector = PoseDetectorWeb()
                annotated_image, detected_poses = detector.process_image(image)
                
                st.image(annotated_image, caption="Analyzed Image", use_container_width=True)
                
                st.subheader("Detected Poses:")
                for pose in detected_poses:
                    if "No Specific Pose" in pose:
                        st.warning(pose)
                    else:
                        st.success(pose)
        else:
            st.info("👆 Upload gambar untuk memulai deteksi pose")
    
    # Info boxes
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🤔 Thinking Pose",
            value="Hand to Face",
            delta="Distance < 0.15"
        )
    
    with col2:
        st.metric(
            label="☝️ Pointing Up",
            value="Arm Raised",
            delta="Angle > 140°"
        )
    
    with col3:
        st.metric(
            label="😌 Relaxed Pose",
            value="Neutral Position",
            delta="Balanced"
        )
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Cara Menggunakan Sistem:
        
        1. **Upload gambar** menggunakan file uploader di sebelah kiri
        2. Tunggu sistem **menganalisis** pose dalam gambar
        3. Lihat **hasil deteksi** dengan landmark pose yang ditampilkan
        4. **Pose yang terdeteksi** akan ditampilkan dengan emoji yang sesuai
        
        ### Tips untuk Hasil Terbaik:
        
        - Gunakan gambar dengan pencahayaan yang baik
        - Pastikan seluruh tubuh terlihat dalam frame
        - Gunakan background yang kontras dengan subjek
        - Pose harus jelas dan tidak blur
        
        ### Technical Details:
        
        - **Model**: MediaPipe Pose Estimation
        - **Framework**: OpenCV + NumPy
        - **Detection**: 33 landmark points
        - **Confidence**: Min 50%
        """)
    
    # Examples
    with st.expander("📸 Example Poses"):
        st.markdown("""
        ### Contoh Pose yang Dapat Dideteksi:
        
        **Thinking Pose 🤔**
        - Tangan kanan atau kiri di area wajah/dagu
        - Seperti orang sedang berpikir
        - Jarak tangan ke hidung < 15% dari frame
        
        **Pointing Up Pose ☝️**
        - Salah satu lengan terangkat ke atas
        - Jari menunjuk ke atas
        - Sudut siku > 140° (hampir lurus)
        - Pergelangan tangan di atas bahu
        
        **Relaxed Pose 😌**
        - Berdiri atau duduk dengan santai
        - Bahu sejajar (tidak miring)
        - Tidak ada gestur khusus
        """)


if __name__ == "__main__":
    main()
