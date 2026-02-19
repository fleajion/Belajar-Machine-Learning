# 🚀 Panduan Instalasi Sistem Deteksi Pose

## 📋 Prerequisites

Sebelum memulai, pastikan Anda sudah menginstall:

- Python 3.8 atau lebih baru
- pip (Python package installer)
- Webcam (untuk mode real-time)

## 🔧 Step-by-Step Installation

### 1. Clone atau Download Project

```bash
# Jika menggunakan Git
git clone [your-repo-url]
cd pose-detection-system

# Atau download dan extract file ZIP
```

### 2. Buat Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Jika ingin menggunakan Streamlit web app:
```bash
pip install streamlit
```

### 4. Verify Installation

```bash
python -c "import cv2; import mediapipe; print('Installation successful!')"
```

## 🎮 Cara Menjalankan

### Option 1: Command Line Interface

```bash
python pose_detection_system.py
```

Kemudian pilih mode:
- `1` untuk Webcam (Real-time)
- `2` untuk Proses Gambar

### Option 2: Streamlit Web Interface

```bash
streamlit run app_streamlit.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

## 🐛 Troubleshooting

### Error: "No module named 'cv2'"

**Solusi:**
```bash
pip install opencv-python
```

### Error: "No module named 'mediapipe'"

**Solusi:**
```bash
pip install mediapipe
```

### Webcam tidak terdeteksi

**Solusi:**

1. Pastikan tidak ada aplikasi lain yang menggunakan webcam
2. Coba ubah index kamera di `pose_detection_system.py`:
   ```python
   cap = cv2.VideoCapture(0)  # Coba 0, 1, atau 2
   ```
3. Untuk Linux, jalankan:
   ```bash
   sudo apt-get install v4l-utils
   v4l2-ctl --list-devices
   ```

### Error: "DLL load failed" (Windows)

**Solusi:**

Install Visual C++ Redistributable:
- Download dari: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install dan restart komputer

### Performance lambat

**Solusi:**

1. Kurangi resolusi webcam:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

2. Kurangi confidence threshold di MediaPipe:
   ```python
   self.pose = self.mp_pose.Pose(
       min_detection_confidence=0.3,  # dari 0.5
       min_tracking_confidence=0.3     # dari 0.5
   )
   ```

## 🔍 Testing

### Test dengan Gambar Sample

```bash
python pose_detection_system.py
# Pilih: 2
# Masukkan path: test_image.jpg
```

### Test Webcam

```bash
python pose_detection_system.py
# Pilih: 1
# Lakukan berbagai pose
# Tekan 's' untuk screenshot
# Tekan 'q' untuk keluar
```

## 📦 Requirements Details

```
opencv-python==4.8.1.78
- Library untuk computer vision
- Handling image/video processing

mediapipe==0.10.8
- Google's ML framework
- Pose estimation model

numpy==1.24.3
- Numerical computing
- Array operations

streamlit (optional)
- Web interface
- Interactive dashboard
```

## 🌐 Platform Support

### ✅ Tested On:

- Windows 10/11
- macOS (Big Sur and later)
- Ubuntu 20.04/22.04
- Raspberry Pi OS (with performance limitations)

### 📱 Mobile:

Untuk deployment mobile, consider:
- TensorFlow Lite
- MediaPipe Mobile
- React Native / Flutter integration

## 🔐 Permissions

### Camera Access

**Windows:** 
- Settings → Privacy → Camera → Allow apps

**macOS:**
- System Preferences → Security & Privacy → Camera

**Linux:**
- User harus dalam group `video`:
  ```bash
  sudo usermod -a -G video $USER
  ```

## 📊 System Requirements

### Minimum:
- CPU: Intel i3 atau equivalent
- RAM: 4 GB
- Storage: 500 MB
- Webcam: 640x480 @ 30fps

### Recommended:
- CPU: Intel i5 atau equivalent
- RAM: 8 GB
- Storage: 1 GB
- Webcam: 1280x720 @ 30fps
- GPU: Optional (untuk better performance)

## 🎯 Next Steps

Setelah instalasi berhasil:

1. ✅ Test dengan webcam
2. ✅ Try different poses
3. ✅ Customize detection parameters
4. ✅ Add your own poses
5. ✅ Build your application

## 📚 Additional Resources

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [OpenCV Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Streamlit Docs](https://docs.streamlit.io/)

## 💬 Support

Jika mengalami masalah:

1. Check troubleshooting section
2. Verify all dependencies installed
3. Check Python version compatibility
4. Review error messages carefully

---

**Happy Coding! 🚀**
