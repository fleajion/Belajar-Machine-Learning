#!/usr/bin/env python3
"""
Quick Start Script for ML Pose Classification
Automated setup and training pipeline
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


class QuickStart:
    """Quick start setup for ML pose classification"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.venv_dir = self.project_dir / 'venv'
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def print_step(self, step_num, text):
        """Print step number and description"""
        print(f"\n{'='*3} Step {step_num}: {text} {'='*3}")
    
    def check_python(self):
        """Check Python version"""
        self.print_step(1, "Checking Python version")
        
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ required!")
            print("   Please upgrade Python and try again.")
            sys.exit(1)
        
        print("✅ Python version OK!")
    
    def create_venv(self):
        """Create virtual environment"""
        self.print_step(2, "Creating virtual environment")
        
        if self.venv_dir.exists():
            print("⚠️  Virtual environment already exists")
            response = input("   Recreate? (y/n): ").strip().lower()
            
            if response == 'y':
                print("   Removing old venv...")
                shutil.rmtree(self.venv_dir)
            else:
                print("   Using existing venv")
                return
        
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created!")
    
    def get_pip_path(self):
        """Get pip executable path"""
        if sys.platform == 'win32':
            return self.venv_dir / 'Scripts' / 'pip.exe'
        else:
            return self.venv_dir / 'bin' / 'pip'
    
    def get_python_path(self):
        """Get python executable path"""
        if sys.platform == 'win32':
            return self.venv_dir / 'Scripts' / 'python.exe'
        else:
            return self.venv_dir / 'bin' / 'python'
    
    def install_dependencies(self):
        """Install required packages"""
        self.print_step(3, "Installing dependencies")
        
        pip_path = self.get_pip_path()
        
        # Upgrade pip
        print("Upgrading pip...")
        subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
        
        # Install requirements
        requirements_file = 'requirements_ml.txt'
        
        if not os.path.exists(requirements_file):
            requirements_file = 'requirements.txt'
        
        if os.path.exists(requirements_file):
            print(f"Installing packages from {requirements_file}...")
            subprocess.run([str(pip_path), 'install', '-r', requirements_file], check=True)
            print("✅ Dependencies installed!")
        else:
            print("⚠️  No requirements file found")
            print("   Installing core packages...")
            
            packages = [
                'opencv-python',
                'mediapipe',
                'numpy',
                'scikit-learn',
                'pandas',
                'matplotlib',
                'seaborn'
            ]
            
            for package in packages:
                print(f"   Installing {package}...")
                subprocess.run([str(pip_path), 'install', package], check=True)
            
            print("✅ Core packages installed!")
    
    def create_directories(self):
        """Create project directories"""
        self.print_step(4, "Creating project structure")
        
        directories = [
            'dataset/thinking',
            'dataset/pointing_up',
            'dataset/relaxed',
            'dataset/other',
            'dataset_augmented',
            'models',
            'results'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✓ {directory}/")
        
        print("✅ Project structure created!")
    
    def create_sample_script(self):
        """Create sample usage script"""
        self.print_step(5, "Creating sample scripts")
        
        sample_code = '''#!/usr/bin/env python3
"""
Sample script to get started with pose classification
"""

from ml_pose_classifier import PosePredictor

def main():
    print("🎯 Testing Pose Classifier")
    print("Loading model...")
    
    # Check if model exists
    import os
    if not os.path.exists('pose_classifier.pkl'):
        print("❌ Model not found!")
        print("   Please train a model first:")
        print("   python ml_pose_classifier.py")
        return
    
    # Load predictor
    predictor = PosePredictor('pose_classifier.pkl')
    
    # Test with webcam
    print("\\n🎥 Starting webcam...")
    print("Press 'q' to quit\\n")
    predictor.predict_webcam()

if __name__ == "__main__":
    main()
'''
        
        with open('quick_test.py', 'w') as f:
            f.write(sample_code)
        
        print("   ✓ quick_test.py")
        print("✅ Sample scripts created!")
    
    def show_next_steps(self):
        """Show next steps"""
        self.print_header("✅ SETUP COMPLETE!")
        
        print("\n📋 Next Steps:")
        print("\n1️⃣  Activate virtual environment:")
        
        if sys.platform == 'win32':
            print("     venv\\Scripts\\activate")
        else:
            print("     source venv/bin/activate")
        
        print("\n2️⃣  Collect training data:")
        print("     python data_augmentation.py")
        print("     → Choose option 1 (Collect data from webcam)")
        
        print("\n3️⃣  Train the model:")
        print("     python ml_pose_classifier.py")
        print("     → Choose option 3 (Train models)")
        
        print("\n4️⃣  Test the model:")
        print("     python quick_test.py")
        print("     Or:")
        print("     python ml_pose_classifier.py")
        print("     → Choose option 4 (Test on webcam)")
        
        print("\n📚 Documentation:")
        print("   - README_ML.md - Complete documentation")
        print("   - ML_Tutorial.ipynb - Jupyter notebook tutorial")
        print("   - INSTALLATION.md - Detailed installation guide")
        
        print("\n🎯 Alternative workflows:")
        print("   - Streamlit web app: streamlit run app_streamlit.py")
        print("   - Jupyter notebook: jupyter notebook ML_Tutorial.ipynb")
        
        print("\n💡 Tips:")
        print("   - Collect at least 50 images per pose class")
        print("   - Ensure good lighting when collecting data")
        print("   - Make poses clearly different from each other")
        print("   - Use data augmentation to increase dataset size")
        
        print("\n" + "="*60)
    
    def run(self):
        """Run complete setup"""
        self.print_header("🚀 ML POSE CLASSIFICATION - QUICK START")
        
        try:
            self.check_python()
            self.create_venv()
            self.install_dependencies()
            self.create_directories()
            self.create_sample_script()
            self.show_next_steps()
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error during setup: {e}")
            print("   Please check the error message and try again.")
            sys.exit(1)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup interrupted by user")
            sys.exit(1)
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("   Please report this issue.")
            sys.exit(1)


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       ML POSE CLASSIFICATION - QUICK START SETUP         ║
    ║                                                           ║
    ║   This script will:                                       ║
    ║   • Check Python version                                  ║
    ║   • Create virtual environment                            ║
    ║   • Install dependencies                                  ║
    ║   • Create project structure                              ║
    ║   • Generate sample scripts                               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    response = input("Continue with setup? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\nSetup cancelled.")
        sys.exit(0)
    
    quick_start = QuickStart()
    quick_start.run()


if __name__ == "__main__":
    main()
