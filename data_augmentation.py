"""
Data Augmentation for Pose Dataset
Generate more training samples from existing images
"""

import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A


class PoseDataAugmentor:
    """Augment pose images to increase dataset size"""
    
    def __init__(self):
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ])
        
        # Simple transforms without albumentations
        self.simple_transforms = [
            self.flip_horizontal,
            self.rotate_image,
            self.adjust_brightness,
            self.add_noise,
            self.blur_image
        ]
    
    def flip_horizontal(self, image):
        """Flip image horizontally"""
        return cv2.flip(image, 1)
    
    def rotate_image(self, image, angle=None):
        """Rotate image by random angle"""
        if angle is None:
            angle = np.random.randint(-15, 15)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def adjust_brightness(self, image, factor=None):
        """Adjust image brightness"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def add_noise(self, image):
        """Add Gaussian noise"""
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
    
    def blur_image(self, image):
        """Apply slight blur"""
        kernel_size = np.random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def augment_image_simple(self, image, num_augments=5):
        """Generate augmented versions using simple transforms"""
        augmented_images = [image.copy()]
        
        for _ in range(num_augments):
            aug_image = image.copy()
            
            # Apply random transforms
            num_transforms = np.random.randint(1, 4)
            selected_transforms = np.random.choice(
                self.simple_transforms, 
                size=num_transforms, 
                replace=False
            )
            
            for transform in selected_transforms:
                aug_image = transform(aug_image)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def augment_image_advanced(self, image, num_augments=5):
        """Generate augmented versions using albumentations"""
        augmented_images = [image.copy()]
        
        for _ in range(num_augments):
            augmented = self.transform(image=image)
            augmented_images.append(augmented['image'])
        
        return augmented_images
    
    def augment_dataset(self, input_dir, output_dir, num_augments=5, use_advanced=False):
        """Augment all images in dataset"""
        print("🔄 Starting data augmentation...")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Augmentations per image: {num_augments}")
        
        total_images = 0
        
        # Process each class folder
        for class_name in os.listdir(input_dir):
            class_input_dir = os.path.join(input_dir, class_name)
            
            if not os.path.isdir(class_input_dir):
                continue
            
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            print(f"\n📂 Processing class: {class_name}")
            
            image_count = 0
            
            for image_file in os.listdir(class_input_dir):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                image_path = os.path.join(class_input_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"   ✗ Failed to read: {image_file}")
                    continue
                
                # Generate augmented images
                if use_advanced:
                    try:
                        augmented_images = self.augment_image_advanced(image, num_augments)
                    except:
                        print("   ⚠️  Falling back to simple augmentation")
                        augmented_images = self.augment_image_simple(image, num_augments)
                else:
                    augmented_images = self.augment_image_simple(image, num_augments)
                
                # Save augmented images
                base_name = Path(image_file).stem
                ext = Path(image_file).suffix
                
                for idx, aug_image in enumerate(augmented_images):
                    if idx == 0:
                        output_name = f"{base_name}_original{ext}"
                    else:
                        output_name = f"{base_name}_aug{idx}{ext}"
                    
                    output_path = os.path.join(class_output_dir, output_name)
                    cv2.imwrite(output_path, aug_image)
                    image_count += 1
                
                print(f"   ✓ {image_file} → {len(augmented_images)} images")
            
            print(f"   Total images in {class_name}: {image_count}")
            total_images += image_count
        
        print(f"\n✅ Augmentation complete!")
        print(f"   Total images generated: {total_images}")
        
        return total_images


class WebcamDataCollector:
    """Collect training data from webcam"""
    
    def __init__(self, output_dir='dataset'):
        self.output_dir = output_dir
        self.current_class = None
        self.image_count = 0
    
    def collect_data(self, class_name, target_count=50):
        """Collect images for a specific pose class"""
        self.current_class = class_name
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Count existing images
        existing = len([f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.image_count = existing
        
        cap = cv2.VideoCapture(0)
        
        print(f"\n📸 Collecting data for: {class_name}")
        print(f"   Target: {target_count} images")
        print(f"   Existing: {existing} images")
        print(f"   Needed: {max(0, target_count - existing)} images")
        print("\nControls:")
        print("   SPACE - Capture image")
        print("   Q - Quit")
        
        while cap.isOpened() and self.image_count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display
            display_frame = frame.copy()
            
            # Add text overlay
            remaining = target_count - self.image_count
            text1 = f"Class: {class_name}"
            text2 = f"Captured: {self.image_count}/{target_count}"
            text3 = f"Remaining: {remaining}"
            
            cv2.putText(display_frame, text1, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, text2, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, text3, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                # Save image
                filename = f"{class_name}_{self.image_count:04d}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, frame)
                self.image_count += 1
                print(f"   ✓ Saved: {filename} ({self.image_count}/{target_count})")
            
            elif key == ord('q'):
                print("\n⚠️  Collection cancelled by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collection complete: {self.image_count} images saved")
        
        return self.image_count
    
    def collect_all_classes(self, classes, images_per_class=50):
        """Collect data for multiple pose classes"""
        print("="*60)
        print("📸 AUTOMATED DATA COLLECTION")
        print("="*60)
        
        for class_name in classes:
            input(f"\nPress ENTER to start collecting '{class_name}' poses...")
            self.collect_data(class_name, images_per_class)
        
        print("\n" + "="*60)
        print("✅ ALL DATA COLLECTION COMPLETE")
        print("="*60)


def main():
    """Main data augmentation pipeline"""
    print("="*60)
    print("📊 DATA AUGMENTATION & COLLECTION")
    print("="*60)
    
    print("\nWhat would you like to do?")
    print("1. Collect data from webcam")
    print("2. Augment existing dataset")
    print("3. Collect + Augment (full pipeline)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Collect data
        collector = WebcamDataCollector()
        
        print("\nPose classes to collect:")
        print("1. thinking")
        print("2. pointing_up")
        print("3. relaxed")
        print("4. other")
        print("5. Custom classes")
        
        class_choice = input("\nEnter choice: ").strip()
        
        if class_choice == '5':
            classes = input("Enter class names (comma-separated): ").split(',')
            classes = [c.strip() for c in classes]
        else:
            classes = ['thinking', 'pointing_up', 'relaxed', 'other']
        
        images_per_class = int(input("Images per class (default 50): ") or "50")
        
        collector.collect_all_classes(classes, images_per_class)
        
    elif choice == '2':
        # Augment data
        input_dir = input("Input dataset directory (default: dataset): ").strip() or 'dataset'
        output_dir = input("Output directory (default: dataset_augmented): ").strip() or 'dataset_augmented'
        num_augments = int(input("Augmentations per image (default: 5): ") or "5")
        
        augmentor = PoseDataAugmentor()
        augmentor.augment_dataset(input_dir, output_dir, num_augments)
        
    elif choice == '3':
        # Full pipeline
        print("\n🔄 Starting full pipeline...")
        
        # Step 1: Collect data
        collector = WebcamDataCollector()
        classes = ['thinking', 'pointing_up', 'relaxed', 'other']
        collector.collect_all_classes(classes, images_per_class=30)
        
        # Step 2: Augment data
        print("\n🔄 Starting augmentation...")
        augmentor = PoseDataAugmentor()
        augmentor.augment_dataset('dataset', 'dataset_augmented', num_augments=5)
        
        print("\n✅ Full pipeline complete!")
        print("   Next step: Train your model using ml_pose_classifier.py")
    
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
