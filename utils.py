"""
Utility module for configuration management
"""

import configparser
import os
import logging
from datetime import datetime


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            print(f"✅ Configuration loaded from {self.config_file}")
        else:
            print(f"⚠️  Config file not found. Using default values.")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self.config['DETECTION_THRESHOLDS'] = {
            'thinking_pose_distance': '0.15',
            'pointing_pose_angle': '140',
            'relaxed_pose_shoulder_diff': '0.05',
            'relaxed_pose_hip_diff': '0.05'
        }
        
        self.config['MEDIAPIPE_SETTINGS'] = {
            'min_detection_confidence': '0.5',
            'min_tracking_confidence': '0.5',
            'static_image_mode': 'false',
            'model_complexity': '1'
        }
        
        self.config['CAMERA_SETTINGS'] = {
            'camera_index': '0',
            'frame_width': '1280',
            'frame_height': '720',
            'target_fps': '30'
        }
    
    def get_float(self, section, key, default=0.0):
        """Get float value from config"""
        try:
            return self.config.getfloat(section, key)
        except:
            return default
    
    def get_int(self, section, key, default=0):
        """Get integer value from config"""
        try:
            return self.config.getint(section, key)
        except:
            return default
    
    def get_boolean(self, section, key, default=False):
        """Get boolean value from config"""
        try:
            return self.config.getboolean(section, key)
        except:
            return default
    
    def get_string(self, section, key, default=''):
        """Get string value from config"""
        try:
            return self.config.get(section, key)
        except:
            return default
    
    def get_detection_thresholds(self):
        """Get all detection thresholds"""
        return {
            'thinking_distance': self.get_float('DETECTION_THRESHOLDS', 'thinking_pose_distance', 0.15),
            'pointing_angle': self.get_float('DETECTION_THRESHOLDS', 'pointing_pose_angle', 140),
            'relaxed_shoulder': self.get_float('DETECTION_THRESHOLDS', 'relaxed_pose_shoulder_diff', 0.05),
            'relaxed_hip': self.get_float('DETECTION_THRESHOLDS', 'relaxed_pose_hip_diff', 0.05)
        }
    
    def get_mediapipe_settings(self):
        """Get MediaPipe configuration"""
        return {
            'min_detection_confidence': self.get_float('MEDIAPIPE_SETTINGS', 'min_detection_confidence', 0.5),
            'min_tracking_confidence': self.get_float('MEDIAPIPE_SETTINGS', 'min_tracking_confidence', 0.5),
            'static_image_mode': self.get_boolean('MEDIAPIPE_SETTINGS', 'static_image_mode', False),
            'model_complexity': self.get_int('MEDIAPIPE_SETTINGS', 'model_complexity', 1)
        }
    
    def get_camera_settings(self):
        """Get camera configuration"""
        return {
            'index': self.get_int('CAMERA_SETTINGS', 'camera_index', 0),
            'width': self.get_int('CAMERA_SETTINGS', 'frame_width', 1280),
            'height': self.get_int('CAMERA_SETTINGS', 'frame_height', 720),
            'fps': self.get_int('CAMERA_SETTINGS', 'target_fps', 30)
        }


class Logger:
    """Manages logging for the application"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging configuration"""
        enable_logging = self.config.get_boolean('LOGGING', 'enable_logging', True)
        
        if not enable_logging:
            logging.disable(logging.CRITICAL)
            return
        
        log_level = self.config.get_string('LOGGING', 'log_level', 'INFO')
        log_file = self.config.get_string('LOGGING', 'log_file', 'pose_detection.log')
        
        # Convert string to logging level
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        level = level_map.get(log_level.upper(), logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("=" * 50)
        logging.info("Pose Detection System Started")
        logging.info("=" * 50)
    
    @staticmethod
    def log_pose_detection(pose_name, confidence=None):
        """Log pose detection event"""
        if confidence:
            logging.info(f"Detected: {pose_name} (Confidence: {confidence:.2f})")
        else:
            logging.info(f"Detected: {pose_name}")
    
    @staticmethod
    def log_error(error_message):
        """Log error message"""
        logging.error(f"Error: {error_message}")
    
    @staticmethod
    def log_warning(warning_message):
        """Log warning message"""
        logging.warning(f"Warning: {warning_message}")


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = datetime.now()
        self.fps_history = []
    
    def update(self):
        """Update frame count"""
        self.frame_count += 1
    
    def get_fps(self):
        """Calculate current FPS"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            return fps
        return 0
    
    def get_average_fps(self):
        """Get average FPS"""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0
    
    def reset(self):
        """Reset counters"""
        self.frame_count = 0
        self.start_time = datetime.now()
        self.fps_history = []
    
    def get_stats(self):
        """Get performance statistics"""
        return {
            'current_fps': self.get_fps(),
            'average_fps': self.get_average_fps(),
            'total_frames': self.frame_count,
            'elapsed_time': (datetime.now() - self.start_time).total_seconds()
        }


class PoseStatistics:
    """Track pose detection statistics"""
    
    def __init__(self):
        self.pose_counts = {}
        self.session_start = datetime.now()
    
    def record_pose(self, pose_name):
        """Record a detected pose"""
        if pose_name in self.pose_counts:
            self.pose_counts[pose_name] += 1
        else:
            self.pose_counts[pose_name] = 1
    
    def get_statistics(self):
        """Get pose statistics"""
        total_detections = sum(self.pose_counts.values())
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        stats = {
            'total_detections': total_detections,
            'session_duration': session_duration,
            'poses': self.pose_counts.copy()
        }
        
        # Calculate percentages
        if total_detections > 0:
            stats['percentages'] = {
                pose: (count / total_detections * 100) 
                for pose, count in self.pose_counts.items()
            }
        else:
            stats['percentages'] = {}
        
        return stats
    
    def print_summary(self):
        """Print statistics summary"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("POSE DETECTION STATISTICS")
        print("=" * 50)
        print(f"Session Duration: {stats['session_duration']:.2f} seconds")
        print(f"Total Detections: {stats['total_detections']}")
        print("\nPose Breakdown:")
        
        for pose, count in stats['poses'].items():
            percentage = stats['percentages'].get(pose, 0)
            print(f"  {pose}: {count} ({percentage:.1f}%)")
        
        print("=" * 50 + "\n")
    
    def reset(self):
        """Reset statistics"""
        self.pose_counts = {}
        self.session_start = datetime.now()


# Example usage
if __name__ == "__main__":
    # Test configuration manager
    config = ConfigManager()
    
    print("\n📋 Detection Thresholds:")
    print(config.get_detection_thresholds())
    
    print("\n⚙️  MediaPipe Settings:")
    print(config.get_mediapipe_settings())
    
    print("\n📷 Camera Settings:")
    print(config.get_camera_settings())
    
    # Test logger
    logger = Logger(config)
    logger.log_pose_detection("Thinking Pose", 0.95)
    logger.log_warning("Low frame rate detected")
    
    # Test performance monitor
    perf = PerformanceMonitor()
    for i in range(100):
        perf.update()
    
    print("\n📊 Performance Stats:")
    print(perf.get_stats())
    
    # Test pose statistics
    stats = PoseStatistics()
    stats.record_pose("Thinking Pose")
    stats.record_pose("Pointing Up")
    stats.record_pose("Thinking Pose")
    stats.record_pose("Relaxed Pose")
    stats.print_summary()
