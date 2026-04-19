"""
Visualize inference results on video with keypoints, action predictions, and statistics.

This script:
1. Loads behavior predictions from video_inference directory
2. Loads keypoint detection results
3. Draws bounding boxes, keypoints, and skeleton connections
4. Displays action predictions with color coding
5. Calculates and displays scratching statistics (time, frequency)
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


class VideoVisualizer:
    """Visualize behavior predictions and keypoints on video."""
    
    # Define skeleton connections for mouse (quadruped)
    SKELETON_CONNECTIONS = [
        # Spine
        (0, 1), (1, 2), (2, 3),
        # Head
        (0, 4), (0, 5),
        # Front left leg
        (1, 6), (6, 7), (7, 8),
        # Front right leg
        (1, 9), (9, 10), (10, 11),
        # Back left leg
        (2, 12), (12, 13), (13, 14),
        # Back right leg
        (2, 15), (15, 16), (16, 17),
        # Tail
        (3, 18), (18, 19), (19, 20)
    ]
    
    # Colors for different elements (BGR format)
    COLORS = {
        'bbox': (0, 255, 0),  # Green
        'keypoint': (0, 255, 0),  # Green (changed from red)
        'skeleton': (255, 0, 0),  # Blue
        'no_behavior_bg': (255, 200, 150),  # Light blue background
        'scratching_bg': (150, 230, 255),  # Light yellow background
        'label_text': (0, 0, 0),  # Black text
        'text_bg': (0, 0, 0),  # Black
        'text': (255, 255, 255)  # White
    }
    
    def __init__(self, 
                 video_path: str,
                 inference_dir: str,
                 keypoint_file: str,
                 output_path: str,
                 start_time: float = 0.0):
        """
        Initialize the visualizer.
        
        Args:
            video_path: Path to input video
            inference_dir: Directory containing predictions.npy and statistics.json
            keypoint_file: Path to keypoint JSON file
            output_path: Path to save output video
            start_time: Starting time in seconds (default: 0.0)
        """
        self.video_path = Path(video_path)
        self.inference_dir = Path(inference_dir)
        self.keypoint_file = Path(keypoint_file)
        self.output_path = Path(output_path)
        self.start_time = start_time
        
        # Load data
        self.predictions = self._load_predictions()
        self.statistics = self._load_statistics()
        self.keypoints_data = self._load_keypoints()
        
        # Calculate statistics
        self.scratch_stats = self._calculate_scratch_statistics()
        
    def _load_predictions(self) -> np.ndarray:
        """Load behavior predictions."""
        pred_path = self.inference_dir / "predictions.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        return np.load(pred_path)
    
    def _load_statistics(self) -> Dict:
        """Load statistics JSON."""
        stats_path = self.inference_dir / "statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        with open(stats_path, 'r') as f:
            return json.load(f)
    
    def _load_keypoints(self) -> Dict:
        """Load keypoint detection results."""
        if not self.keypoint_file.exists():
            raise FileNotFoundError(f"Keypoint file not found: {self.keypoint_file}")
        with open(self.keypoint_file, 'r') as f:
            return json.load(f)
    
    def _calculate_scratch_statistics(self) -> Dict:
        """Calculate scratching duration and frequency statistics."""
        segments = self.statistics.get('segments', [])
        
        scratch_durations = []
        scratch_count = 0
        total_scratch_time = 0.0
        
        for segment in segments:
            if segment['label'] in ('scratching', 'scracthing'):  # Note: typo in original data
                duration = segment['duration']
                scratch_durations.append(duration)
                scratch_count += 1
                total_scratch_time += duration
        
        # Calculate statistics
        mean_scratch_duration = np.mean(scratch_durations) if scratch_durations else 0.0
        total_duration_min = self.statistics['duration_sec'] / 60.0
        frequency_per_min = scratch_count / total_duration_min if total_duration_min > 0 else 0.0
        
        return {
            'total_count': scratch_count,
            'total_duration_sec': total_scratch_time,
            'mean_duration_sec': mean_scratch_duration,
            'frequency_per_min': frequency_per_min,
            'durations': scratch_durations
        }
    
    def _draw_keypoints(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw keypoints and skeleton on frame."""
        # Get keypoints for this frame
        if frame_idx >= len(self.keypoints_data):
            return frame
        
        frame_data = self.keypoints_data[frame_idx]
        
        bodyparts_list = frame_data.get('bodyparts', [])
        bboxes_list = frame_data.get('bboxes', [])
        bbox_scores_list = frame_data.get('bbox_scores', [])
        
        for idx in range(len(bodyparts_list)):
            if idx < len(bboxes_list):
                bbox = bboxes_list[idx]
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.COLORS['bbox'], 2)
                
                if idx < len(bbox_scores_list):
                    bbox_score = bbox_scores_list[idx]
                    cv2.putText(frame, f"{bbox_score:.2f}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['bbox'], 1)
            
            keypoints = bodyparts_list[idx]
            if not keypoints:
                continue
            
            # Convert keypoints to array for easier processing
            kp_array = []
            for kp in keypoints:
                if len(kp) >= 3:
                    kp_array.append((int(kp[0]), int(kp[1]), kp[2]))
                else:
                    kp_array.append((0, 0, 0))
            
            # Draw skeleton connections
            # for conn in self.SKELETON_CONNECTIONS:
            #     if conn[0] < len(kp_array) and conn[1] < len(kp_array):
            #         pt1 = kp_array[conn[0]]
            #         pt2 = kp_array[conn[1]]
                    
            #         # Only draw if both points have sufficient confidence
            #         if pt1[2] > 0.3 and pt2[2] > 0.3:
            #             cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]),
            #                     self.COLORS['skeleton'], 2)
            
            # Draw keypoints
            for kp in kp_array:
                if kp[2] > 0.3:  # Confidence threshold
                    # Green keypoint with white outline
                    cv2.circle(frame, (kp[0], kp[1]), 4, self.COLORS['keypoint'], -1)  # Green fill
                    cv2.circle(frame, (kp[0], kp[1]), 4, (255, 255, 255), 1)  # White outline
        
        return frame
    
    def _draw_action_label(self, frame: np.ndarray, prediction: int, 
                          frame_idx: int) -> np.ndarray:
        """Draw action prediction label on frame."""
        # Map prediction to label
        label_map = {0: 'no behavior', 1: 'scratching'}
        label = label_map.get(prediction, 'unknown')
        
        # Choose background color based on action
        bg_color = self.COLORS['scratching_bg'] if prediction == 1 else self.COLORS['no_behavior_bg']
        
        # Draw label background - using same font style and size as statistics
        text = f"Action: {label}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (10, 10), (20 + text_w, 30 + text_h), 
                     bg_color, -1)
        
        # Draw label text in black - same font as statistics panel
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLORS['label_text'], 1)
        
        return frame
    
    def _calculate_current_statistics(self, current_prediction_index: int, fps: float) -> Dict:
        """Calculate statistics up to the current frame."""
        predictions_so_far = self.predictions[:current_prediction_index + 1]
        
        scratch_durations = []
        scratch_count = 0
        total_scratch_frames = 0
        
        # Find scratching segments in predictions so far
        in_scratch = False
        scratch_start = 0
        
        for i, pred in enumerate(predictions_so_far):
            if pred == 1:  # scratching
                if not in_scratch:
                    in_scratch = True
                    scratch_start = i
                total_scratch_frames += 1
            else:  # no behavior
                if in_scratch:
                    # End of a scratching segment
                    duration = (i - scratch_start) / fps
                    scratch_durations.append(duration)
                    scratch_count += 1
                    in_scratch = False
        
        # Handle case where video ends during scratching
        if in_scratch:
            duration = (len(predictions_so_far) - scratch_start) / fps
            scratch_durations.append(duration)
            scratch_count += 1
        
        # Calculate statistics
        total_scratch_time = total_scratch_frames / fps
        mean_scratch_duration = np.mean(scratch_durations) if scratch_durations else 0.0
        total_duration_sec = len(predictions_so_far) / fps
        frequency_per_min = (scratch_count / total_duration_sec * 60.0) if total_duration_sec > 0 else 0.0
        
        return {
            'total_count': scratch_count,
            'total_duration_sec': total_scratch_time,
            'mean_duration_sec': mean_scratch_duration,
            'frequency_per_min': frequency_per_min
        }
    
    def _draw_statistics(self, frame: np.ndarray, frame_idx: int, 
                        current_time_sec: float, relative_frame: int, relative_time: float,
                        current_stats: Dict) -> np.ndarray:
        """Draw statistics panel on frame."""
        h, w = frame.shape[:2]
        panel_x = w - 400
        panel_y = 10
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + 200),
                     self.COLORS['text_bg'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw statistics text
        y_offset = panel_y + 25
        line_height = 25
        
        texts = [
            "=== Scratching Statistics ===",
            f"Total Count: {current_stats['total_count']}",
            f"Total Duration: {current_stats['total_duration_sec']:.2f}s",
            f"Mean Duration: {current_stats['mean_duration_sec']:.2f}s",
            f"Frequency: {current_stats['frequency_per_min']:.2f}/min",
            f"",
            f"Current Time: {relative_time:.2f}s",
            f"Frame: {relative_frame}"
        ]
        
        for text in texts:
            cv2.putText(frame, text, (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
            y_offset += line_height
        
        return frame
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS.mmm format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"
    
    def process_video(self):
        """Process video and create visualization."""
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # The frame_offset is where predictions start in the original video
        # start_time is additional offset in seconds FROM the beginning of predictions
        frame_offset = self.statistics.get('frame_offset', 0)
        additional_frames = int(self.start_time * fps)
        actual_start = frame_offset + additional_frames
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Predictions start at frame: {frame_offset}")
        print(f"  Additional offset: {self.start_time:.2f}s ({additional_frames} frames)")
        print(f"  Actual start frame: {actual_start}")
        print(f"  Predictions: {len(self.predictions)} frames")
        
        # Validate frame position
        if actual_start >= total_frames:
            raise RuntimeError(f"Start position ({actual_start}) exceeds video length ({total_frames} frames)")
        
        if additional_frames >= len(self.predictions):
            raise RuntimeError(f"Start time offset ({additional_frames} frames) exceeds prediction length ({len(self.predictions)} frames)")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        
        # Create video writer with more compatible codec
        # Try different codecs in order of preference
        output_path_str = str(self.output_path)
        
        # Determine codec based on file extension
        ext = self.output_path.suffix.lower()
        if ext == '.avi':
            # MJPG is most reliable for AVI, though creates larger files
            fourcc_options = [
                ('MJPG', 'Motion JPEG'),
                ('XVID', 'Xvid'),
                ('DIVX', 'DivX')
            ]
        else:  # .mp4 or other
            fourcc_options = [
                ('mp4v', 'MPEG-4'),
                ('avc1', 'H.264'),
                ('X264', 'x264'),
                ('MJPG', 'Motion JPEG')
            ]
        
        out = None
        for fourcc_code, codec_name in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                out = cv2.VideoWriter(output_path_str, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Using codec: {codec_name} ({fourcc_code})")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"Failed to use {codec_name}: {e}")
                continue
        
        if out is None or not out.isOpened():
            raise RuntimeError("Failed to initialize video writer with any available codec. Try changing output file extension to .avi")
        
        print(f"\nProcessing video...")
        print(f"Output: {self.output_path}")
        
        # Process frames starting from the offset in predictions
        num_predictions = len(self.predictions) - additional_frames
        for i in range(num_predictions):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {i}")
                break
            
            # Calculate current time
            current_frame = actual_start + i
            current_time = current_frame / fps
            
            # Calculate relative time/frame from predictions start
            relative_frame = additional_frames + i
            relative_time = relative_frame / fps
            
            # Get prediction for this frame (offset by additional_frames)
            prediction = self.predictions[additional_frames + i]
            
            # Calculate current statistics up to this frame
            current_stats = self._calculate_current_statistics(additional_frames + i, fps)
            
            # Draw keypoints (use the actual frame index in the video)
            frame = self._draw_keypoints(frame, current_frame)
            
            # Draw action label
            frame = self._draw_action_label(frame, prediction, i)
            
            # Draw statistics
            frame = self._draw_statistics(frame, current_frame, current_time, relative_frame, relative_time, current_stats)
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if (i + 1) % 100 == 0:
                progress = (i + 1) / num_predictions * 100
                print(f"  Progress: {i+1}/{num_predictions} ({progress:.1f}%)")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Verify output file was created
        if not self.output_path.exists():
            raise RuntimeError(f"Output file was not created: {self.output_path}")
        
        file_size = self.output_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            raise RuntimeError(f"Output file is too small ({file_size} bytes). Video encoding may have failed.")
        
        print(f"\nVideo processing complete!")
        print(f"Output saved to: {self.output_path}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        print(f"\nSummary:")
        print(f"  Total scratching events: {self.scratch_stats['total_count']}")
        print(f"  Total scratching time: {self.scratch_stats['total_duration_sec']:.2f}s")
        print(f"  Mean scratching duration: {self.scratch_stats['mean_duration_sec']:.2f}s")
        print(f"  Scratching frequency: {self.scratch_stats['frequency_per_min']:.2f} events/min")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize behavior predictions and keypoints on video"
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--inference-dir',
        type=str,
        required=True,
        help='Directory containing predictions.npy and statistics.json'
    )
    
    parser.add_argument(
        '--keypoint-file',
        type=str,
        required=True,
        help='Path to keypoint detection JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output video file'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = VideoVisualizer(
        video_path=args.video,
        inference_dir=args.inference_dir,
        keypoint_file=args.keypoint_file,
        output_path=args.output,
        start_time=0.0  # Always start from beginning of predictions
    )
    
    # Process video
    visualizer.process_video()


if __name__ == '__main__':
    main()
