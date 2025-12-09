from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class ChangeDetection:
    """Data class for storing detected changes"""
    change_type: str
    description: str
    confidence: float
    severity: str  # 'low', 'medium', 'high'
    location: Tuple[int, int] = None
    affected_area_percentage: float = 0.0


@dataclass
class AnalysisReport:
    """Complete analysis report structure"""
    timestamp: str
    overall_classification: str  # 'NORMAL' or 'ABNORMAL'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence_score: float
    total_changes_detected: int
    semantic_changes: List[ChangeDetection]
    structural_changes: List[ChangeDetection]
    visual_changes: List[ChangeDetection]
    quality_assessment: Dict[str, Any]
    warnings: List[str]
    reasoning: str


class MultiLayerCCTVAnalyzer:
    """
    Comprehensive CCTV image comparison system with multi-layer analysis
    """
    
    def __init__(self, yolo_model='yolov8n.pt'):
        """Initialize all detection modules"""
        print("Initializing Multi-Layer CCTV Analysis System...")
        
        
        self.yolo_model = YOLO(yolo_model)
        
        
        self.yolo_confidence = 0.25
        self.blur_threshold = 100
        self.brightness_change_threshold = 30
        self.structural_change_threshold = 0.15
        
        print("✓ System initialized successfully\n")
    
    # ==================== LAYER 1: SEMANTIC ANALYSIS ====================
    
    def semantic_analysis(self, img1_path: str, img2_path: str) -> List[ChangeDetection]:
        """
        Detect high-level object changes using YOLO
        - New objects appeared
        - Objects disappeared
        - Object movement
        """
        changes = []
        
       
        objects1 = self._detect_objects(img1_path)
        objects2 = self._detect_objects(img2_path)
        
      
        new_objects = self._find_new_objects(objects1, objects2)
        for obj in new_objects:
            changes.append(ChangeDetection(
                change_type='OBJECT_APPEARED',
                description=f"New {obj['class_name']} detected",
                confidence=obj['confidence'],
                severity='high' if obj['class_name'] in ['person', 'car', 'truck'] else 'medium',
                location=(int(obj['center'][0]), int(obj['center'][1])),
                affected_area_percentage=self._calculate_bbox_area_percentage(obj['bbox'], img2_path)
            ))
        
       
        disappeared_objects = self._find_new_objects(objects2, objects1)
        for obj in disappeared_objects:
            changes.append(ChangeDetection(
                change_type='OBJECT_DISAPPEARED',
                description=f"{obj['class_name']} is no longer visible",
                confidence=obj['confidence'],
                severity='medium',
                location=(int(obj['center'][0]), int(obj['center'][1])),
                affected_area_percentage=self._calculate_bbox_area_percentage(obj['bbox'], img1_path)
            ))
        
        
        movement_changes = self._detect_object_movement(objects1, objects2)
        changes.extend(movement_changes)
        
        return changes
    
    def _detect_objects(self, image_path: str) -> List[Dict]:
        """Run YOLO detection"""
        results = self.yolo_model(image_path, conf=self.yolo_confidence, verbose=False)
        
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                obj = {
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': [float(x) for x in bbox.tolist()],
                    'center': (float(center[0]), float(center[1]))
                }
                detected_objects.append(obj)
        
        return detected_objects
    
    def _find_new_objects(self, objects1: List[Dict], objects2: List[Dict]) -> List[Dict]:
        """Find objects in objects2 that are not in objects1"""
        new_objects = []
        for obj2 in objects2:
            is_new = True
            for obj1 in objects1:
                if self._objects_match(obj1, obj2):
                    is_new = False
                    break
            if is_new:
                new_objects.append(obj2)
        return new_objects
    
    def _objects_match(self, obj1: Dict, obj2: Dict, distance_threshold: float = 100) -> bool:
        """Check if two objects are the same (same class, close position)"""
        if obj1['class_name'] != obj2['class_name']:
            return False
        
        x1, y1 = obj1['center']
        x2, y2 = obj2['center']
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance < distance_threshold
    
    def _detect_object_movement(self, objects1: List[Dict], objects2: List[Dict]) -> List[ChangeDetection]:
        """Detect if same objects moved between frames"""
        changes = []
        
        for obj1 in objects1:
            for obj2 in objects2:
                if obj1['class_name'] == obj2['class_name']:
                    x1, y1 = obj1['center']
                    x2, y2 = obj2['center']
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if 50 < distance < 150: 
                        changes.append(ChangeDetection(
                            change_type='OBJECT_MOVEMENT',
                            description=f"{obj1['class_name']} moved {int(distance)} pixels",
                            confidence=min(obj1['confidence'], obj2['confidence']),
                            severity='medium',
                            location=(int(x2), int(y2)),
                            affected_area_percentage=0.0
                        ))
                        break
        
        return changes
    
    def _calculate_bbox_area_percentage(self, bbox: List[float], image_path: str) -> float:
        """Calculate what percentage of image the bounding box covers"""
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_width * img_height
        
        return (bbox_area / img_area) * 100
    
    # ==================== LAYER 2: STRUCTURAL ANALYSIS ====================
    
    def structural_analysis(self, img1_path: str, img2_path: str) -> List[ChangeDetection]:
        """
        Detect structural/geometric changes
        - Edge changes
        - Contour modifications
        - Shape differences
        """
        changes = []
        
      
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
       
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        
        edge_diff = cv2.absdiff(edges1, edges2)
        structural_change_percentage = (np.count_nonzero(edge_diff) / edge_diff.size) * 100
        
        if structural_change_percentage > self.structural_change_threshold * 100:
            changes.append(ChangeDetection(
                change_type='STRUCTURAL_CHANGE',
                description=f"Significant structural changes detected ({structural_change_percentage:.1f}% of frame)",
                confidence=min(structural_change_percentage / 50, 1.0),
                severity='high' if structural_change_percentage > 30 else 'medium',
                affected_area_percentage=structural_change_percentage
            ))
        
        
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
        
        if len(significant_contours) > 0:
            total_change_area = sum(cv2.contourArea(c) for c in significant_contours)
            change_percentage = (total_change_area / (gray1.shape[0] * gray1.shape[1])) * 100
            
            changes.append(ChangeDetection(
                change_type='MOTION_DETECTED',
                description=f"Motion detected in {len(significant_contours)} regions ({change_percentage:.1f}% of frame)",
                confidence=0.8,
                severity='medium',
                affected_area_percentage=change_percentage
            ))
        
        return changes
    
    # ==================== LAYER 3: VISUAL ANALYSIS ====================
    
    def visual_analysis(self, img1_path: str, img2_path: str) -> List[ChangeDetection]:
        """
        Detect low-level visual changes
        - Brightness/lighting changes
        - Color shifts
        - Blur/quality degradation
        """
        changes = []
        
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Brightness analysis
        brightness_change = self._analyze_brightness_change(img1, img2_resized)
        if brightness_change:
            changes.append(brightness_change)
        
        # Color shift analysis
        color_changes = self._analyze_color_shift(img1, img2_resized)
        changes.extend(color_changes)
        
        # Blur/quality analysis
        blur_change = self._analyze_blur_change(img1_path, img2_path)
        if blur_change:
            changes.append(blur_change)
        
        return changes
    
    def _analyze_brightness_change(self, img1: np.ndarray, img2: np.ndarray) -> ChangeDetection:
        """Detect overall brightness/lighting changes"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        mean1 = np.mean(gray1)
        mean2 = np.mean(gray2)
        
        diff = abs(mean2 - mean1)
        
        if diff > self.brightness_change_threshold:
            direction = "brighter" if mean2 > mean1 else "darker"
            return ChangeDetection(
                change_type='LIGHTING_CHANGE',
                description=f"Scene became significantly {direction} (Δ{diff:.1f})",
                confidence=min(diff / 100, 1.0),
                severity='low',
                affected_area_percentage=100.0
            )
        return None
    
    def _analyze_color_shift(self, img1: np.ndarray, img2: np.ndarray) -> List[ChangeDetection]:
        """Detect color distribution changes"""
        changes = []
        
        # Analyze each color channel
        channels = ['Blue', 'Green', 'Red']
        for i, channel_name in enumerate(channels):
            mean1 = np.mean(img1[:, :, i])
            mean2 = np.mean(img2[:, :, i])
            diff = abs(mean2 - mean1)
            
            if diff > 20:
                changes.append(ChangeDetection(
                    change_type='COLOR_SHIFT',
                    description=f"{channel_name} channel shifted by {diff:.1f}",
                    confidence=0.7,
                    severity='low',
                    affected_area_percentage=100.0
                ))
        
        return changes
    
    def _analyze_blur_change(self, img1_path: str, img2_path: str) -> ChangeDetection:
        """Detect blur or quality degradation"""
        blur1 = self._calculate_blur(img1_path)
        blur2 = self._calculate_blur(img2_path)
        
        diff = abs(blur2 - blur1)
        
        if diff > 50 or blur2 < self.blur_threshold:
            quality = "blurry" if blur2 < blur1 else "sharper"
            return ChangeDetection(
                change_type='QUALITY_CHANGE',
                description=f"Image 2 is {quality} (blur score: {blur2:.1f})",
                confidence=0.8,
                severity='low',
                affected_area_percentage=100.0
            )
        return None
    
    def _calculate_blur(self, image_path: str) -> float:
        """Calculate blur score using Laplacian variance"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var
    
    # ==================== IMAGE QUALITY ASSESSMENT ====================
    
    def assess_image_quality(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """
        Assess quality of both images and detect corruption/issues
        """
        quality = {
            'image1': self._assess_single_image(img1_path),
            'image2': self._assess_single_image(img2_path),
            'comparable': True,
            'warnings': []
        }
        
        # Check if images are comparable
        if quality['image1']['corrupted'] or quality['image2']['corrupted']:
            quality['comparable'] = False
            quality['warnings'].append("One or both images appear corrupted")
        
        if abs(quality['image1']['aspect_ratio'] - quality['image2']['aspect_ratio']) > 0.2:
            quality['warnings'].append("Significant aspect ratio difference detected")
        
        return quality
    
    def _assess_single_image(self, image_path: str) -> Dict[str, Any]:
        """Assess quality of a single image"""
        img = cv2.imread(image_path)
        
        if img is None:
            return {'corrupted': True, 'error': 'Cannot read image'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return {
            'corrupted': False,
            'width': img.shape[1],
            'height': img.shape[0],
            'aspect_ratio': img.shape[1] / img.shape[0],
            'blur_score': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'mean_brightness': np.mean(gray),
            'is_blurry': cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold,
            'is_too_dark': np.mean(gray) < 50,
            'is_too_bright': np.mean(gray) > 200
        }
    
    # ==================== MAIN ANALYSIS PIPELINE ====================
    
    def analyze(self, img1_path: str, img2_path: str) -> AnalysisReport:
        """
        Complete multi-layer analysis pipeline
        """
        print("="*70)
        print("MULTI-LAYER CCTV CHANGE ANALYSIS")
        print("="*70)
        
       
        print("\n[1/5] Assessing image quality...")
        quality = self.assess_image_quality(img1_path, img2_path)
        
        warnings = quality['warnings'].copy()
        
        if not quality['comparable']:
            return self._create_error_report(quality, warnings)
        
       
        print("[2/5] Performing semantic analysis (object detection)...")
        semantic_changes = self.semantic_analysis(img1_path, img2_path)
        
        
        print("[3/5] Performing structural analysis (edges, contours)...")
        structural_changes = self.structural_analysis(img1_path, img2_path)
        
       
        print("[4/5] Performing visual analysis (lighting, color, blur)...")
        visual_changes = self.visual_analysis(img1_path, img2_path)
        
       
        print("[5/5] Generating final report...")
        report = self._generate_final_report(
            semantic_changes,
            structural_changes,
            visual_changes,
            quality,
            warnings
        )
        
        return report
    
    def _generate_final_report(self, semantic, structural, visual, quality, warnings) -> AnalysisReport:
        """Generate final analysis report with classification"""
        
        # Calculate total changes
        all_changes = semantic + structural + visual
        total_changes = len(all_changes)
        
        # Classify as NORMAL or ABNORMAL
        high_severity_changes = [c for c in all_changes if c.severity == 'high']
        critical_objects = ['person', 'car', 'truck']
        
        has_critical_object_change = any(
            c.change_type in ['OBJECT_APPEARED', 'OBJECT_DISAPPEARED'] and
            any(obj in c.description.lower() for obj in critical_objects)
            for c in semantic
        )
        
        
        if has_critical_object_change:
            classification = 'ABNORMAL'
            risk_level = 'HIGH'
        elif len(high_severity_changes) >= 2:
            classification = 'ABNORMAL'
            risk_level = 'MEDIUM'
        elif total_changes > 5:
            classification = 'ABNORMAL'
            risk_level = 'MEDIUM'
        elif total_changes > 2:
            classification = 'NORMAL'
            risk_level = 'LOW'
        else:
            classification = 'NORMAL'
            risk_level = 'LOW'
        
       
        avg_confidence = np.mean([c.confidence for c in all_changes]) if all_changes else 0.0
        confidence_score = avg_confidence * (1.0 - len(warnings) * 0.1)  # Reduce for warnings
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        
        reasoning = self._generate_reasoning(semantic, structural, visual, classification)
        
        return AnalysisReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_classification=classification,
            risk_level=risk_level,
            confidence_score=confidence_score,
            total_changes_detected=total_changes,
            semantic_changes=semantic,
            structural_changes=structural,
            visual_changes=visual,
            quality_assessment=quality,
            warnings=warnings,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, semantic, structural, visual, classification) -> str:
        """Generate human-readable reasoning"""
        lines = []
        
        if classification == 'ABNORMAL':
            lines.append("Classification: ABNORMAL")
            lines.append("Reasoning:")
            
            if semantic:
                lines.append(f"  • {len(semantic)} semantic changes detected (objects appeared/disappeared/moved)")
            if structural:
                lines.append(f"  • {len(structural)} structural changes detected (scene geometry altered)")
            if visual:
                lines.append(f"  • {len(visual)} visual changes detected (lighting, color, quality)")
            
            critical_changes = [c for c in semantic if c.severity == 'high']
            if critical_changes:
                lines.append(f"  • {len(critical_changes)} high-severity changes require attention")
        else:
            lines.append("Classification: NORMAL")
            lines.append("Reasoning:")
            lines.append("  • Changes detected are within expected parameters")
            lines.append("  • No critical object movements or appearances")
            if visual:
                lines.append("  • Minor environmental changes only (lighting/quality)")
        
        return "\n".join(lines)
    
    def _create_error_report(self, quality, warnings) -> AnalysisReport:
        """Create report for corrupted/unreadable images"""
        return AnalysisReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_classification='ERROR',
            risk_level='UNKNOWN',
            confidence_score=0.0,
            total_changes_detected=0,
            semantic_changes=[],
            structural_changes=[],
            visual_changes=[],
            quality_assessment=quality,
            warnings=warnings,
            reasoning="Unable to perform analysis due to image quality issues"
        )
    
    def print_report(self, report: AnalysisReport):
        """Print formatted analysis report"""
        print("\n" + "="*70)
        print("ANALYSIS REPORT")
        print("="*70)
        print(f"Timestamp: {report.timestamp}")
        print(f"Classification: {report.overall_classification}")
        print(f"Risk Level: {report.risk_level}")
        print(f"Confidence Score: {report.confidence_score:.2%}")
        print(f"Total Changes Detected: {report.total_changes_detected}")
        
        if report.warnings:
            print(f"\n⚠️  WARNINGS ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  • {warning}")
        
        if report.semantic_changes:
            print(f"\n📦 SEMANTIC CHANGES ({len(report.semantic_changes)}):")
            for change in report.semantic_changes:
                print(f"  • [{change.severity.upper()}] {change.description}")
                print(f"    Confidence: {change.confidence:.2%}")
        
        if report.structural_changes:
            print(f"\n🔧 STRUCTURAL CHANGES ({len(report.structural_changes)}):")
            for change in report.structural_changes:
                print(f"  • [{change.severity.upper()}] {change.description}")
        
        if report.visual_changes:
            print(f"\n🎨 VISUAL CHANGES ({len(report.visual_changes)}):")
            for change in report.visual_changes:
                print(f"  • [{change.severity.upper()}] {change.description}")
        
        print(f"\n💡 REASONING:")
        print(report.reasoning)
        
        print("\n" + "="*70)
    
    def save_report_json(self, report: AnalysisReport, output_path: str = 'analysis_report.json'):
        """Save report to JSON file"""
        
        def convert_to_json_serializable(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
       
        report_dict = {
            'timestamp': report.timestamp,
            'overall_classification': report.overall_classification,
            'risk_level': report.risk_level,
            'confidence_score': report.confidence_score,
            'total_changes_detected': report.total_changes_detected,
            'semantic_changes': [asdict(c) for c in report.semantic_changes],
            'structural_changes': [asdict(c) for c in report.structural_changes],
            'visual_changes': [asdict(c) for c in report.visual_changes],
            'quality_assessment': report.quality_assessment,
            'warnings': report.warnings,
            'reasoning': report.reasoning
        }
        
       
        report_dict = convert_to_json_serializable(report_dict)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"✓ JSON report saved to: {output_path}")

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution"""
    
    
    analyzer = MultiLayerCCTVAnalyzer(yolo_model='yolov8n.pt')
    img1_path = 'reference.jpg'
    img2_path = 'current.jpg'
    report = analyzer.analyze(img1_path, img2_path)  
    analyzer.print_report(report)
    analyzer.save_report_json(report)
    print("\n✅ Analysis complete!")
    return report


if __name__ == "__main__":
    report = main()