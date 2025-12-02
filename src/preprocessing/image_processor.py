"""
Image preprocessing pipeline for handwriting OCR.
Includes deskewing, denoising, binarization, and layout analysis.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from PIL import Image
import albumentations as A
from scipy import ndimage


class ImagePreprocessor:
    """Preprocessor for handwritten document images."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration dict
        """
        self.config = config or {}
        self.deskew_enabled = self.config.get('deskew', True)
        self.denoise_enabled = self.config.get('denoise', True)
        self.binarization = self.config.get('binarization', 'adaptive')
        self.contrast_enabled = self.config.get('contrast_enhancement', True)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Denoise
        if self.denoise_enabled:
            gray = self.denoise(gray)

        # Deskew
        if self.deskew_enabled:
            gray = self.deskew(gray)

        # Enhance contrast
        if self.contrast_enabled:
            gray = self.enhance_contrast(gray)

        # Binarize
        if self.binarization != 'none':
            binary = self.binarize(gray)
        else:
            binary = gray

        # Remove borders
        if self.config.get('remove_borders', True):
            binary = self.remove_borders(binary)

        return binary

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.

        Args:
            image: Grayscale image

        Returns:
            Denoised image
        """
        # Non-local means denoising (good for handwriting)
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in document image.

        Args:
            image: Grayscale image

        Returns:
            Deskewed image
        """
        # Binarize for skew detection
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find coordinates of all non-zero points
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) == 0:
            return image

        # Calculate minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # Correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate image
        if abs(angle) < 0.5:  # Skip if almost straight
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.

        Args:
            image: Grayscale image

        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize image using specified method.

        Args:
            image: Grayscale image

        Returns:
            Binary image
        """
        if self.binarization == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif self.binarization == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 10
            )

        elif self.binarization == 'sauvola':
            # Sauvola binarization (good for handwriting)
            binary = self._sauvola_threshold(image)

        else:
            binary = image

        return binary

    def _sauvola_threshold(self, image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
        """
        Apply Sauvola local thresholding.

        Args:
            image: Grayscale image
            window_size: Size of local window
            k: Sauvola parameter

        Returns:
            Binary image
        """
        mean = cv2.boxFilter(image, cv2.CV_32F, (window_size, window_size))
        mean_sq = cv2.boxFilter(image ** 2, cv2.CV_32F, (window_size, window_size))
        std = np.sqrt(mean_sq - mean ** 2)

        threshold = mean * (1 + k * (std / 128 - 1))
        binary = (image > threshold).astype(np.uint8) * 255

        return binary

    def remove_borders(self, image: np.ndarray, border_size: int = 5) -> np.ndarray:
        """
        Remove border artifacts.

        Args:
            image: Binary image
            border_size: Size of border to remove

        Returns:
            Image with borders removed
        """
        h, w = image.shape[:2]
        return image[border_size:h-border_size, border_size:w-border_size]


class LayoutDetector:
    """Detect text regions and layout in document images."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize layout detector.

        Args:
            config: Configuration dict
        """
        self.config = config or {}
        self.min_height = self.config.get('min_text_height', 10)
        self.min_width = self.config.get('min_text_width', 10)

    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in image.

        Args:
            image: Binary image

        Returns:
            List of region dicts with 'bbox', 'type' keys
        """
        # Find contours
        contours, _ = cv2.findContours(
            255 - image if image.mean() > 127 else image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if w < self.min_width or h < self.min_height:
                continue

            # Classify region type based on aspect ratio
            aspect_ratio = w / h
            if aspect_ratio > 5:
                region_type = 'line'
            elif aspect_ratio > 1.5:
                region_type = 'text_block'
            else:
                region_type = 'other'

            regions.append({
                'bbox': (x, y, w, h),
                'type': region_type,
                'area': w * h
            })

        # Sort by vertical position
        regions = sorted(regions, key=lambda r: r['bbox'][1])

        return regions

    def segment_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Segment image into text lines.

        Args:
            image: Binary image

        Returns:
            List of line bounding boxes (x, y, w, h)
        """
        # Horizontal projection profile
        h_projection = np.sum(255 - image if image.mean() > 127 else image, axis=1)

        # Find line separators (valleys in projection)
        threshold = np.mean(h_projection) * 0.1
        in_line = False
        lines = []
        start_y = 0

        for y, val in enumerate(h_projection):
            if val > threshold and not in_line:
                start_y = y
                in_line = True
            elif val <= threshold and in_line:
                # End of line
                if y - start_y > self.min_height:
                    lines.append((0, start_y, image.shape[1], y - start_y))
                in_line = False

        # Handle last line
        if in_line and image.shape[0] - start_y > self.min_height:
            lines.append((0, start_y, image.shape[1], image.shape[0] - start_y))

        return lines


class DataAugmenter:
    """Data augmentation for handwriting images."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config or {}
        self.prob = self.config.get('augmentation_prob', 0.7)

        # Define augmentation pipeline
        self.transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.5),
            ], p=0.3),

            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(p=0.3),
            ], p=0.2),

            A.Rotate(limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.3),

            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                shear=(-2, 2),
                p=0.2
            ),
        ])

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to image.

        Args:
            image: Input image

        Returns:
            Augmented image
        """
        if len(image.shape) == 2:
            # Add channel dimension for albumentation
            image = np.expand_dims(image, -1)
            augmented = self.transform(image=image)['image']
            return augmented.squeeze()
        else:
            return self.transform(image=image)['image']


def process_image_for_ocr(
    image_path: str,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Complete preprocessing pipeline for OCR.

    Args:
        image_path: Path to input image
        config: Processing configuration

    Returns:
        Tuple of (preprocessed image, detected regions)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Preprocess
    preprocessor = ImagePreprocessor(config)
    processed = preprocessor.process(image)

    # Detect layout
    detector = LayoutDetector(config)
    regions = detector.detect_text_regions(processed)

    return processed, regions


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        processed, regions = process_image_for_ocr(image_path)

        print(f"Processed image shape: {processed.shape}")
        print(f"Detected {len(regions)} text regions:")
        for i, region in enumerate(regions[:5]):  # Show first 5
            print(f"  Region {i}: {region}")

        # Save result
        output_path = image_path.replace('.', '_processed.')
        cv2.imwrite(output_path, processed)
        print(f"Saved preprocessed image to: {output_path}")
    else:
        print("Usage: python image_processor.py <image_path>")
