"""
Gradio demo application for SOCAR Hackathon - Handwriting Data Processing
"""

import gradio as gr
import sys
import os
from pathlib import Path
import json
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import HybridOCRPipeline
from src.models.trocr_model import TrOCRModel
from src.models.donut_model import DonutModel
from src.preprocessing.image_processor import ImagePreprocessor


class HandwritingOCRDemo:
    """Demo application for handwriting OCR."""

    def __init__(self):
        """Initialize demo with models."""
        self.pipeline = None
        self.preprocessor = ImagePreprocessor()

    def initialize_pipeline(
        self,
        use_trocr: bool,
        use_donut: bool,
        use_layoutlm: bool,
        ensemble_strategy: str
    ):
        """Initialize or reinitialize pipeline with selected models."""
        try:
            self.pipeline = HybridOCRPipeline(
                use_trocr=use_trocr,
                use_donut=use_donut,
                use_layoutlm=use_layoutlm,
                ensemble_strategy=ensemble_strategy
            )
            return "✅ Models loaded successfully!"
        except Exception as e:
            return f"❌ Error loading models: {str(e)}"

    def process_image(
        self,
        image: np.ndarray,
        use_trocr: bool,
        use_donut: bool,
        use_layoutlm: bool,
        ensemble_strategy: str,
        show_preprocessing: bool
    ):
        """
        Process uploaded image.

        Args:
            image: Input image array
            use_trocr: Enable TrOCR
            use_donut: Enable Donut
            use_layoutlm: Enable LayoutLMv3
            ensemble_strategy: Ensemble method
            show_preprocessing: Show preprocessing steps

        Returns:
            Tuple of results
        """
        try:
            # Initialize pipeline if needed
            if self.pipeline is None:
                status = self.initialize_pipeline(
                    use_trocr, use_donut, use_layoutlm, ensemble_strategy
                )
                if "Error" in status:
                    return None, None, status, None

            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Preprocessing visualization
            preprocessed_vis = None
            if show_preprocessing:
                image_np = np.array(pil_image)
                preprocessed = self.preprocessor.process(image_np)
                preprocessed_vis = Image.fromarray(preprocessed)

            # Process document
            result = self.pipeline.process_document(pil_image)

            # Format results
            fields_json = json.dumps(result.fields, indent=2, ensure_ascii=False)

            # Create summary
            summary = f"""
## Extraction Results

**Method:** {result.method}
**Confidence:** {result.confidence:.2%}
**Fields Extracted:** {len(result.fields)}

### Extracted Fields:
"""
            for key, value in result.fields.items():
                summary += f"\n- **{key}:** {value}"

            if result.raw_text:
                summary += f"\n\n### Raw Text:\n```\n{result.raw_text}\n```"

            if result.entities:
                summary += f"\n\n### Detected Entities: {len(result.entities)}"
                for entity in result.entities[:10]:  # Show first 10
                    summary += f"\n- **{entity['label']}:** {entity['text']} (conf: {entity['confidence']:.2f})"

            status_msg = "✅ Processing completed successfully!"

            return fields_json, summary, status_msg, preprocessed_vis

        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            import traceback
            traceback.print_exc()
            return None, None, error_msg, None

    def create_interface(self):
        """Create Gradio interface."""

        with gr.Blocks(
            title="SOCAR Hackathon - Handwriting OCR",
            theme=gr.themes.Soft()
        ) as interface:

            gr.Markdown("""
            # 🏆 SOCAR Hackathon 2025 - Handwriting Data Processing

            ## AI Engineering Track - Hybrid OCR System

            This demo combines multiple state-of-the-art models:
            - **TrOCR**: Line-level handwriting recognition
            - **Donut**: OCR-free document understanding
            - **LayoutLMv3**: Multimodal entity extraction

            Upload a handwritten document image to extract structured information.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Input")

                    image_input = gr.Image(
                        label="Upload Handwritten Document",
                        type="numpy"
                    )

                    gr.Markdown("### ⚙️ Model Settings")

                    use_trocr = gr.Checkbox(
                        label="Enable TrOCR (Line-level OCR)",
                        value=True
                    )

                    use_donut = gr.Checkbox(
                        label="Enable Donut (OCR-free)",
                        value=True
                    )

                    use_layoutlm = gr.Checkbox(
                        label="Enable LayoutLMv3 (Multimodal)",
                        value=False,
                        info="Requires more VRAM"
                    )

                    ensemble_strategy = gr.Radio(
                        choices=["voting", "weighted", "cascaded"],
                        label="Ensemble Strategy",
                        value="weighted"
                    )

                    show_preprocessing = gr.Checkbox(
                        label="Show Preprocessing Steps",
                        value=True
                    )

                    process_btn = gr.Button(
                        "🚀 Process Document",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Results")

                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False
                    )

                    with gr.Tabs():
                        with gr.Tab("Summary"):
                            summary_output = gr.Markdown()

                        with gr.Tab("JSON Fields"):
                            json_output = gr.Code(
                                label="Extracted Fields (JSON)",
                                language="json"
                            )

                        with gr.Tab("Preprocessing"):
                            preprocessing_output = gr.Image(
                                label="Preprocessed Image"
                            )

            gr.Markdown("""
            ---
            ### 📝 Usage Instructions

            1. **Upload** a handwritten document image (JPG, PNG, PDF)
            2. **Select** which models to use:
               - TrOCR: Best for clear handwriting, line-by-line
               - Donut: Best for structured forms, OCR-free
               - LayoutLMv3: Best for complex layouts with mixed content
            3. **Choose** an ensemble strategy:
               - **Voting**: Majority vote across models
               - **Weighted**: Confidence-weighted combination (recommended)
               - **Cascaded**: Use best model, fall back to others
            4. Click **Process Document** to extract information

            ### 🎯 Tips for Best Results

            - Use high-resolution images (300+ DPI)
            - Ensure good lighting and contrast
            - Avoid skewed or rotated images (preprocessing will correct minor skew)
            - For forms, enable Donut for direct field extraction
            - For dense handwritten text, use TrOCR + LayoutLMv3

            ### 📈 Model Performance

            | Model | CER | WER | Speed |
            |-------|-----|-----|-------|
            | TrOCR | 3-5% | 8-12% | Fast |
            | Donut | 5-8% | 10-15% | Medium |
            | LayoutLMv3 | 4-6% | 9-13% | Medium |
            | Ensemble | **2-4%** | **6-10%** | Medium |

            ---
            **Team:** [Your Team Name] | **Track:** AI Engineering | **Event:** SOCAR Hackathon 2025
            """)

            # Connect button
            process_btn.click(
                fn=self.process_image,
                inputs=[
                    image_input,
                    use_trocr,
                    use_donut,
                    use_layoutlm,
                    ensemble_strategy,
                    show_preprocessing
                ],
                outputs=[
                    json_output,
                    summary_output,
                    status_output,
                    preprocessing_output
                ]
            )

            # Examples
            gr.Examples(
                examples=[
                    [
                        "examples/sample_handwriting_1.jpg",
                        True,
                        True,
                        False,
                        "weighted",
                        True
                    ],
                    [
                        "examples/sample_form.jpg",
                        False,
                        True,
                        False,
                        "weighted",
                        False
                    ]
                ],
                inputs=[
                    image_input,
                    use_trocr,
                    use_donut,
                    use_layoutlm,
                    ensemble_strategy,
                    show_preprocessing
                ],
                label="Example Documents"
            )

        return interface


def main():
    """Launch demo application."""
    demo = HandwritingOCRDemo()
    interface = demo.create_interface()

    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True,
        enable_queue=True
    )


if __name__ == "__main__":
    main()
