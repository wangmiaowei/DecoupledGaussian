import torch
import imageio
# Use bfloat16 precision for operations
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Enable TensorFloat-32 on Ampere or newer GPUs for faster matrix multiplications
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import os
import argparse
import re
import numpy as np
import imageio.v2 as iio
import gradio as gr
from loguru import logger as guru
from pathlib import Path
import cv2

from sam2.build_sam import build_sam2_video_predictor

def is_image_file(filename):
    """
    Check if the file has an image extension.
    Supported formats: .png, .jpg, .jpeg, .bmp
    """
    return Path(filename).suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']

class PromptGUI:
    def __init__(self, checkpoint_dir, model_cfg, image_dir, output_dir):
        # Paths and directories
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.image_dir = str(Path(image_dir).resolve())
        self.output_dir = str(Path(output_dir).resolve())

        # Collect image paths and sort by natural numeric order
        base = Path(self.image_dir)
        all_images = [str(p) for p in base.iterdir() if is_image_file(p.name)]
        def natural_key(s):
            nums = re.findall(r"\d+", Path(s).stem)
            return [int(n) for n in nums] if nums else [0]
        self.image_paths = sorted(all_images, key=natural_key)

        # Initialize selection and state variables
        self.selected_points = []
        self.selected_labels = []
        self.current_label = 1  # 1 = positive, 0 = negative
        self.object_id = 1
        self.inference_state = None

        # Load the SAM model
        self._load_sam_model()

    def _load_sam_model(self):
        """Load the SAM video predictor model."""
        self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
        guru.info(f"Loaded SAM checkpoint: {self.checkpoint_dir}")

    def setup(self):
        """
        Initialize SAM state for the sequence and return instruction + first frame.
        """
        self.inference_state = self.sam_model.init_state(video_path=self.image_dir)
        guru.info("SAM features extracted for sequence.")
        first_frame = iio.imread(self.image_paths[0])[:, :, :3]
        instruction = (
            "Click on the image to add positive/negative points. "
            "Then click 'Submit' to propagate masks, play video, and save masks."
        )
        return instruction, first_frame

    def add_point(self, image, evt: gr.SelectData):
        """
        Record a point on the first frame and overlay current mask.
        """
        x, y = evt.index
        self.selected_points.append([x, y])
        self.selected_labels.append(self.current_label)
        pts = np.array(self.selected_points, dtype=np.float32)
        lbs = np.array(self.selected_labels, dtype=np.int32)

        # Update mask on frame 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, obj_ids, logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=self.object_id,
                points=pts,
                labels=lbs
            )
        mask = (logits[0] > 0).squeeze().cpu().numpy().astype(np.uint8)

        # Overlay mask on image
        orig = np.array(image)
        color_mask = np.zeros_like(orig)
        color_mask[mask == 1] = [0, 255, 0]
        overlay = cv2.addWeighted(orig, 0.7, color_mask, 0.3, 0)
        for (px, py), lbl in zip(self.selected_points, self.selected_labels):
            color = (0,255,0) if lbl==1 else (255,0,0)
            cv2.circle(overlay, (px, py), 5, color, -1)
        return overlay

    def set_positive_mode(self):
        """Switch to positive point selection mode."""
        self.current_label = 1
        return "Positive point mode activated."

    def set_negative_mode(self):
        """Switch to negative point selection mode."""
        self.current_label = 0
        return "Negative point mode activated."

    def submit(self):
        """
        Propagate mask across all frames, create video, save .npy masks,
        and return video path for playback.
        """
        frames = []
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        for idx, obj_ids, logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
            mask = (logits[0] > 0).squeeze().cpu().numpy().astype(np.uint8)
            # Save mask
            stem = Path(self.image_paths[idx]).stem
            np.save(os.path.join(self.output_dir, f"{stem}.npy"), mask)
            # Read original and overlay
            img = iio.imread(self.image_paths[idx])[:, :, :3]
            color_mask = np.zeros_like(img)
            color_mask[mask==1] = [0,255,0]
            overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
            frames.append(overlay)
        # Write video
        vid_path = os.path.join(self.output_dir, 'tracked.mp4')
        imageio.mimwrite(vid_path, frames, fps=30)
        
        guru.info(f"Wrote masked video to {vid_path}")
        return vid_path


def launch_demo(checkpoint, config, sequence):
    """
    Launch Gradio interface for a sequence under '../input_dataset/'.
    """
    base = Path(__file__).parent.resolve() / '..' / 'input_dataset' / sequence
    image_dir = base / 'images'
    output_dir = base / 'sam2_mask'
    gui = PromptGUI(checkpoint, config, str(image_dir), str(output_dir))

    with gr.Blocks() as demo:
        instr = gr.Textbox(interactive=False)
        img = gr.Image(interactive=True, label="First Frame")
        btn_pos = gr.Button('Positive')
        btn_neg = gr.Button('Negative')
        btn_sub = gr.Button('Submit')
        video = gr.Video(label="Masked Sequence")

        demo.load(fn=gui.setup, outputs=[instr, img])
        img.select(fn=gui.add_point, inputs=[img], outputs=[img])
        btn_pos.click(fn=gui.set_positive_mode, outputs=[instr])
        btn_neg.click(fn=gui.set_negative_mode, outputs=[instr])
        btn_sub.click(fn=gui.submit, outputs=[video])

    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SAM2-GUI on an image sequence')
    parser.add_argument('sequence', help='Sequence folder name under ../input_dataset')
    parser.add_argument('--checkpoint', default='../weights/sam2_hiera_large.pt')
    parser.add_argument('--config', default='sam2_hiera_l.yaml')
    args = parser.parse_args()
    launch_demo(args.checkpoint, args.config, args.sequence)
