import os
import sys
import torch
import argparse
from PIL import Image

# Ensure local llava modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import cv2


class FastVLM:
    def __init__(self, model_path, device="cuda"):
        # Based on predict.py logic
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)

        # Load model using the builder
        # Note: predict.py accesses args.model_base, assuming None/default behavior if not provided.
        # It also hardcodes device="mps" in the viewed file, but we should respect the 'device' arg or use cuda if available/requested.
        # The existing predict.py said `device="mps"`, but likely we want "cuda" for a server unless on Mac.
        # Given the container environment, I'll pass the requested device, but load_pretrained_model might handle it specific ways.
        # Looking at predict.py line 31: load_pretrained_model(..., device="mps")
        # I will change "mps" to the passed device.

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base=None, model_name=model_name, device=device
        )

        # Default conversation mode from predict.py args
        self.conv_mode = "qwen_2"

        self.device = device

    def generate_form_image(self, images, user_prompt, system_prompt):
        """
        Generates text based on images and prompt.
        Args:
            images: List of PIL images (or BGR arrays which we convert).
            user_prompt: content input from user.
            system_prompt: (Optional) system instruction.
        """
        try:
            # Handle image input - API seems to pass list of PIL/numpy
            # FastVLM/LLaVA usually handles one image or multiple.
            # The predict.py example handles a single image.
            # Qwen implementation handles multiple but primarily uses the first one in simple calls or constructs a list.
            # We will take the first image if multiple are provided, as basic LLaVA usage often focuses on single image context.
            # Or we can check if the model supports multi-image. For now, let's assume single image for safety unless we loop.

            if not images:
                return None

            frame = images[0]
            if isinstance(frame, Image.Image):
                image = frame.convert("RGB")
            else:
                # Assume numpy BGR
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Construct prompt
            qs = user_prompt
            if system_prompt:
                # Prepend system prompt if useful, or just append to user prompt?
                # LLaVA conversation templates might handle system prompts differently.
                # qwen_2 template might have a system slot.
                # For now simplify: just use user_prompt as main query.
                pass

            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Set pad token
            if self.model.generation_config.pad_token_id is None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

            # Tokenize
            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(self.device)
            )

            # Process Image
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]

            # Inference
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().to(self.device),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=256,
                    use_cache=True,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs

        except Exception as e:
            print(f"Error in FastVLM generate_form_image: {e}")
            import traceback

            traceback.print_exc()
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/app/vlms/fastvlm/checkpoints/llava-fastvithd_1.5b_stage3")
    parser.add_argument("--image-file", type=str, required=True, help="Path to the image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="User prompt")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path} on {args.device}...")
    vlm = FastVLM(model_path=args.model_path, device=args.device)

    print(f"Loading image from {args.image_file}...")
    try:
        image = Image.open(args.image_file)

        print("Generating response...")
        response = vlm.generate_form_image(images=[image], user_prompt=args.prompt, system_prompt=None)
        print("-" * 20)
        print("Output:", response)
        print("-" * 20)
    except Exception as e:
        print(f"Error during verification: {e}")
