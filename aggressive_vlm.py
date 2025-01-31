
'''pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate

pip install qwen-vl-utils
'''

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# model = Qwen2VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
# )

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "qwen2vl_2b_instruct_lora_merged", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

output_path = "test"

def query_video(prompt,frames=[], video_path=None):
    if frames:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "max_pixels": 256 * 256,
                        "fps": 3,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"{video_path}",
                            "max_pixels": 256 * 256,
                            "fps": 2,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        print(f"Using  entire video for inference.")

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.no_grad():  # Use no_grad to save memory during inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    torch.cuda.empty_cache()
    return output_text
    




# query_video("Is there any aggressive scene in this video like fighting or any demonstration of aggresive behaviour?",frames=[],
#             video_path="/home/jay/SSD2000/WORK/STREAM_VISION/Action-Recongnition/fight.mp4")