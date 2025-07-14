import gradio as gr
import requests
import json
import torch
import numpy as np
from PIL import Image
import base64
import io

# Model server configuration
MODEL_SERVER_URL = "http://localhost:8000/v2/models/hidream_i1/infer"

# Model configurations (for parameter setting)
MODEL_CONFIGS = {
    "dev": {
        "model_type": "dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0
    },
    "full": {
        "model_type": "full", 
        "guidance_scale": 5.0,
        "num_inference_steps": 28,
        "shift": 3.0
    },
    "fast": {
        "model_type": "fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

def make_inference_request(prompt, parameters):
    """Make HTTP request to triton model server"""
    payload = {
        "inputs": [
            {
                "name": "prompt",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt]
            },
            {
                "name": "parameters",
                "shape": [1], 
                "datatype": "BYTES",
                "data": [json.dumps(parameters)]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(MODEL_SERVER_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")

def decode_image_from_response(response):
    """Decode image from triton server response"""
    try:
        # Extract image data from response
        outputs = response.get("outputs", [])
        image_output = None
        
        for output in outputs:
            if output.get("name") == "images":
                image_output = output
                break
        
        if not image_output:
            raise Exception("No image output found in response")
        
        # Get image data and shape
        image_data = image_output.get("data", [])
        shape = image_output.get("shape", [])
        
        if not image_data or not shape:
            raise Exception("Invalid image data or shape in response")
        
        # Convert to numpy array and reshape
        # Assuming the data is in UINT8 format as mentioned in README
        image_array = np.array(image_data, dtype=np.uint8)
        
        # Reshape to [height, width, channels] - remove batch dimension
        if len(shape) == 4:  # [batch, height, width, channels]
            image_array = image_array.reshape(shape[1], shape[2], shape[3])
        else:  # [height, width, channels]
            image_array = image_array.reshape(shape[0], shape[1], shape[2])
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        return image
        
    except Exception as e:
        raise Exception(f"Failed to decode image: {str(e)}")

# Generate image function
def generate_image(prompt, resolution, seed, model_type="dev"):
    # Get configuration for selected model
    config = MODEL_CONFIGS[model_type]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    # Prepare parameters
    parameters = {
        "model_type": config["model_type"],
        "height": height,
        "width": width,
        "seed": int(seed),
        "num_images_per_prompt": 1
    }
    
    try:
        # Make request to model server
        response = make_inference_request(prompt, parameters)
        
        # Decode image from response
        image = decode_image_from_response(response)
        
        return image, seed
        
    except Exception as e:
        # Return error message as text overlay on a blank image
        error_img = Image.new('RGB', (512, 512), color='red')
        print(f"Error generating image: {str(e)}")
        return error_img, seed

# Create Gradio interface
with gr.Blocks(title="HiDream Image Generator (HTTP Client)") as demo:
    gr.Markdown("# HiDream Image Generator (HTTP Client)")
    gr.Markdown("*Using HTTP requests to triton model server at localhost:8000*")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt", 
                value="A sports team logo with dynamic mascot and team colors in a modern, sleek design with name 'BeaverDreams' and slogan 'Dream Big, Beaver!'",
                lines=3
            )
            
            model_type = gr.Radio(
                choices=list(MODEL_CONFIGS.keys()),
                value="dev",
                label="Model Type",
                info="Select model configuration"
            )
            
            resolution = gr.Radio(
                choices=RESOLUTION_OPTIONS,
                value=RESOLUTION_OPTIONS[0],
                label="Resolution",
                info="Select image resolution"
            )
            
            seed = gr.Number(
                label="Seed (use -1 for random)", 
                value=-1, 
                precision=0
            )
            
            generate_btn = gr.Button("Generate Image")
            seed_used = gr.Number(label="Seed Used", interactive=False)
            
        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, resolution, seed, model_type],
        outputs=[output_image, seed_used]
    )

# Launch app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)