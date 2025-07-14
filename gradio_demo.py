import torch
import gradio as gr
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": "/home/ubuntu/.cache/huggingface/hub/models--azaneko--HiDream-I1-Dev-nf4/snapshots/c80f18e2b05a37aff40aa2c5842f025ce08bcb01",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": "azaneko/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 28,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
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

# Load models
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer
    
    return pipe, config

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

# Generate image function
def generate_image(prompt, resolution, seed):
    global pipe, current_model
    
    # Get configuration for current model
    config = MODEL_CONFIGS[current_model]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator
    ).images
    
    return images[0], seed

# Initialize with default model
print("Loading default model (dev)...")
current_model = "dev"
pipe, _ = load_models(current_model)
print("Model loaded successfully!")

# Create Gradio interface
with gr.Blocks(title="HiDream Image Generator (Dev Model 4bit Quantized)") as demo:
    gr.Markdown("# HiDream Image Generator")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt", 
                value="A sports team logo with dynamic mascot and team colors in a modern, sleek design with name 'BeaverDreams' and slogan 'Dream Big, Beaver!",
                lines=3
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
        inputs=[prompt, resolution, seed],
        outputs=[output_image, seed_used]
    )

# Launch app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)