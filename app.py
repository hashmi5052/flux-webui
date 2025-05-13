import gradio as gr
import numpy as np
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from optimum.quanto import QuantizedDiffusersModel
from PIL import Image
import os
import pandas as pd
import devicetorch

# ========= Prompt Enhancements =========
POSITIVE_PHRASES = "masterpiece, highly detailed, cinematic lighting, DSLR, bokeh, shallow depth of field, studio lighting, photo, realistic"
NEGATIVE_PROMPT = ""  # Add negative prompts here if needed

# ========= Utilities =========
def read_prompt_file(file_path):
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_csv(file_path, names=["prompt"])
        df["image number"] = range(1, len(df) + 1)
        return df
    else:
        raise ValueError("Unsupported file type.")

    df.columns = df.columns.str.strip().str.lower()
    required_columns = ["prompt", "image number"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}. Found columns: {df.columns.tolist()}")

    df = df.dropna(subset=["prompt"])
    df["prompt"] = df["prompt"].astype(str)
    df["image number"] = df["image number"].astype(int)
    return df

def get_starting_number(file):
    try:
        df = read_prompt_file(file.name)
        return int(df["image number"].iloc[0])
    except Exception as e:
        print(f"Error detecting starting number: {e}")
        return 0

# ========= Config and Globals =========
class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

dtype = torch.bfloat16
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
selected = None
run_index = 1
stop_flag = False
pipe = None

# ========= Load Pipeline =========
def load_pipeline(checkpoint):
    global pipe, selected

    if selected != checkpoint:
        if checkpoint == "sayakpaul/FLUX.1-merged":
            repo = "cocktailpeanut/xulf-d"
            model_id = "cocktailpeanut/flux1-merged-q8"
        else:
            repo = "cocktailpeanut/xulf-s"
            model_id = "cocktailpeanut/flux1-schnell-q8"

        print(f"Loading transformer: {model_id}")
        transformer = QuantizedFluxTransformer2DModel.from_pretrained(model_id, cache_dir="models")
        transformer.to(device=device, dtype=dtype)

        print("Loading pipeline...")
        pipe = FluxPipeline.from_pretrained(repo, transformer=None, torch_dtype=dtype, cache_dir="models")
        pipe.transformer = transformer
        pipe.to(device)

        # ========= CUDA Memory Optimization =========
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        selected = checkpoint
        print("Pipeline loaded!")

# ========= Generate from File =========
def generate_from_excel(file, checkpoint, width, height, num_images_per_prompt, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=False)):
    global run_index, stop_flag

    try:
        df = read_prompt_file(file.name)
    except Exception as e:
        yield None, f"Error reading file: {e}", "0% completed"
        return

    if df.empty:
        yield None, "No valid rows in file", "0% completed"
        return

    load_pipeline(checkpoint)

    run_folder = f"output/run-{run_index:02d}"
    os.makedirs(run_folder, exist_ok=True)

    for i, row in df.iterrows():
        if stop_flag:
            break

        image_number = int(row["image number"])
        prompt = f"{POSITIVE_PHRASES}, {row['prompt']}"
        image_path = f"{run_folder}/{image_number}.png"

        generator = torch.Generator().manual_seed(42)
        images = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            negative_prompt=NEGATIVE_PROMPT
        ).images

        images[0].save(image_path)
        torch.cuda.empty_cache()  # ✅ Clear CUDA memory

        percent = int(((i + 1) / len(df)) * 100)
        yield images[0], f"{image_number}", f"{percent}% completed"

    run_index += 1
    stop_flag = False

# ========= Stop Button Handler =========
def stop_generation():
    global stop_flag
    stop_flag = True
    return "Stopped generation."

# ========= Preview Image Number on File Upload =========
def preview_first_image_number(file):
    return str(get_starting_number(file))

# ========= UI =========
css = """
nav { text-align: center; }
#logo { width: 50px; display: inline; }
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("<nav><img id='logo' src='file/icon.png'/><h3>🖼️ Batch Image Generator</h3></nav>")

        with gr.Row():
            file = gr.File(label="Upload file (.xlsx, .csv, .txt)", file_types=[".xlsx", ".csv", ".txt"])
            checkpoint = gr.Dropdown(
                label="Model",
                value="black-forest-labs/FLUX.1-schnell",
                choices=["black-forest-labs/FLUX.1-schnell", "sayakpaul/FLUX.1-merged"]
            )

        with gr.Row():
            start_btn = gr.Button("🚀 Start Generation")
            stop_btn = gr.Button("🛑 Stop")

        with gr.Row():
            width = gr.Slider(label="Width", minimum=256, maximum=2048, step=32, value=1024)
            height = gr.Slider(label="Height", minimum=256, maximum=2048, step=32, value=576)

        with gr.Row():
            num_images_per_prompt = gr.Slider(label="Images per Prompt", minimum=1, maximum=10, step=1, value=1)
            num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=28)
            guidance_scale = gr.Number(label="Guidance Scale", minimum=0.0, maximum=50.0, value=3.5)

        with gr.Row():
            out_image = gr.Image(label="🖼️ Current Image", interactive=False)
            out_number = gr.Textbox(label="🆔 Image Number", interactive=False)
            out_progress = gr.Textbox(label="📊 Progress", interactive=False)

        start_btn.click(
            generate_from_excel,
            inputs=[file, checkpoint, width, height, num_images_per_prompt, num_inference_steps, guidance_scale],
            outputs=[out_image, out_number, out_progress]
        )

        stop_btn.click(stop_generation, outputs=[out_progress])
        file.change(fn=preview_first_image_number, inputs=file, outputs=out_number)

# ========= Launch =========
demo.launch(share=True)
