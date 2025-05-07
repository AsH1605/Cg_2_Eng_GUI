import modal
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from safetensors.torch import load_file  # Safetensors loading

# Use App instead of Stub (deprecation fix)
stub = modal.App(name="cg-eng-translation")

# Define model directory for cg_eng_small
MODEL_DIR = "/models/cg_eng_small"
volume = modal.Volume.from_name("AST_MODAL_MONO_JAVA", create_if_missing=True)

# Define container/image with the required dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "sentencepiece", "safetensors"
)

@stub.function(
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=300
)
def translate_cg_to_eng():
    # Load tokenizer from the mounted volume
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Load model weights from safetensors
    model_path = Path(MODEL_DIR) / "model.safetensors"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # Hardcoded input Chhattisgarhi text (example text)
    cg_text = "तें का करत हस?"  # Replace with actual input text for translation

    # Tokenize input text and generate translation
    inputs = tokenizer(cg_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    eng_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Output the translated text
    print("Chhattisgarhi:", cg_text)
    print("English Translation:", eng_text)

# Run locally without CLI input
if __name__ == "__main__":
    with stub.run():
        translate_cg_to_eng.remote()
