import modal
import click

app = modal.App("cg-eng-translator")

# Load volume with your model
volume = modal.Volume.from_name("cg-eng-model-vol", create_if_missing=False)

# Create base image
image = modal.Image.debian_slim().pip_install("transformers", "torch", "click")

@app.function(
    image=image,
    volumes={"/model": volume},
)
@modal.concurrent(max_inputs=5)
def translate_chhattisgarhi_to_english(text: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_path = "/model/AST_MODAL_MONO_JAVA/cg_eng_small"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.local_entrypoint()
@click.option('--text', type=str, help="Text to be translated")
def main(text: str):
    result = translate_chhattisgarhi_to_english.remote(text)
    print("Translation:", result)
