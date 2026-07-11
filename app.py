import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================
# Configuration
# ============================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "tstuti7/nepali-qlora-model"

# ============================================
# Load Tokenizer
# ============================================
print("=" * 50)
print("Loading tokenizer...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
print("✅ Tokenizer loaded.")

# ============================================
# Load Base Model
# ============================================
print("=" * 50)
print("Loading base model...")
print("=" * 50)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if torch.cuda.is_available():
    print("Using GPU")
    base_model = base_model.to(dtype=torch.float16, device="cuda")
else:
    print("Using CPU")
    base_model = base_model.to(dtype=torch.float32)

print("✅ Base model loaded.")

# ============================================
# Load LoRA Adapter
# ============================================
print("=" * 50)
print("Loading LoRA adapter...")
print("=" * 50)

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)
model.eval()
print("✅ LoRA adapter loaded.")
print("✅ Model Loaded Successfully!")

# ============================================
# Summarization Function
# ============================================
def summarize(text):
    if not text.strip():
        return "कृपया नेपाली समाचार लेख्नुहोस्।"

    prompt = f"""### Instruction:
निम्न समाचारको छोटो सारांश लेख्नुहोस्।

### Input:
{text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
       outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.4,
    top_p=0.85,
    do_sample=True,
    repetition_penalty=1.15,
    pad_token_id=tokenizer.eos_token_id
)
        

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in full_output:
        summary = full_output.split("### Response:")[-1].strip()
    else:
        summary = full_output.strip()

    return summary

# ============================================
# Example article for the demo
# ============================================
EXAMPLE_ARTICLE = """काठमाडौं । सरकारले आगामी आर्थिक वर्षका लागि सडक पूर्वाधार विकासमा विशेष जोड दिने भएको छ। अर्थ मन्त्रालयका अनुसार, ग्रामीण क्षेत्रका सडकहरू स्तरोन्नति गर्न र नयाँ राजमार्गहरू निर्माण गर्न ठूलो बजेट विनियोजन गरिने छ। यसका साथै, सरकारले विद्युतीय सवारी साधनको प्रयोगलाई प्रोत्साहन गर्न कर छुटको योजना पनि ल्याउने जनाएको छ।"""

# ============================================
# Gradio Interface (polished frontend)
# ============================================
theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

custom_css = """
#title { text-align: center; margin-bottom: 0.2em; }
#subtitle { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 1.5em; }
.gradio-container { max-width: 900px !important; margin: auto; }
footer { visibility: hidden; }
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("# 🇳🇵 Nepali News Summarizer", elem_id="title")
    gr.Markdown(
        "Fine-tuned **Qwen2.5-1.5B-Instruct** with **QLoRA** to summarize Nepali news articles.",
        elem_id="subtitle"
    )

    with gr.Row():
        with gr.Column():
            article_input = gr.Textbox(
                lines=14,
                label="नेपाली समाचार (Nepali News Article)",
                placeholder="यहाँ नेपाली समाचार पेस्ट गर्नुहोस्..."
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Summarize", variant="primary")

        with gr.Column():
            summary_output = gr.Textbox(
                lines=14,
                label="सारांश (Generated Summary)",
                interactive=False
            )

    gr.Examples(
        examples=[[EXAMPLE_ARTICLE]],
        inputs=article_input,
        label="Try an example article"
    )

    gr.Markdown(
        "⚠️ Running on free CPU hardware — generation may take 30–90 seconds.  \n"
        "Base model: Qwen2.5-1.5B-Instruct · Fine-tuning: QLoRA · Adapter hosted on Hugging Face"
    )

    submit_btn.click(fn=summarize, inputs=article_input, outputs=summary_output)
    clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[article_input, summary_output])

# ============================================
# Launch App
# ============================================
if __name__ == "__main__":
    print("Launching Gradio...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )