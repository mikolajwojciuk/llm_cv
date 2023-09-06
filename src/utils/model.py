from constants import LLM
import transformers


def get_tokenizer_and_model(model_id: str = LLM):
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", offload_folder="."
    ).eval()

    return tokenizer, model
