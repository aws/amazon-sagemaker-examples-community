from transformers import AutoTokenizer, AutoModel
import torch
import json


def predict_fn(input_data, model_and_tokenizer):
    print("Inside predict_fn")
    print("input_data: ", input_data)
    print("model_and_tokenizer: ", model_and_tokenizer)

    model, tokenizer = model_and_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    inputs = tokenizer(input_data['inputs'], return_tensors="pt", padding=True).to(device)

    print("inputs: ", inputs)
    outputs = model(**inputs)
    print("Outputs: ", outputs)
    return outputs.last_hidden_state, inputs['attention_mask']


def output_fn(prediction, content_type):
    print("Inside output_fn")
    print("prediction: ", prediction)
    print("content_type: ", content_type)

    hidden_states, attention_mask = prediction
    print("hidden_states: ", hidden_states)
    print("attention_mask: ", attention_mask)

    # Apply mean pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    print("input_mask_expanded: ", input_mask_expanded)
    pooled_output = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    print("pooled_output: ", pooled_output)

    # Return the pooled output in the desired format
    return json.dumps({"embeddings": pooled_output.tolist()})


def model_fn(model_dir,context=None):
    # Load model from HuggingFace Hub

    print("Inside model_fn")
    print("model_dir: ", model_dir)
    print("context: ", context)

    # Ensure Model is using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    print("Before AutoModel: ")
    model = AutoModel.from_pretrained(model_dir).to(device)
    print("After Automodel: ", model)

    print("before AutoTokenizer: ")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("After Autotokenizer: ", tokenizer)
    return model, tokenizer