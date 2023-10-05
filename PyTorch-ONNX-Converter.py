import torch
from transformers import AutoTokenizer, AutoModel,DistilBertForSequenceClassification
import onnxruntime
import torch.onnx
import numpy as np
import torch.nn.functional as F
import onnxruntime.quantization as quantization
providers = ['CPUExecutionProvider']


# Choose desired PyTorch Model & Specify desired output directory
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
output_directory = "distilbert-emotion.onnx"

# Choose desired tokenizer and model from tranformers library
# View other options here: https://huggingface.co/docs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Specify the type of data you'll be feeding this model
input_text = "This is my example input text"

# Tokenize the input text and get the input_ids as a PyTorch tensor
tokens = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
input_ids = tokens["input_ids"]

# Export to ONNX
torch.onnx.export(model, input_ids, output_directory, export_params=True, verbose=True)


#The snippet below can be uncommented to test your model conversion.
#____________________________________________________________________________
# ort_session = onnxruntime.InferenceSession(output_directory, providers=providers)
# input_name = ort_session.get_inputs()[0].name

# # Convert PyTorch tensor to numpy array before feeding it to ONNX Runtime
# input_dict = {input_name: input_ids.detach().numpy()}
# results = ort_session.run(None, input_dict)

# # Format the output
# logits = results[0]
# probs = F.softmax(torch.tensor(logits), dim=-1)
# label_map = model.config.id2label
# probs_np = probs.detach().numpy()

# output = [
#     {'label': label_map[i], 'score': prob} 
#     for i, prob in enumerate(probs_np[0])
# ]

# print("Output:", [output])
