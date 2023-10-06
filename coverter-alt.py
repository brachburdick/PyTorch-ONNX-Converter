import transformers
import transformers.convert_graph_to_onnx as onnx_convert
from transformers import AutoTokenizer, AutoModel,DistilBertForSequenceClassification

from pathlib import Path
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

pipeline = transformers.pipeline("text-classifcation", model = model, tokenizer = tokenizer)
onnx_convert.convert_pytorch(pipeline, opset =11, output = Path('alt.onnx'), use_external_format = False)