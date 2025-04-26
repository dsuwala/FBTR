from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "FartLabs/FART_Augmented"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained("./model")
model.save_pretrained("./model")

