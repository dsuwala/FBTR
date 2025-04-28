from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_name = "FartLabs/FART_Augmented"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer.save_pretrained("/app/model")
    model.save_pretrained("/app/model")


if __name__ == "__main__":
    main()

