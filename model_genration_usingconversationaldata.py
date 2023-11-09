from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"D:\NOSQLBLOG\pythonProject\regularchatbot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path)
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Tokenize user input
    user_input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response
    response_ids = model.generate(user_input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode and print the response
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    print(f"Bot: {bot_response}")
