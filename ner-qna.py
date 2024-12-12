import spacy
import warnings
from huggingface_hub import login
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig

# Suppress specific UserWarnings related to spaCy model versions
warnings.filterwarnings("ignore", category=UserWarning, module="spacy")
warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_name = "./QA/qna-model/llama-3B-adapter"

print("Load QnA model...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype="auto",
    device_map="auto"
)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)

# Load the PEFT adapter
model = PeftModel.from_pretrained(
    base_model, 
    adapter_name  # Path to the saved PEFT adapter
)

# Set the model to evaluation mode
model.eval()


def ner(user_input):
    # Load the custom SpaCy model
    nlp = spacy.load("./NER/output/model-best")

    # Process the user input through the model
    doc = nlp(user_input)
    
    res = {}
    # Extract recognized entities
    for ent in doc.ents:
        if ent.label_ == "FROM_CITY":
            res["from"] = ent.text
        elif ent.label_ == "TO_CITY":
            res["to"] = ent.text
    
    print("FROM: ", res["from"])
    print("TO: ", res["to"])
    
    # Validate that both 'from' and 'to' locations are identified
    if len(res) != 2:
        print("Error: Could not identify both origin and destination locations.")
        return None
    
    return res

def qna(ner_res):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    # Prepare input
    input_text = f"Question: How can I get to {ner_res['to']} from {ner_res['from']}\nAnswer: "
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate response
    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    
    
    # Decode and extract answer
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    answer = full_response.split("Answer: ")[-1].strip()
    return answer

def main():
    """
    Main CLI application loop for travel assistant.
    Continuously prompts user for input until they choose to exit.
    """
    print("Welcome to the Travel Assistant CLI!")
    print("Enter a travel route query (e.g., 'How to get from New York to Los Angeles')")
    print("Type 'exit' or press Ctrl+C to quit the application.")
    
    while True:
        try:
            # Prompt for user input
            user_input = input("\nEnter your travel query: ")
            
            # Check for exit condition
            if user_input.strip().lower() in ['exit', 'quit', 'q']:
                print("Thank you for using the Travel Assistant. Goodbye!")
                break
            
            # Validate input
            if not user_input:
                print("Please enter a valid query.")
                continue
            
            # Perform Named Entity Recognition
            print(user_input)
            ner_res = ner(user_input)
            
            # If NER fails, continue the loop
            if ner_res is None:
                continue
            
            # Generate and print travel route
            route_info = qna(ner_res)
            print("\nTravel Route Information:")
            print(route_info)
        
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nApplication terminated by user. Goodbye!")
            break
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()