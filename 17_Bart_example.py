# Bart example

from transformers import BartForConditionalGeneration, BartTokenizer

# load model & tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", 
                                                     forced_bos_token_id=0)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# prepare input
example_sentence = "UN Chief Says There Is No <mask> in Syria"
inputs = tokenizer(example_sentence, return_tensors="pt")
print(inputs)

# generate output
generated = model.generate(inputs["input_ids"],max_new_tokens=50)
print(generated)
outputs = tokenizer.decode(generated[0], skip_special_tokens=True)
print(outputs)
