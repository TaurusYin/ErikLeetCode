from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import sentencepiece

model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang="en", tgt_lang="fr")

if __name__ == "__main__":
    src_text = "Life is like a box of chocolates."
    tgt_lang = "人生はチョコレートの箱のようなものだ。"
    src_text_records = [src_text, src_text]

    tokenizer.src_lang = "en"
    model_inputs = tokenizer(src_text_records, return_tensors="pt")
    generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("ja"))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print(result[0])
