import unittest
import http.client
import json, time
from threading import Thread


class CaseTest(unittest.TestCase):
    def test_translate_sdk_task(self):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
        tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang="en", tgt_lang="fr")
        src_text = "Life is like a box of chocolates."
        tgt_lang = "人生はチョコレートの箱のようなものだ。"
        tokenizer.src_lang = "en"
        model_inputs = tokenizer(src_text, return_tensors="pt")
        generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("ja"))
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        assert tgt_lang == result[0]

    def test_translator_api_tasks(self):
        for i in range(10):
            print('task start: {}'.format(i))
            task()
            time.sleep(1)

        assert True



def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper


@async_call
def task():
    conn = http.client.HTTPConnection("127.0.0.1:9527")

    data = {"payload": {
        "fromLang": "en",
        "records": [
            {
                "id": "123",
                "text": "Life is like a box of chocolates."
            },
            {
                "id": "789",
                "text": "Life is like a box of chocolates."
            }
        ],
        "toLang": "ja"
    }
    }
    payload = json.dumps(data)
    text_records = list(map(lambda x: x['text'], data['payload']['records']))
    ids = list(map(lambda x: x['id'], data['payload']['records']))
    result = []
    for id, text in zip(ids, text_records):
        result.append({"id": id, "text": text})

    conn.request("POST", "/translation", payload)
    res = conn.getresponse()
    data = res.read()
    response = data.decode("utf-8")
    response_dict = json.loads(response)
    print(response_dict)
