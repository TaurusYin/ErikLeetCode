# File name: translate_service.py
# from accelerate import Accelerator
import json
from starlette.requests import Request
from ray import serve
from app.utils.base_decorator import CommonLogger


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0}, route_prefix='/translation')
class Translator:
    """
    num_replicas determines how many copies of our deployment process run in Ray.
    Requests are load balanced across these replicas, allowing you to scale your deployments horizontally.
    """

    @CommonLogger()
    def __init__(self):
        """
        initialize the task of loading models which takes a bit long time.
        LightSeq can be considered to optimize performance
        """
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M', cache_dir='/tmp')
        self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang="en", tgt_lang="fr",
                                                         cache_dir='/tmp')
        # self.accelerator = Accelerator(split_batches=True, dispatch_batches=False)

    @CommonLogger()
    def translate(self, text_records: list, src_lang="en", dst_lang="ja") -> list:
        """
        translate text to target language
        ['Life is like a box of chocolates.'] -> ['人生はチョコレートの箱のようなものだ']
        :param text_records: translation text which is required to translate
        :param src_lang: resource language
        :param dst_lang: target language
        :return: the list of the text records with target language
        """
        self.tokenizer.src_lang = src_lang
        model_inputs = self.tokenizer(text_records, return_tensors="pt")
        generated_tokens = self.model.generate(**model_inputs, forced_bos_token_id=self.tokenizer.get_lang_id(dst_lang))
        # generated_tokens = self.accelerator.unwrap_model(self.model).generate(**model_inputs,forced_bos_token_id=self.tokenizer.get_lang_id(dst_lang))
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return result

    @CommonLogger()
    def inference_tasks(self, records: list, src_lang="en", dst_lang="ja") -> str:
        """
        :param text_records: translation text which is required to translate
        :param src_lang: resource language
        :param dst_lang: target language
        :return: the json string output: {'result': [{'id': '123', 'text': '人生はチョコレートの箱のようなものだ。'}]}
        """
        text_records = list(map(lambda x: x['text'], records))
        translated_text_records = self.translate(text_records, src_lang, dst_lang)
        ids = list(map(lambda x: x['id'], records))
        result = []
        for idx, text in zip(ids, translated_text_records):
            result.append({"id": idx, "text": text})
        output_dict = {"result": result}
        print(result)
        output_json_str = json.dumps(output_dict, ensure_ascii=False)
        return output_json_str

    async def __call__(self, http_request: Request) -> str:
        try:
            req_dict = await http_request.json()
            pay_load = req_dict['payload']
            records, src_lang, dst_lang = pay_load['records'], pay_load['fromLang'], pay_load['toLang']
            return self.inference_tasks(records, src_lang, dst_lang)

        except Exception as e:
            print('please check the service problem: {}'.format(e))


translator = Translator.bind()
