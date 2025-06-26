from notebook_utils import collect_telemetry
from llm_config import convert_and_compress_model
from llm_config import compare_model_size

class GetModelDir:
    def get_model_dir(self):
        
        collect_telemetry("deepseek-r1.ipynb")
        
        model_id = "DeepSeek-R1-Distill-Qwen-1.5B"
        model_configuration = {'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'genai_chat_template': "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}", 'system_prompt': "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.", 'stop_strings': ['<｜end▁of▁sentence｜>', '<｜User｜>', '</User|>', '<|User|>', '<|end_of_sentence|>', '</｜']}
        compression_variant = "INT4"
        use_preconverted=True

        model_dir = convert_and_compress_model(model_id, model_configuration, compression_variant, use_preconverted=use_preconverted)

        # compare_model_size(model_dir)
        return model_dir
