import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    """A class for translating text between Vietnamese and English using Hugging Face models.

    Args:
        device (str): The device to run the models on, e.g., 'cuda' or 'cpu'.
    """

    def __init__(self, device):
        """Initialize the Translator class with Hugging Face models for translation.

        Args:
            device (str): The device to run the models on, e.g., 'cuda' or 'cpu'.
        """
        self.tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
        self.model_vi2en.to(device)

        self.tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
        self.model_en2vi.to(device)

    def translate_vi2en(self, vi_texts):
        """Translate Vietnamese text to English.

        Args:
            vi_texts (str or list): The input Vietnamese text or list of texts to translate.

        Returns:
            str or list: The translated English text or list of texts.
        """
        input_ids = self.tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to('cuda')
        output_ids = self.model_vi2en.generate(
            **input_ids,
            decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        en_texts = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
        return en_texts

    def translate_en2vi(self, en_texts):
        """Translate English text to Vietnamese.

        Args:
            en_texts (str or list): The input English text or list of texts to translate.

        Returns:
            str or list: The translated Vietnamese text or list of texts.
        """
        input_ids = self.tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to('cuda')
        output_ids = self.model_en2vi.generate(
            **input_ids,
            decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        vi_texts = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        return vi_texts

    def Viet2Eng(self, Viet_list):
        """Translate a list of Vietnamese text to English.

        Args:
            Viet_list (list): A list of Vietnamese texts to translate.

        Returns:
            list: A list of translated English texts.
        """
        step = 6
        start = 0
        end = 0
        Eng_list = []
        while True:
            output = self.translate_vi2en(Viet_list[start+end:end+step])
            Eng_list.extend(output)

            end = end+step

            if end==len(Viet_list):
                return Eng_list

    def Eng2Viet(self, Eng_list):
        """Translate a list of English text to Vietnamese.

        Args:
            Eng_list (list): A list of English texts to translate.

        Returns:
            list: A list of translated Vietnamese texts.
        """
        step = 6
        start = 0
        end = 0
        Viet_list = []
        while True:
            output = self.translate_en2vi(Eng_list[start+end:end+step])
            Viet_list.extend(output)

            end = end+step

            if end==len(Eng_list):
                return Viet_list
            
    def __call__(self, ori_topic, ori_context):
        """Translate topics and contexts between Vietnamese and English.

        Args:
            ori_topic (list): The list of original topics in Vietnamese.
            ori_context (list): The list of original contexts in Vietnamese.

        Returns:
            tuple: A tuple containing the final translated topics and contexts.
        """
        final_Topic = []
        final_Context = []

        # Topic translate
        eng_Topic = self.Viet2Eng(ori_topic)
        viet_Topic = self.Eng2Viet(eng_Topic)

        final_Topic.extend(ori_topic)
        final_Context.extend(ori_context)
        final_Topic.extend(viet_Topic)
        final_Context.extend(ori_context)

        return final_Topic, final_Context


if __name__ == '__main__':
    dataset_path = ''  # Fill in the path to your dataset CSV file
    dataset = pd.read_csv(dataset_path)

    ori_topic = list(dataset['Topic'])
    ori_context = list(dataset['Context'])

    translator = Translator()
    final_Topic, final_Context = translator(ori_topic, ori_context)

    new_df = pd.DataFrame({'Topic': final_Topic, 'Context': final_Context})

    new_df.to_csv('')
