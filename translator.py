from transformers import MarianMTModel,MarianTokenizer
import torch
class MarianMTModelTranslator():
    def __init__(self,type,origin_code,target_code,device_name,batch_size) -> None:
        """翻译工具，使用MarianMTModel翻译，支持的模型可以从https://huggingface.co/Helsinki-NLP查询。
        https://huggingface.co/docs/transformers/model_doc/marian
        中文digits code为“zh”,英文为“en”，法文为“fr”

        Args:
            type (_type_): "translate" or "back_translate"
            origin_code (_type_): 原文本的digits code
            target_code (_type_): 目标文本的digits code
            device_name (str): 'cuda:0' or "cpu" etc.
            batch_size (int): batch size
        """               
        self.type=type 
        self.batch_size=batch_size
        self.device=torch.device(device_name)
        if type == 'translate':
            #翻译任务
            first_model_name=f'Helsinki-NLP/opus-mt-{origin_code}-{target_code}'
            self.first_model_tkn=MarianTokenizer.from_pretrained(first_model_name)
            self.first_model=MarianMTModel.from_pretrained(first_model_name).to(self.device)

        elif type == 'back_translate':
            #回译任务
            first_model_name=f'Helsinki-NLP/opus-mt-{origin_code}-{target_code}'
            self.first_model_tkn=MarianTokenizer.from_pretrained(first_model_name)
            self.first_model=MarianMTModel.from_pretrained(first_model_name).to(self.device)
            
            second_model_name=f'Helsinki-NLP/opus-mt-{target_code}-{origin_code}'
            self.second_model_tkn=MarianTokenizer.from_pretrained(second_model_name)
            self.second_model=MarianMTModel.from_pretrained(second_model_name).to(self.device)


    def translate(self,origin_texts):
        """翻译，输入为 a list of origin string

        Args:
            origin_texts (_type_): 字符串的列表

        Returns:
            list: 目标语言列表
        """        
        assert self.type=='translate'
        results=[]

        for left in range((len(origin_texts)-1)//self.batch_size + 1):

            data=origin_texts[left*self.batch_size:(left+1)*self.batch_size]

            inputs=self.first_model_tkn(data,return_tensors='pt',padding=True)
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            outputs=self.first_model.generate(**inputs)
            output=[self.first_model_tkn.decode(t,skip_special_tokens=True) for t in outputs]
            results.extend(output)
        return results

    def back_translate(self,origin_texts):
        """将原文进行回译，输入是原文的字符串列表。

        Args:
            origin_texts (_type_): 输入是原文的字符串列表。

        Returns:
            list: 新的原文字符串列表
        """        
        assert self.type == 'back_translate'
        results=[]

        for left in range((len(origin_texts)-1)//self.batch_size + 1):

            data=origin_texts[left*self.batch_size:(left+1)*self.batch_size]

            inputs=self.first_model_tkn(data,return_tensors='pt',padding=True)
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            outputs=self.first_model.generate(**inputs)
            output_texts=[self.first_model_tkn.decode(t,skip_special_tokens=True) for t in outputs]

            inputs=self.second_model_tkn(output_texts,return_tensors='pt',padding=True)
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            outputs=self.second_model.generate(**inputs)
            output=[self.second_model_tkn.decode(t,skip_special_tokens=True) for t in outputs]
            results.extend(output)
        return results


if __name__ =='__main__':
    #回译任务
    translator=MarianMTModelTranslator(type='back_translate',origin_code='zh',target_code='en',device_name='cuda:1',batch_size=20)
    output_texts=translator.back_translate(['今天中午吃什么','你明天要去哪里玩','苦海无涯回头是岸'])
    print(output_texts)

    #翻译任务
    translator=MarianMTModelTranslator(type='translate',origin_code='zh',target_code='en',device_name='cuda:1',batch_size=20)
    output_texts=translator.translate(['今天中午吃什么','你明天要去哪里玩','苦海无涯回头是岸'])
    print(output_texts)


