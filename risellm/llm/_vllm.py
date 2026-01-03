import os
import openai
from openai import OpenAI
from transformers import AutoTokenizer

class VLLM:
    def __init__(self, 
                 model_name="llama3.1-8b-instruct", 
                 deployed_model_name="",
                 base_url="localhost",
                 port="8000",
                 api_key=None):
        self.model_name = model_name
        self.deployed_model_name = deployed_model_name
        self.port = port
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = f"http://{base_url}:{port}/v1"
        openai.api_base = self.openai_base_url
        openai.api_key = self.api_key
        
        if "llama" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self,
                messages,
                chat=True,
                do_sample=True,
                temperature=0.0,
                max_new_tokens=1024,
                top_p=1.0,):
        
        client = OpenAI(
            base_url=self.openai_base_url,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # if chat:
        #     prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        #     print(prompt)
        
        if chat:
        
            completion = client.chat.completions.create(
                model=self.deployed_model_name,
                messages=messages,
                temperature=temperature if do_sample else 0.0,
                max_tokens=max_new_tokens,
                top_p=top_p if do_sample else 1.0
            )
            return completion.choices[0].message.content
   
        
if __name__ == "__main__":
	llm = VLLM(
		model_name="meta-llama/Llama-3.1-8B-Instruct",
        deployed_model_name="llama3.1-8b-instruct",
		base_url="localhost",
		port="8000",
	)

	print(llm.generate(
        chat=True,
        messages=[
            {"role": "system", "content": "Answer to the question in Korean"},
            {"role": "user", "content": "Explain about PPO algorithm"},
        ]
     ))