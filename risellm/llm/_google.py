import os
from google import genai
from google.genai import types


class GoogleLLM:
    def __init__(self, model_name, api_key=os.getenv("GOOGLE_API_KEY")):
        self.model = model_name
        self.client = genai.Client(api_key=api_key)

    def generate(self,
                 user_prompt="",
                 system_prompt="",
                 thinking_budget=128,
                 max_tokens=1024, 
                 temperature=1.0, 
                 top_p=1):
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=thinking_budget
                ), 
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )	
        )
        return response.text
    
if __name__ == "__main__":
    llm = GoogleLLM(model_name="gemini-2.5-flash")
    print(llm.generate(user_prompt="where is the moon?", system_prompt="Answer in Korean"))