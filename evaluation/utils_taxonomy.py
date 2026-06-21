import os
from google import genai
from google.genai import types


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class TaxonomyBench_utils:
    def __init__(self):
        self.client = client
        return
    def llm_judge(self, answer, predict):
        print(answer, predict)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents = [
                'You are an evaluator who rates model answers by comparing them to the reference answer.'
                'The reference answer is: ' + answer,
                'The model answer is: ' + predict,
                'Please determine if the model answer is correct or incorrect. Return "True" if correct, "False" if incorrect.'
            ]
        )  
        return response.text
    def is_correct(self, answer, predict):
        return self.llm_judge(answer, predict)
        # predict = str(predict)
        # answer = str(answer)
        # if predict == answer:
        #     return True
        # return False

       

