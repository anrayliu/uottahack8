import os
from abc import ABC, abstractmethod

from groq import Groq
from openai import OpenAI


class Llm(ABC):
    def __init__(self, instructions):
        self.client = None

        # initial context given to models
        self.instructions = instructions

    @abstractmethod
    def init(self):
        ...

    @abstractmethod
    def get_response(self, prompt):
        ...

class Gpt4(Llm):
    def __init__(self, instructions=""):
        super().__init__(instructions)

        self.init()

    def init(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def get_response(self, prompt):
        response = self.client.responses.create(
            model="gpt-4o",
            instructions=self.instructions,
            input=prompt
        )
        return response.output[0].content[0].text

class GroqModel(Llm, ABC):
    def __init__(self, model, instructions):
        super().__init__(instructions)

        self.groq_model = model

        self.init()

    def init(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def get_response(self, prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "system",
                    "content": self.instructions,
                },
            ],
            model=self.groq_model,
        )
        return chat_completion.choices[0].message.content

class KimiK2(GroqModel):
    def __init__(self, instructions=""):
        super().__init__("moonshotai/kimi-k2-instruct-0905", instructions)

        self.init()

class GptOss(GroqModel):
    def __init__(self, instructions=""):
        super().__init__("openai/gpt-oss-120b", instructions)

        self.init()

class Qwen3(GroqModel):
    def __init__(self, instructions=""):
        super().__init__("qwen/qwen3-32b", instructions)

        self.init()

class Llama33(GroqModel):
    def __init__(self, instructions=""):
        super().__init__("llama-3.3-70b-versatile", instructions)

        self.init()


# for testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = KimiK2()
    print(client.get_response("hello what model are you"))
