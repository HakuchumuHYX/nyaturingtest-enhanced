from openai import AsyncOpenAI

class LLMClient:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate_response(self, prompt: str, model: str, temperature: float = 0.7) -> str | None:
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,  # [修改] 使用传入的参数
            timeout=60,
        )
        return response.choices[0].message.content
