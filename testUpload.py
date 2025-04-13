from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from pydantic import BaseModel
import os
import asyncio

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель для входных данных
class GherkinRequest(BaseModel):
    base64_image: str
    description: str
    image_type: str = "image/jpeg"

@app.get("/test")
async def test():
    return {"message": "Server is working"}

@app.post("/generate-gherkin")
async def generate_gherkin(request: GherkinRequest):
    """
    Генерирует Gherkin-тесткейс на основе изображения
    """
    api_key = os.getenv("MISTRAL_AI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generate a Gherkin test case in English for: {request.description}. Only provide the Gherkin syntax, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{request.image_type};base64,{request.base64_image}"
                    }
                ]
            }
        ]

        chat_response = await client.chat.complete_async(
            model="pixtral-12b-2409",
            messages=messages
        )

        response_text = chat_response.choices[0].message.content.strip()
        return JSONResponse(content={"gherkin": response_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)