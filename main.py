from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List
import ollama
from dotenv import load_dotenv
import os
import json

load_dotenv()
API_KEYS = {os.getenv("API_KEY"): 5}


def verify_api_key(x_api_key: str = Header(None)):
    credits = API_KEYS.get(x_api_key, 5)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key or no credits")
    return x_api_key


class MessageItem(BaseModel):
    message: str


class TranslationRequest(BaseModel):
    source_language: str = Field(..., alias="SourceLanguage")
    language: str = Field(..., alias="language")
    messages: List[MessageItem]   = Field(..., alias="Messages")

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias      = True
        schema_extra = {
            "example": {
                "SourceLanguage": "espanish",
                "language": "english",
                "Messages": [
                    {"message": "Hola, ¿cómo estás?"},
                    {"message": "Planta"}
                ]
            }
        }


app = FastAPI()


@app.post("/generate")
def generate(
    req: TranslationRequest,
    x_api_key: str = Depends(verify_api_key)
):

    input_json = req.model_dump_json(by_alias=True, indent=2)


    example_response = {
        "Language": req.language,
        "Messages": [
            {"message": "<traducción del primer mensaje>"},
            {"message": "<traducción del segundo mensaje>"}
        ]
    }
    example_json = json.dumps(example_response, ensure_ascii=False, indent=2)

    prompt = f"""
You are a translation engine.
Context: The texts come from SmartLPA, a site for plant floor audits.

Input JSON:
{input_json}

Task:
- Translate each message into **exactly** the  language: **{req.language}**.
- **Respond ONLY** with a JSON object in this exact format (no extra fields, no explanations):
- You have to translate the messages from the source language **{req.source_language}**  target language **{req.language}**.

{example_json}
"""


    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "assistant", "content": prompt}]
    )

 
    return {"response": response["message"]["content"]}

