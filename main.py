from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List
import ollama
from dotenv import load_dotenv
import os
import json
import re
from fastapi.concurrency import run_in_threadpool

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
    messages: List[MessageItem] = Field(..., alias="Messages")

    class Config:
        validate_by_name = True
        allow_population_by_alias = True

app = FastAPI()

@app.post("/generate")
async def generate(
    req: TranslationRequest,
    x_api_key: str = Depends(verify_api_key)
):
    input_json = req.model_dump_json(by_alias=True, indent=2)
    num_expected = len(req.messages)

    example_list = json.dumps(
        ["<traducción del primer mensaje>", "<traducción del segundo mensaje>"],
        ensure_ascii=False,
        indent=2
    )


    base_prompt = f"""
You are a translation engine.
You are a translator with deep expertise in industrial manufacturing and plant‐floor audit terminology.
All input texts come from SmartLPA, the leading web‐based software platform used by global manufacturing companies to run high‐frequency plant‐floor audits. SmartLPA supports Layered Process Audits (LPAs), 5S audits, safety audits, environmental and COVID-19 audits, mistake-proofing, and any other checklist‐based inspections. These messages will appear directly in SmartLPA’s interface, user guides, and audit reports—so consistency, precision, and correct context are essential.

Input JSON:
{input_json}

Task:
- Translate each message from **{req.source_language}** into **{req.language}**.
- Respond **only** with a JSON array of strings.
- Do NOT wrap the array in code fences or add any commentary.
You have to:
- Translate just **{req.language}** and be precise with the grammatical and industrial context.
Tone, Grammar & Style:
- Use a formal-technical register appropriate for industrial manuals or software UIs.
- Write short, clear sentences without ambiguity.
- Preserve original structure: do not split, merge, or reorder sentences.

Example of correct output:
{example_list}
"""

    async def call_and_parse(prompt_text: str) -> List[str]:
        response = await run_in_threadpool(
            lambda: ollama.chat(
                model="gemma3",
                messages=[{"role": "assistant", "content": prompt_text}]
            )
        )
        raw = response["message"]["content"]

        
        content = raw.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

       
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"(\[.*?\])", content, re.S)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
            
                    raise HTTPException(
                        500,
                        detail=f"JSON válido no encontrado en: {match.group(1)!r}"
                    )
            else:
        
                raise HTTPException(
                    500,
                    detail=f"Could not parse JSON array from model output: {raw!r}"
                )

        # Validar que sea lista de strings
        if not isinstance(parsed, list):
            raise HTTPException(
                500,
                detail="The model did not return a JSON array."
            )
        for idx, item in enumerate(parsed):
            if not isinstance(item, str):
                raise HTTPException(
                    400,
                    detail=f"Element at position {idx} is not a string: {item!r}"
                )
        return parsed
    prompt_text = base_prompt
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        translations = await call_and_parse(prompt_text)

        if len(translations) == num_expected:
            return translations
        else:
            mismatched = len(translations)
            error_feedback = (
                f"The previous output had {mismatched} items, but {num_expected} were expected.\n"
                f"Previous output:\n{json.dumps(translations, ensure_ascii=False, indent=2)}\n\n"
            )
            correction_prompt = f"""
{error_feedback}
Please re-translate ensuring you return exactly {num_expected} strings.
Follow all previous instructions strictly (only a JSON array of strings, no code fences, no commentary).

Input JSON:
{input_json}
"""
            prompt_text = correction_prompt
    raise HTTPException(
        500,
        detail=(
            f"Failed to produce exactly {num_expected} items after {max_attempts} attempts. "
            f"Last output had {len(translations)} items."
        )
    )
