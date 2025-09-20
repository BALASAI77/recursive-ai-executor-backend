from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import subprocess
import json
import re
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://recursive-ai-executor-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

class PromptRequest(BaseModel):
    prompt: str

def extract_code(response_text):
    """Extracts Python code from triple-backtick blocks."""
    pattern = r'```python\n([\s\S]*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return matches[0] if matches else response_text

def execute_code(code):
    """Executes code in sandbox with 5s timeout."""
    os.makedirs("sandbox", exist_ok=True)
    with open('sandbox/code.py', 'w') as f:
        f.write(code)
    try:
        result = subprocess.run(
            ['python', 'sandbox/code.py'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout, result.stderr, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", "Execution timed out", False

@app.post("/execute")
async def execute_prompt(request: PromptRequest):
    prompt = request.prompt
    log = {"prompt": prompt, "attempts": [], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}
    for attempt in range(5):
        try:
            response = model.generate_content(prompt)
            code = extract_code(response.text)
            stdout, stderr, success = execute_code(code)
            log["attempts"].append({
                "attempt": attempt + 1,
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
                "success": success
            })
            if success:
                break
            prompt = f"{prompt}\nPrevious code had error: {stderr}. Fix it."
        except Exception as e:
            log["attempts"].append({
                "attempt": attempt + 1,
                "code": "",
                "stdout": "",
                "stderr": str(e),
                "success": False
            })
    log_file = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    return {
        "prompt": prompt,
        "final_code": log["attempts"][-1]["code"],
        "output": log["attempts"][-1]["stdout"],
        "error": log["attempts"][-1]["stderr"],
        "success": log["attempts"][-1]["success"],
        "attempts_count": len(log["attempts"]),
        "log_file": log_file
    }

@app.get("/logs/{filename}")
async def get_log(filename: str):
    try:
        with open(f"logs/{filename}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Log file not found"}