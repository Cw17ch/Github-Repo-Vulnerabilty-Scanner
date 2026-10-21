import os
import re
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================================
# 0. CONFIGURATION & ENVIRONMENT SETUP
# =====================================================================
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing in .env file.")

# Initialize Gemini with fast timeouts and low retries
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.2,
    timeout=30,
    max_retries=1
)

app = FastAPI(
    title="GitHub Repo Vulnerability Scanner",
    description="Fetch GitHub repo files, inspect code, find vulnerabilities inside a file and give solution."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported source file extensions
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb", ".html", ".css", ".json", ".yaml", ".yml",
    ".md", ".sql", ".sh",""
}

# Directories and vendor locks to ignore
IGNORED_PATHS = {
    "node_modules", "venv", ".git", "__pycache__", "dist", "build", ".next",
    "package-lock.json", "yarn.lock", "poetry.lock"
}

# =====================================================================
# 1. GITHUB REPOSITORY FETCHING LOGIC
# =====================================================================

def parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from a GitHub URL."""
    clean_url = url.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", clean_url)
    if not match:
        raise ValueError("Invalid GitHub URL format. Expected: https://github.com/owner/repo")
    return match.group(1), match.group(2)

def fetch_repo_files(owner: str, repo: str) -> List[Dict[str, str]]:
    """Fetch all source code files from GitHub REST API."""
    repo_api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    repo_res = requests.get(repo_api_url, headers=headers)
    if repo_res.status_code == 404:
        raise HTTPException(status_code=404, detail="Repository not found or is private.")
    elif repo_res.status_code != 200:
        raise HTTPException(status_code=repo_res.status_code, detail="Failed to fetch repository info from GitHub.")
    
    default_branch = repo_res.json().get("default_branch", "main")

    tree_api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    tree_res = requests.get(tree_api_url, headers=headers)
    
    if tree_res.status_code != 200:
        raise HTTPException(status_code=tree_res.status_code, detail="Failed to retrieve file tree from GitHub.")

    tree_data = tree_res.json().get("tree", [])
    extracted_files = []

    for item in tree_data:
        path = item.get("path", "")
        item_type = item.get("type", "")

        if item_type != "blob" or any(ignored in path.split("/") for ignored in IGNORED_PATHS):
            continue

        _, ext = os.path.splitext(path)
        if ext.lower() in ALLOWED_EXTENSIONS:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{path}"
            raw_res = requests.get(raw_url)
            
            content = raw_res.text if raw_res.status_code == 200 else f"Error reading content (HTTP {raw_res.status_code})"
            
            extracted_files.append({
                "path": path,
                "language": ext.lstrip("."),
                "content": content
            })

    return extracted_files

# =====================================================================
# 2. SCHEMAS & API ENDPOINTS
# =====================================================================

class RepoRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/psf/requests")

class RepoResponse(BaseModel):
    owner: str
    repo: str
    total_files: int
    files: List[Dict[str, str]]

class FilePromptRequest(BaseModel):
    file_path: str
    code_content: str
    custom_prompt: str

class FilePromptResponse(BaseModel):
    file_path: str
    ai_response: str

@app.post("/api/v1/inspect", response_model=RepoResponse)
async def inspect_repository(request: RepoRequest):
    try:
        owner, repo = parse_github_url(request.repo_url)
        files = fetch_repo_files(owner, repo)
        return RepoResponse(
            owner=owner,
            repo=repo,
            total_files=len(files),
            files=files
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agent-process", response_model=FilePromptResponse)
async def run_gemini_on_file(request: FilePromptRequest):
    """Executes a custom user prompt over the provided code file context."""
    try:
        path = request.file_path
        code = request.code_content
        user_instructions = request.custom_prompt.strip()

        if not user_instructions:
            user_instructions = "Show all security vulnerabilities (e.g., OWASP Top 10, SQL Injection, Command Injection, Insecure Deserialization, Hardcoded Secrets).Rewrite the code to completely fix and patch all identified security flaws while maintaining the original functionality."

        # Combine system framing, custom user prompt, and file code into one payload
        formatted_prompt = f"""
You are an expert Application Security Auditor (AppSec Engineer).
Analyze the following Python source code from file '{path}'.
    
Identify all security vulnerabilities (e.g., OWASP Top 10, SQL Injection, Command Injection, Insecure Deserialization, Hardcoded Secrets).
Provide your analysis in this format:
    1. VULNERABILITY NAME & SEVERITY (High/Medium/Low)
    2. AFFECTED LINE/FUNCTION
    3. DETAILED EXPLANATION OF RISK
Source Code:
FILE PATH: '{path}'

USER INSTRUCTION / PROMPT:
{user_instructions}
You are a Senior Security Software Engineer.
Given this vulnerability report
Rewrite the code to completely fix and patch all identified security flaws while maintaining the original functionality.
"""

        # Call Gemini LLM
        response = llm.invoke(formatted_prompt)
        
        # Format text output cleanly
        text_content = response.content
        if isinstance(text_content, list) and len(text_content) > 0:
            if isinstance(text_content[0], dict) and "text" in text_content[0]:
                text_content = text_content[0]["text"]

        return FilePromptResponse(
            file_path=path,
            ai_response=str(text_content)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 3. WEB DASHBOARD FRONTEND
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GitHub Repo Vulnerability Scanner</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                --bg-color: #0f172a;
                --card-bg: #1e293b;
                --text-color: #f8fafc;
                --accent-color: #38bdf8;
                --border-color: #334155;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                margin: 0;
                padding: 30px 20px;
                display: flex;
                justify-content: center;
            }
            .container {
                max-width: 1200px;
                width: 100%;
            }
            h1 { color: var(--accent-color); margin-bottom: 6px; }
            p.subtitle { color: #94a3b8; margin-top: 0; margin-bottom: 20px; }
            
            .input-group { display: flex; gap: 12px; margin-bottom: 20px; }
            input[type="text"], textarea {
                border-radius: 8px;
                border: 1px solid var(--border-color);
                background-color: var(--card-bg);
                color: var(--text-color);
                font-size: 15px;
                outline: none;
                padding: 12px 16px;
            }
            input[type="text"] { flex: 1; }
            input[type="text"]:focus, textarea:focus { border-color: var(--accent-color); }
            
            button {
                padding: 12px 24px; border-radius: 8px; border: none;
                background-color: var(--accent-color); color: #0f172a;
                font-weight: 600; font-size: 15px; cursor: pointer; transition: opacity 0.2s;
            }
            button:hover { opacity: 0.9; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }

            .status-bar {
                display: none; padding: 12px 16px; background-color: var(--card-bg);
                border-left: 4px solid var(--accent-color); border-radius: 4px;
                margin-bottom: 20px; font-size: 14px; color: #cbd5e1;
            }

            .explorer-layout {
                display: none;
                gap: 20px;
                height: 650px;
            }
            .sidebar {
                width: 280px;
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                overflow-y: auto;
                padding: 10px;
            }
            .sidebar-title {
                font-size: 11px;
                text-transform: uppercase;
                color: #94a3b8;
                padding: 6px 8px;
                letter-spacing: 0.05em;
                border-bottom: 1px solid var(--border-color);
                margin-bottom: 8px;
            }
            .file-item {
                padding: 8px 10px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 13px;
                font-family: monospace;
                color: #cbd5e1;
                margin-bottom: 4px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .file-item:hover { background-color: #334155; color: var(--accent-color); }
            .file-item.active { background-color: #0284c7; color: #ffffff; font-weight: bold; }

            .main-viewer {
                flex: 1;
                display: flex;
                flex-direction: column;
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                overflow: hidden;
            }
            .viewer-header {
                background-color: #090d16;
                padding: 10px 16px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .active-path {
                font-family: monospace;
                font-size: 14px;
                color: var(--accent-color);
            }
            
            .content-split {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .code-editor {
                flex: 1;
                padding: 14px;
                margin: 0;
                background-color: #090d16;
                color: #e2e8f0;
                font-family: Consolas, Monaco, "Andale Mono", monospace;
                font-size: 13px;
                line-height: 1.5;
                overflow: auto;
                white-space: pre;
                border-bottom: 1px solid var(--border-color);
            }

            .prompt-panel {
                background-color: #1e293b;
                padding: 14px;
                border-top: 1px solid var(--border-color);
            }
            .prompt-input-row {
                display: flex;
                gap: 10px;
            }
            textarea.prompt-box {
                flex: 1;
                height: 48px;
                resize: none;
                font-family: inherit;
            }

            .agent-output-card {
                display: none;
                background-color: #0f172a;
                padding: 20px;
                border-top: 2px solid var(--accent-color);
                max-height: 250px;
                overflow-y: auto;
                line-height: 1.6;
                font-size: 14px;
            }
            .agent-output-card h1, .agent-output-card h2, .agent-output-card h3 {
                color: #f1f5f9;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 4px;
                margin-top: 12px;
            }
            .agent-output-card code {
                background-color: #1e293b;
                padding: 2px 6px;
                border-radius: 4px;
                color: #38bdf8;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GitHub Repo Vulnerability Scanner</h1>
            <p class="subtitle">Fetch repository files, choose a file, and scan for vulnerabilities.</p>
            
            <div class="input-group">
                <input type="text" id="repoInput" placeholder="e.g. https://github.com/psf/requests">
                <button id="submitBtn" onclick="inspectRepo()">Fetch Repo Tree</button>
            </div>

            <div id="statusBar" class="status-bar">
                ⚡ Fetching repository files from GitHub API...
            </div>

            <div id="explorerLayout" class="explorer-layout">
                <!-- Left Sidebar: File Tree -->
                <div class="sidebar">
                    <div class="sidebar-title" id="fileCountTitle">Files (0)</div>
                    <div id="fileTreeList"></div>
                </div>

                <!-- Right Pane: Code View & Prompt Bar -->
                <div class="main-viewer">
                    <div class="viewer-header">
                        <span id="activePath" class="active-path">Select a file from sidebar</span>
                    </div>
                    
                    <div class="content-split">
                        <pre id="codeEditor" class="code-editor">// Select a file from the sidebar to view code...</pre>
                        
                        <div class="prompt-panel">
                            <div class="prompt-input-row">
                                
                                <button id="agentBtn" onclick="runAgentOnFile()">Scan for Vulnerabilities</button>
                            </div>
                        </div>

                        <div id="agentOutput" class="agent-output-card"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentRepoFiles = [];
            let activeIndex = -1;

            async function inspectRepo() {
                const repoInput = document.getElementById("repoInput");
                const submitBtn = document.getElementById("submitBtn");
                const statusBar = document.getElementById("statusBar");
                const explorerLayout = document.getElementById("explorerLayout");
                const fileTreeList = document.getElementById("fileTreeList");

                const url = repoInput.value.trim();
                if (!url) return;

                submitBtn.disabled = true;
                statusBar.style.display = "block";
                explorerLayout.style.display = "none";
                fileTreeList.innerHTML = "";

                try {
                    const response = await fetch("/api/v1/inspect", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ repo_url: url })
                    });

                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.detail || "Failed to fetch repository.");
                    }

                    const data = await response.json();
                    currentRepoFiles = data.files;

                    document.getElementById("fileCountTitle").textContent = `Files (${data.total_files})`;

                    data.files.forEach((file, index) => {
                        const item = document.createElement("div");
                        item.className = "file-item";
                        item.textContent = "📄 " + file.path;
                        item.onclick = () => selectFile(index);
                        fileTreeList.appendChild(item);
                    });

                    explorerLayout.style.display = "flex";
                    
                    if (data.files.length > 0) {
                        selectFile(0);
                    }
                } catch (err) {
                    alert("Error: " + err.message);
                } finally {
                    submitBtn.disabled = false;
                    statusBar.style.display = "none";
                }
            }

            function selectFile(index) {
                activeIndex = index;
                const file = currentRepoFiles[index];

                const items = document.querySelectorAll(".file-item");
                items.forEach((item, i) => {
                    if (i === index) item.classList.add("active");
                    else item.classList.remove("active");
                });

                document.getElementById("activePath").textContent = file.path;
                document.getElementById("codeEditor").textContent = file.content;
                document.getElementById("agentOutput").style.display = "none";
            }

            async function runAgentOnFile() {
                if (activeIndex === -1) return;

                const file = currentRepoFiles[activeIndex];
                const customPrompt = document.getElementById("customPrompt").value;
                const agentBtn = document.getElementById("agentBtn");
                const agentOutput = document.getElementById("agentOutput");

                agentBtn.disabled = true;
                agentOutput.style.display = "block";
                agentOutput.innerHTML = "<em>⚡ Running Scanner on " + file.path + "...</em>";

                try {
                    const response = await fetch("/api/v1/agent-process", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            file_path: file.path,
                            code_content: file.content,
                            custom_prompt: customPrompt
                        })
                    });

                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.detail || "LLM request failed.");
                    }

                    const data = await response.json();
                    
                    // Render Gemini response with Markdown
                    agentOutput.innerHTML = marked.parse(data.ai_response);
                } catch (err) {
                    agentOutput.innerHTML = "<span style='color:#f87171;'>Error: " + err.message + "</span>";
                } finally {
                    agentBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("repo_agent:app", host="127.0.0.1", port=8000, reload=True)