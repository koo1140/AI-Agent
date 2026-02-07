#!/usr/bin/env python3
"""
AI AGENT LEVEL 3 - ARCHITECT EDITION
Powered by Mistral AI | Features: Rich UI, Real Sub-Agents, Smart Context
"""

import os
import json
import time
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# -----------------------------------------------------------------------------
# 1. DEPENDENCY CHECK & INSTALLATION
# -----------------------------------------------------------------------------
def install_dependencies():
    """Installs missing dependencies automatically."""
    required = ["mistralai", "rich", "requests"]
    installed = False
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            installed = True
    if installed:
        print("Dependencies installed. Restarting...")
        os.execv(sys.executable, ['python'] + sys.argv)

install_dependencies()

# Imports after installation
from mistralai import Mistral
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.style import Style
from rich.theme import Theme
from rich.status import Status
from rich.traceback import install as install_rich_traceback
import requests

# -----------------------------------------------------------------------------
# 2. CONFIGURATION & THEMES
# -----------------------------------------------------------------------------
install_rich_traceback()

# Custom Theme for semantic highlighting
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "success": "bold green",
    "user.prompt": "bold gold1 on #332b00",
    "agent.style": "misty_rose1 on #1a0505",  # Combined style for agent
    "tool.name": "bold blue",
    "tool.path": "underline cyan",
})

console = Console(theme=custom_theme)

CONFIG = {
    "model_main": "mistral-large-latest",
    "model_sub": "mistral-large-latest", # Switch to 'mistral-small-latest' to save cost
    "api_delay": 1.2, # Seconds to wait between API calls (The "Heartbeat")
    "max_loops": 20,  # Safety limit for autonomy
    "workspace": "./workspace"
}

# -----------------------------------------------------------------------------
# 3. UI COMPONENTS
# -----------------------------------------------------------------------------
class AgentUI:
    """Handles the visual presentation of the agent's thought process."""
    
    @staticmethod
    def header():
        console.clear()
        title = """[bold white]AI AGENT[/bold white] [bold orange1]LEVEL 3 ARCHITECT[/bold orange1]
[dim]Powered by Mistral AI • Smart Context • Surgical Tools[/dim]"""
        console.print(Panel(title, border_style="orange1", expand=False))
        console.print(f"[dim]Running Model: {CONFIG['model_main']} | Delay: {CONFIG['api_delay']}s[/dim]\n")

    @staticmethod
    def user_input() -> str:
        console.print(f"\n[user.prompt] USER [/user.prompt] > ", end="")
        return input()

    @staticmethod
    def assistant_message(content: str):
        # Render markdown content elegantly
        # FIX: Use direct color values instead of theme references in style parameter
        md = Markdown(content)
        console.print(Panel(md, title="[bold]AGENT[/bold]", border_style="red", style="misty_rose1 on #1a0505"))

    @staticmethod
    def step_log(step_num: int, msg: str, is_sub: bool = False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "  ┃" if is_sub else ""
        icon = "⚡" if is_sub else "🔹"
        color = "yellow" if is_sub else "cyan"
        console.print(f"[dim]{timestamp}[/dim] {prefix} [{color}]{icon} Step {step_num}:[/{color}] {msg}")

    @staticmethod
    def tool_use(name: str, summary: str, is_sub: bool = False):
        prefix = "  ┃  " if is_sub else "  "
        console.print(f"{prefix}[dim]>> Executing:[/dim] [tool.name]{name}[/tool.name] [dim]({summary})[/dim]")

    @staticmethod
    def sub_agent_start(role: str, task: str):
        console.print(f"\n  [bold magenta]╭── ⚡ SPAWNING SUB-AGENT: {role.upper()} ──╮[/bold magenta]")
        console.print(f"  [magenta]│ Task: {task}[/magenta]")

    @staticmethod
    def sub_agent_end(summary: str):
        console.print(f"  [bold magenta]╰── ✅ SUB-AGENT FINISHED ──────────────╯[/bold magenta]")
        console.print(f"  [dim]Result: {summary[:100]}...[/dim]\n")

    @staticmethod
    def error(msg: str):
        console.print(f"[error]ERROR: {msg}[/error]")

# -----------------------------------------------------------------------------
# 4. SURGICAL TOOLS
# -----------------------------------------------------------------------------
class Toolbox:
    def __init__(self):
        self.work_dir = Path(CONFIG["workspace"])
        self.work_dir.mkdir(exist_ok=True)
        self.memory_file = Path("memory/agent_memory.json")
        self.memory = self._load_memory()

    def _load_memory(self):
        if self.memory_file.exists():
            return json.loads(self.memory_file.read_text())
        return {}

    def _save_memory(self):
        self.memory_file.parent.mkdir(exist_ok=True)
        self.memory_file.write_text(json.dumps(self.memory, indent=2))

    # --- FILE TOOLS ---
    def file_info(self, path: str) -> str:
        p = Path(path)
        if not p.exists(): return f"Error: {path} not found"
        try:
            lines = sum(1 for _ in open(p, encoding='utf-8', errors='ignore'))
            return f"File: {path} | Size: {p.stat().st_size} bytes | Lines: {lines}"
        except Exception as e: return f"Error reading info: {e}"

    def read_lines(self, path: str, start: int = 1, end: int = 50) -> str:
        p = Path(path)
        if not p.exists(): return f"Error: {path} not found"
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            # Handle end=-1 for "read until end"
            if end == -1: end = len(lines)
            
            selected = lines[start-1:end]
            numbered = "".join([f"{i+start}| {line}" for i, line in enumerate(selected)])
            return f"--- {path} Lines {start}-{end} ---\n{numbered}"
        except Exception as e: return f"Read Error: {e}"

    def edit_lines(self, path: str, start: int, end: int, new_content: str) -> str:
        p = Path(path)
        if not p.exists(): return f"Error: {path} not found"
        try:
            with open(p, 'r', encoding='utf-8') as f: lines = f.readlines()
            
            # Allow empty string to delete lines
            new_lines_list = [l + '\n' for l in new_content.split('\n')] if new_content else []
            # Remove last newline if it was added artificially to an empty list
            if new_lines_list and new_lines_list[-1] == "\n" and not new_content.endswith("\n"):
                 new_lines_list[-1] = new_lines_list[-1].strip()

            # Python list slicing is zero-based, lines are 1-based
            lines[start-1:end] = new_lines_list
            with open(p, 'w', encoding='utf-8') as f: f.writelines(lines)
            return f"Successfully modified lines {start}-{end} in {path}"
        except Exception as e: return f"Edit Error: {e}"

    def write_new_file(self, path: str, content: str) -> str:
        p = Path(path)
        if p.exists(): return f"Error: {path} exists. Use edit_lines."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return f"Created {path}"

    def tree(self, directory: str = ".", depth: int = 2) -> str:
        # Simplified tree for context saving
        output = []
        root = Path(directory)
        if not root.exists(): return "Directory not found"
        
        for path in sorted(root.rglob('*')):
            try:
                level = len(path.relative_to(root).parts)
                if level <= depth:
                    indent = '  ' * (level - 1)
                    mark = "📁" if path.is_dir() else "📄"
                    output.append(f"{indent}{mark} {path.name}")
            except ValueError: continue # Path issue
            
        return "\n".join(output)

    # --- MEMORY & SEARCH ---
    def memory_store(self, key: str, value: str) -> str:
        self.memory[key] = value
        self._save_memory()
        return f"Stored info under key '{key}'"

    def memory_recall(self, key: str) -> str:
        return self.memory.get(key, "Key not found in memory.")

    def web_search(self, query: str) -> str:
        # DuckDuckGo API (Lite)
        try:
            r = requests.get("https://api.duckduckgo.com/", params={'q': query, 'format': 'json', 'no_html': 1})
            data = r.json()
            return data.get('AbstractText', "No abstract found.") or "No results."
        except: return "Search failed (No internet?)"

    def execute_shell(self, command: str) -> str:
        # Safety: Restricted to non-destructive commands ideally, but this is "Architect Level"
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            return f"Exit: {res.returncode}\n{res.stdout}\n{res.stderr}"
        except Exception as e: return f"Shell Error: {e}"

# -----------------------------------------------------------------------------
# 5. AGENT BRAIN (Reusable for Main & Sub)
# -----------------------------------------------------------------------------
class AgentBrain:
    def __init__(self, model: str, toolbox: Toolbox):
        self.client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        self.model = model
        self.toolbox = toolbox
        self.tools = self._define_tools()
    
    def run_cycle(self, messages: List[Dict], is_sub_agent: bool = False) -> str:
        """The core ReAct loop: Think -> Act -> Observe"""
        
        loops = 0
        while loops < CONFIG["max_loops"]:
            loops += 1
            
            # 1. API Call (Think)
            # The "Heartbeat" delay to prevent rate limits and give UX breathing room
            time.sleep(CONFIG["api_delay"])
            
            try:
                if not is_sub_agent:
                    with console.status("[bold green]Thinking...", spinner="dots"):
                         response = self.client.chat.complete(model=self.model, messages=messages, tools=self.tools)
                else:
                    # Sub-agents don't get the main spinner to avoid UI clutter
                    response = self.client.chat.complete(model=self.model, messages=messages, tools=self.tools)
            except Exception as e:
                return f"API Error: {e}"

            msg = response.choices[0].message
            messages.append(msg)

            # 2. Check for Tool Calls (Act)
            if not msg.tool_calls:
                return msg.content # Final answer found

            # 3. Execute Tools (Observe)
            for tool in msg.tool_calls:
                fn_name = tool.function.name
                args = json.loads(tool.function.arguments)
                
                # UX: Show concise action
                summary = self._get_tool_summary(fn_name, args)
                AgentUI.tool_use(fn_name, summary, is_sub=is_sub_agent)
                
                # Special Case: Spawn Sub-Agent
                if fn_name == "spawn_sub_agent":
                    result = self._spawn_sub_agent(args["role"], args["task"], args.get("context", ""))
                else:
                    result = self._execute_tool(fn_name, args)
                
                # Append result to history
                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": str(result),
                    "tool_call_id": tool.id
                })
                
                # UX: Update Trace
                AgentUI.step_log(loops, f"Completed {fn_name}", is_sub=is_sub_agent)

        return "Error: Maximum iteration limit reached."

    def _execute_tool(self, name: str, args: Dict) -> str:
        t = self.toolbox
        try:
            if name == "file_info": return t.file_info(args["path"])
            if name == "read_lines": return t.read_lines(args["path"], args.get("start", 1), args.get("end", 50))
            if name == "edit_lines": return t.edit_lines(args["path"], args["start"], args["end"], args["new_content"])
            if name == "write_new_file": return t.write_new_file(args["path"], args["content"])
            if name == "tree": return t.tree(args.get("directory", "."), args.get("depth", 2))
            if name == "memory_store": return t.memory_store(args["key"], args["value"])
            if name == "memory_recall": return t.memory_recall(args["key"])
            if name == "web_search": return t.web_search(args["query"])
            if name == "execute_shell": return t.execute_shell(args["command"])
            return f"Unknown tool: {name}"
        except Exception as e: return f"Tool Execution Failed: {e}"

    def _spawn_sub_agent(self, role: str, task: str, context: str) -> str:
        AgentUI.sub_agent_start(role, task)
        
        # New Brain for the sub-agent
        sub_brain = AgentBrain(model=CONFIG["model_sub"], toolbox=self.toolbox)
        
        sys_prompt = f"""You are a sub-agent. Role: {role}.
Task: {task}
Context: {context}
Constraint: You must perform the task and return a CONCISE summary.
Do not ask the user for input. Use tools to verify your work."""
        
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Begin."}]
        
        result = sub_brain.run_cycle(msgs, is_sub_agent=True)
        
        AgentUI.sub_agent_end(result)
        return f"Sub-Agent '{role}' finished. Report: {result}"

    def _get_tool_summary(self, name: str, args: Dict) -> str:
        if name == "read_lines": return f"{args['path']} ({args.get('start')}-{args.get('end')})"
        if name == "edit_lines": return f"{args['path']} (Lines {args['start']}-{args['end']})"
        if name == "web_search": return f"Query: {args['query']}"
        if name == "spawn_sub_agent": return f"Role: {args['role']}"
        return str(args)[:50] + "..."

    def _define_tools(self):
        return [
            {"type": "function", "function": {"name": "file_info", "description": "Check file size/lines", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "read_lines", "description": "Read file segment", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "edit_lines", "description": "Replace lines in file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}, "new_content": {"type": "string"}}, "required": ["path", "start", "end", "new_content"]}}},
            {"type": "function", "function": {"name": "write_new_file", "description": "Create NEW file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "tree", "description": "List directory", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "depth": {"type": "integer"}}, "required": []}}},
            {"type": "function", "function": {"name": "spawn_sub_agent", "description": "Delegate complex task", "parameters": {"type": "object", "properties": {"role": {"type": "string"}, "task": {"type": "string"}, "context": {"type": "string"}}, "required": ["role", "task"]}}},
            {"type": "function", "function": {"name": "web_search", "description": "Search internet", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "execute_shell", "description": "Run shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "memory_store", "description": "Save info", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
            {"type": "function", "function": {"name": "memory_recall", "description": "Get info", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
        ]

# -----------------------------------------------------------------------------
# 6. MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    if not os.environ.get("MISTRAL_API_KEY"):
        AgentUI.error("MISTRAL_API_KEY not found in environment variables.")
        return

    toolbox = Toolbox()
    brain = AgentBrain(model=CONFIG["model_main"], toolbox=toolbox)
    
    # Persistent Chat History
    history = [{
        "role": "system", 
        "content": "You are a Level 3 AI Agent. You are an Architect. You use tools surgically. You spawn sub-agents for heavy lifting. Use 'memory_store' to save facts."
    }]

    AgentUI.header()

    while True:
        try:
            user_in = AgentUI.user_input()
            if user_in.lower() in ["exit", "quit"]: break
            if not user_in.strip(): continue

            history.append({"role": "user", "content": user_in})
            
            # Run the agent cycle
            response = brain.run_cycle(history)
            
            # Display final response
            AgentUI.assistant_message(response)
            
            # Keep history manageable (last 20 messages)
            if len(history) > 20:
                history = [history[0]] + history[-19:]
                
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user.[/bold red]")
            break
        except Exception as e:
            AgentUI.error(str(e))

if __name__ == "__main__":
    main()