#!/usr/bin/env python3
"""
AI Agent Level 6 - ARCHITECT Implementation
Powered by Mistral AI | Features: Smart File Ops, Real Sub-Agents, Context Management
"""

import os
import json
import time
import subprocess
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import sys

# -----------------------------------------------------------------------------
# Dependency Check & Initialization
# -----------------------------------------------------------------------------
def install_dependencies():
    packages = ["colorama", "mistralai", "requests"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

install_dependencies()

from colorama import Fore, Back, Style, init
from mistralai import Mistral

init(autoreset=True)

# -----------------------------------------------------------------------------
# UI / UX Components
# -----------------------------------------------------------------------------
class BeautifulUI:
    """Handles all terminal UI rendering with advanced formatting"""
    
    @staticmethod
    def header():
        print(f"\n{Fore.CYAN}{'═'*80}")
        print(f"{Fore.CYAN}║ {Fore.YELLOW}{Style.BRIGHT}{'AI AGENT LEVEL 6':^76}{Style.RESET_ALL}{Fore.CYAN} ║")
        print(f"{Fore.CYAN}║ {Fore.WHITE}{'Architect Edition: Smart Tools, Sub-Agents & Memory':^76}{Fore.CYAN} ║")
        print(f"{Fore.CYAN}║ {Fore.MAGENTA}{'Powered by Mistral AI':^76}{Fore.CYAN} ║")
        print(f"{Fore.CYAN}{'═'*80}\n")
    
    @staticmethod
    def system_msg(msg: str, status: str = "INFO", sub_indent: bool = False):
        icons = {
            "INFO": f"{Fore.BLUE}ℹ",
            "SUCCESS": f"{Fore.GREEN}✓",
            "ERROR": f"{Fore.RED}✗",
            "WARNING": f"{Fore.YELLOW}⚠",
            "PROCESS": f"{Fore.CYAN}⚙",
            "AGENT": f"{Fore.MAGENTA}🤖",
            "SUBAGENT": f"{Fore.MAGENTA}⚡",
            "MEMORY": f"{Fore.YELLOW}💾",
            "FILE": f"{Fore.CYAN}📄",
            "THINK": f"{Fore.MAGENTA}🧠"
        }
        icon = icons.get(status, icons["INFO"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        indent = "    " if sub_indent else ""
        print(f"{indent}{Fore.WHITE}[{timestamp}] {icon} {Style.BRIGHT}{msg}{Style.RESET_ALL}")
    
    @staticmethod
    def ai_thinking(text: str = "AI is thinking...", sub_agent: bool = False):
        indent = "    " if sub_agent else ""
        print(f"\n{indent}{Fore.MAGENTA}🧠 {text}{Style.RESET_ALL}")
    
    @staticmethod
    def tool_call(tool_name: str, inputs: Dict, sub_agent: bool = False):
        indent = "    " if sub_agent else ""
        color = Fore.GREEN if not sub_agent else Fore.YELLOW
        
        # Simplify display for certain inputs
        display_inputs = inputs.copy()
        if "content" in display_inputs and len(display_inputs["content"]) > 100:
            display_inputs["content"] = f"<...{len(display_inputs['content'])} chars...>"
            
        print(f"\n{indent}{color}╔═══ TOOL: {tool_name.upper()} {'═'*(50-len(tool_name))}╗")
        params_str = json.dumps(display_inputs, indent=2)
        for line in params_str.split('\n'):
            print(f"{indent}{color}║ {Fore.CYAN}{line:<70}{color} ║")
        print(f"{indent}{color}╚{'═'*75}╝")
    
    @staticmethod
    def tool_result(result: str, sub_agent: bool = False):
        indent = "    " if sub_agent else ""
        lines = result.split('\n')
        preview_len = 10 if len(lines) > 10 else len(lines)
        
        print(f"{indent}{Fore.CYAN}▶ RESULT ({len(result)} chars):{Style.RESET_ALL}")
        for line in lines[:preview_len]:
            print(f"{indent}{Fore.WHITE}  {line}")
        if len(lines) > preview_len:
            print(f"{indent}{Fore.WHITE}  ... ({len(lines)-preview_len} more lines)")
            
    @staticmethod
    def sub_agent_start(role: str, task: str):
        print(f"\n{Fore.MAGENTA}╭{'─'*78}╮")
        print(f"{Fore.MAGENTA}│ {Fore.YELLOW}{Style.BRIGHT}⚡ SPAWNING SUB-AGENT: {role.upper():<54}{Fore.MAGENTA}│")
        print(f"{Fore.MAGENTA}│ {Fore.WHITE}Task: {task[:70]:<68}{Fore.MAGENTA}   │")
        print(f"{Fore.MAGENTA}╰{'─'*78}╯")

    @staticmethod
    def user_prompt():
        print(f"\n{Fore.CYAN}{'─'*80}")
        return input(f"{Fore.GREEN}{Style.BRIGHT}YOU >{Style.RESET_ALL} ")
    
    @staticmethod
    def ai_response(text: str):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}AGENT >{Style.RESET_ALL} {text}\n")
    
    @staticmethod
    def error_box(error_type: str, details: str):
        print(f"\n{Fore.RED}╔{'═'*78}╗")
        print(f"{Fore.RED}║ {Style.BRIGHT}ERROR: {error_type:<69}{Style.RESET_ALL}{Fore.RED}║")
        print(f"{Fore.RED}║ {details[:76]:<76} ║")
        print(f"{Fore.RED}╚{'═'*78}╝\n")

# -----------------------------------------------------------------------------
# Smart File & Directory Tools
# -----------------------------------------------------------------------------
class SmartFileTools:
    """Never reads full files blindly. Operates at line-level precision."""

    def file_info(self, filepath: str) -> str:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found: {filepath}"

        stat = path.stat()
        line_count = 0
        is_binary = False
        try:
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                for line_count, _ in enumerate(f, 1): pass
        except UnicodeDecodeError:
            is_binary = True
            line_count = 0

        size_label = "LARGE" if stat.st_size > 50000 else ("medium" if stat.st_size > 10000 else "small")

        return json.dumps({
            "path": filepath,
            "size_bytes": stat.st_size,
            "size_label": size_label,
            "line_count": line_count,
            "is_binary": is_binary,
            "extension": path.suffix,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "recommendation": "Use read_lines with range" if line_count > 100 else "Safe to read fully"
        }, indent=2)

    def read_lines(self, filepath: str, start: int = 1, end: int = 50) -> str:
        path = Path(filepath)
        if not path.exists(): return f"Error: File not found: {filepath}"
        
        MAX_LINES = 200
        lines = []
        total_lines = 0
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                total_lines = i
                if end == -1:
                    if i >= start and len(lines) < MAX_LINES: lines.append((i, line))
                elif start <= i <= end:
                    lines.append((i, line))
                elif i > end: break
        
        if not lines: return f"No lines found in range {start}-{end} (file has {total_lines} lines)"
        
        result = f"File: {filepath} | Showing lines {lines[0][0]}-{lines[-1][0]} of {total_lines}\n"
        result += "─" * 60 + "\n"
        for num, content in lines:
            result += f"{num:>5} │ {content}"
            if not content.endswith('\n'): result += '\n'
        result += "─" * 60
        
        if len(lines) == MAX_LINES and end == -1:
            result += f"\n⚠ Output capped at {MAX_LINES} lines. Use a narrower range."
            
        return result

    def search_in_file(self, filepath: str, pattern: str, context_lines: int = 2) -> str:
        import re
        path = Path(filepath)
        if not path.exists(): return f"Error: File not found: {filepath}"
        
        try: regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e: return f"Invalid regex: {e}"
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            
        matches = [i for i, line in enumerate(all_lines) if regex.search(line)]
        if not matches: return f"No matches for '{pattern}' in {filepath}"
        
        # Merge contexts
        blocks = []
        for m in matches[:20]: # Cap matches
            start = max(0, m - context_lines)
            end = min(len(all_lines), m + context_lines + 1)
            if not blocks or start > blocks[-1][1]:
                blocks.append((start, end))
            else:
                blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
                
        result = f"Found {len(matches)} matches for '{pattern}' in {filepath}\n" + "─" * 60 + "\n"
        for start, end in blocks:
            if start > 0: result += "      ┄┄┄\n"
            for i in range(start, end):
                marker = ">>> " if i in matches else "    "
                result += f"{marker}{i+1:>5} │ {all_lines[i]}"
        
        return result

    def edit_lines(self, filepath: str, start: int, end: int, new_content: str) -> str:
        path = Path(filepath)
        if not path.exists(): return f"Error: File not found: {filepath}"
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            
        if start < 1 or start > len(all_lines):
            return f"Error: start line {start} out of range"
            
        new_lines = [l + '\n' if not l.endswith('\n') else l for l in new_content.split('\n')]
        
        # Fix for empty new content leaving a trailing newline issue
        if len(new_lines) == 1 and new_lines[0] == "\n" and new_content == "":
             new_lines = []

        all_lines[start-1:end] = new_lines
        
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
            
        return f"Edited {filepath}: Replaced lines {start}-{end} with {len(new_lines)} new lines."

    def insert_lines(self, filepath: str, after_line: int, content: str) -> str:
        path = Path(filepath)
        if not path.exists(): return f"Error: File not found: {filepath}"
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
            
        new_lines = [l + '\n' if not l.endswith('\n') else l for l in content.split('\n')]
        all_lines[after_line:after_line] = new_lines
        
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
            
        return f"Inserted {len(new_lines)} lines after line {after_line} in {filepath}"

    def write_new_file(self, filepath: str, content: str) -> str:
        path = Path(filepath)
        if path.exists():
            return f"Error: {filepath} exists. Use edit_lines/insert_lines to modify."
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Created new file {filepath} ({len(content)} bytes)"

    def delete_file(self, filepath: str) -> str:
        path = Path(filepath)
        if not path.exists(): return f"Error: Not found: {filepath}"
        path.unlink()
        return f"Deleted {filepath}"


class SmartDirectoryTools:
    """Smart directory operations that don't dump massive context."""
    
    def tree(self, directory: str = ".", max_depth: int = 3) -> str:
        root = Path(directory)
        if not root.exists(): return f"Error: {directory} not found"
        
        lines = [f"📁 {root.resolve().name}/"]
        count = 0
        
        def _walk(path, prefix, depth):
            nonlocal count
            if depth > max_depth or count > 100: return
            
            try:
                entries = sorted([e for e in path.iterdir() if not e.name.startswith('.')],
                               key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError: return
            
            for i, entry in enumerate(entries):
                if count > 100:
                    lines.append(f"{prefix}├── ... (truncated)")
                    return
                count += 1
                is_last = (i == len(entries) - 1)
                prefix_next = prefix + ("    " if is_last else "│   ")
                connector = "└── " if is_last else "├── "
                
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}📁 {entry.name}/")
                    _walk(entry, prefix_next, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")

        _walk(root, "", 1)
        return "\n".join(lines)

    def find_files(self, directory: str, name_pattern: str = "*", content_pattern: str = None) -> str:
        root = Path(directory)
        candidates = list(root.rglob(name_pattern))
        
        results = []
        for f in candidates:
            if not f.is_file() or any(p.startswith('.') for p in f.parts): continue
            
            if content_pattern:
                try:
                    if re.search(content_pattern, f.read_text(errors='ignore')):
                        results.append(str(f))
                except: continue
            else:
                results.append(str(f))
                
            if len(results) >= 30: break
            
        return "\n".join(results[:30]) + (f"\n...and {len(candidates)-30} more" if len(candidates)>30 else "")

    def grep(self, directory: str, pattern: str, file_glob: str = "*") -> str:
        root = Path(directory)
        try: regex = re.compile(pattern, re.IGNORECASE)
        except: return "Invalid regex"
        
        results = []
        for f in root.rglob(file_glob):
            if not f.is_file() or '.git' in f.parts: continue
            try:
                with open(f, errors='ignore') as fh:
                    for i, line in enumerate(fh, 1):
                        if regex.search(line):
                            results.append(f"{f}:{i}: {line.strip()[:100]}")
                            if len(results) >= 50: break
            except: continue
            if len(results) >= 50: break
            
        return "\n".join(results) if results else "No matches found."

# -----------------------------------------------------------------------------
# Tool Executor
# -----------------------------------------------------------------------------
class ToolExecutor:
    def __init__(self):
        self.workspace = Path("./workspace")
        self.workspace.mkdir(exist_ok=True)
        self.file_tools = SmartFileTools()
        self.dir_tools = SmartDirectoryTools()
        self.memory = {}
        self._load_memory()
        
    def execute(self, tool_name: str, tool_input: Dict) -> str:
        try:
            # File Ops
            if tool_name == "file_info": return self.file_tools.file_info(tool_input["path"])
            elif tool_name == "read_lines": return self.file_tools.read_lines(tool_input["path"], tool_input.get("start", 1), tool_input.get("end", 50))
            elif tool_name == "search_in_file": return self.file_tools.search_in_file(tool_input["path"], tool_input["pattern"], tool_input.get("context_lines", 3))
            elif tool_name == "edit_lines": return self.file_tools.edit_lines(tool_input["path"], tool_input["start"], tool_input["end"], tool_input["new_content"])
            elif tool_name == "insert_lines": return self.file_tools.insert_lines(tool_input["path"], tool_input["after_line"], tool_input["content"])
            elif tool_name == "write_new_file": return self.file_tools.write_new_file(tool_input["path"], tool_input["content"])
            elif tool_name == "delete_file": return self.file_tools.delete_file(tool_input["path"])
            
            # Directory Ops
            elif tool_name == "tree": return self.dir_tools.tree(tool_input.get("directory", "."), tool_input.get("max_depth", 3))
            elif tool_name == "find_files": return self.dir_tools.find_files(tool_input["directory"], tool_input.get("name_pattern", "*"), tool_input.get("content_pattern"))
            elif tool_name == "grep": return self.dir_tools.grep(tool_input["directory"], tool_input["pattern"], tool_input.get("file_glob", "*"))
            
            # System & Web
            elif tool_name == "execute_shell": return self._execute_shell(tool_input["command"])
            elif tool_name == "web_search": return self._web_search(tool_input["query"])
            elif tool_name == "http_request": return self._http_request(tool_input.get("method", "GET"), tool_input["url"], tool_input.get("headers"), tool_input.get("body"))
            
            # Memory & Interaction
            elif tool_name == "memory_store": return self._memory_store(tool_input["key"], tool_input["value"])
            elif tool_name == "memory_recall": return self._memory_recall(tool_input.get("key"))
            elif tool_name == "ask_user": return self._ask_user(tool_input["question"])
            elif tool_name == "task_complete": return f"TASK COMPLETE: {tool_input['summary']}"
            
            else: return f"Error: Unknown tool {tool_name}"
            
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"

    def _execute_shell(self, command: str) -> str:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        out = f"Exit: {result.returncode}\nSTDOUT:\n{result.stdout[:2000]}"
        if result.stderr: out += f"\nSTDERR:\n{result.stderr[:1000]}"
        return out

    def _web_search(self, query: str) -> str:
        try:
            r = requests.get("https://api.duckduckgo.com/", params={'q': query, 'format': 'json', 'no_html': 1}, timeout=10)
            data = r.json()
            if data.get('AbstractText'): return f"{data['Heading']}: {data['AbstractText']}"
            if data.get('RelatedTopics'): return "\n".join([t['Text'] for t in data['RelatedTopics'][:5] if 'Text' in t])
            return "No text results found."
        except Exception as e: return f"Search failed: {e}"

    def _http_request(self, method, url, headers, body):
        try:
            r = requests.request(method, url, headers=headers, data=body, timeout=10)
            return f"Status: {r.status_code}\nBody:\n{r.text[:2000]}"
        except Exception as e: return f"HTTP Error: {e}"

    def _memory_store(self, key, value):
        self.memory[key] = {"value": value, "time": datetime.now().isoformat()}
        self._save_memory()
        return f"Stored '{key}'"

    def _memory_recall(self, key):
        if not key: return "\n".join(self.memory.keys())
        return self.memory.get(key, {}).get("value", "Not found")

    def _ask_user(self, question):
        print(f"\n{Fore.YELLOW}🤖 QUESTION: {question}")
        return input(f"{Fore.GREEN}ANSWER > {Style.RESET_ALL}")

    def _load_memory(self):
        p = Path("./memory/agent_memory.json")
        if p.exists(): self.memory = json.load(open(p))

    def _save_memory(self):
        Path("./memory").mkdir(exist_ok=True)
        json.dump(self.memory, open("./memory/agent_memory.json", "w"), indent=2)

# -----------------------------------------------------------------------------
# Sub-Agent Manager
# -----------------------------------------------------------------------------
class SubAgentManager:
    def __init__(self, client: Mistral, tool_executor: ToolExecutor, tool_definitions: List[Dict]):
        self.client = client
        self.tool_executor = tool_executor
        self.tool_map = {t["function"]["name"]: t for t in tool_definitions}

    def spawn(self, task: str, role: str, allowed_tools: List[str], context: str = "") -> str:
        BeautifulUI.sub_agent_start(role, task)
        
        filtered_tools = [self.tool_map[name] for name in allowed_tools if name in self.tool_map]
        
        system_prompt = f"""You are a specialized sub-agent.
ROLE: {role}
TASK: {task}
CONTEXT: {context}
INSTRUCTIONS:
- Focus ONLY on the task.
- Use allowed tools to complete the work.
- Return a summary when done. Do not ask user questions.
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Start."}]
        
        for i in range(10): # Max 10 iterations
            BeautifulUI.system_msg(f"Iteration {i+1}", "SUBAGENT", sub_indent=True)
            
            try:
                response = self.client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    tools=filtered_tools if filtered_tools else None
                )
            except Exception as e:
                return f"Sub-agent crashed: {e}"

            msg = response.choices[0].message
            if not msg.tool_calls:
                BeautifulUI.system_msg("Sub-agent finished.", "SUCCESS", sub_indent=True)
                return msg.content or "Task completed (no output)."

            messages.append(msg)
            
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                BeautifulUI.tool_call(name, args, sub_agent=True)
                
                if name not in allowed_tools:
                    result = "Error: Tool not authorized."
                else:
                    result = self.tool_executor.execute(name, args)
                
                BeautifulUI.tool_result(result, sub_agent=True)
                
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result,
                    "tool_call_id": tool_call.id
                })
        
        return "Sub-agent timed out."

# -----------------------------------------------------------------------------
# Central Brain
# -----------------------------------------------------------------------------
class CentralBrain:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key: raise ValueError("MISTRAL_API_KEY not found in environment.")
        
        self.client = Mistral(api_key=api_key)
        self.tool_executor = ToolExecutor()
        self.tools = self._get_tool_definitions()
        self.sub_agent_mgr = SubAgentManager(self.client, self.tool_executor, self.tools)
        
        self.messages = []
        self._load_system_prompt()
        BeautifulUI.system_msg("Central Brain Online", "SUCCESS")

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        BeautifulUI.ai_thinking()
        
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=self.messages,
            tools=self.tools
        )
        
        return self._process_response(response)

    def _process_response(self, response) -> str:
        msg = response.choices[0].message
        
        while msg.tool_calls:
            self.messages.append(msg)
            
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                BeautifulUI.tool_call(name, args)
                
                if name == "spawn_sub_agent":
                    result = self.sub_agent_mgr.spawn(
                        args["task"], args["role"], args["allowed_tools"], args.get("context", "")
                    )
                else:
                    result = self.tool_executor.execute(name, args)
                
                BeautifulUI.tool_result(result)
                self.messages.append({
                    "role": "tool", 
                    "name": name, 
                    "content": result, 
                    "tool_call_id": tool_call.id
                })
            
            BeautifulUI.ai_thinking("Processing results...")
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=self.messages,
                tools=self.tools
            )
            msg = response.choices[0].message
            
        if msg.content:
            self.messages.append({"role": "assistant", "content": msg.content})
            return msg.content
        return "Done."

    def _load_system_prompt(self):
        prompt = """You are AI Agent Level 6.
You have access to surgical file tools. NEVER rewrite full files.
Use 'spawn_sub_agent' for complex tasks (research, massive refactors).
Use 'memory_store' to save important context.
"""
        if Path("boot.md").exists():
            prompt = Path("boot.md").read_text()
        self.messages.append({"role": "system", "content": prompt})

    def _get_tool_definitions(self):
        # Condensed definitions for brevity
        return [
            {"type": "function", "function": {"name": "file_info", "description": "Get file metadata (size, lines) before reading.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "read_lines", "description": "Read specific lines. Use start=1, end=-1 for small files.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "search_in_file", "description": "Regex search in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}, "context_lines": {"type": "integer"}}, "required": ["path", "pattern"]}}},
            {"type": "function", "function": {"name": "edit_lines", "description": "Replace line range.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}, "new_content": {"type": "string"}}, "required": ["path", "start", "end", "new_content"]}}},
            {"type": "function", "function": {"name": "insert_lines", "description": "Insert lines after line number.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "after_line": {"type": "integer"}, "content": {"type": "string"}}, "required": ["path", "after_line", "content"]}}},
            {"type": "function", "function": {"name": "write_new_file", "description": "Create new file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "tree", "description": "Show directory structure.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "max_depth": {"type": "integer"}}, "required": []}}},
            {"type": "function", "function": {"name": "find_files", "description": "Find files by name/content.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "name_pattern": {"type": "string"}, "content_pattern": {"type": "string"}}, "required": ["directory"]}}},
            {"type": "function", "function": {"name": "grep", "description": "Search content in directory.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "pattern": {"type": "string"}, "file_glob": {"type": "string"}}, "required": ["directory", "pattern"]}}},
            {"type": "function", "function": {"name": "execute_shell", "description": "Run shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "web_search", "description": "Search web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "http_request", "description": "HTTP Request.", "parameters": {"type": "object", "properties": {"method": {"type": "string"}, "url": {"type": "string"}, "body": {"type": "string"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "memory_store", "description": "Save to persistent memory.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
            {"type": "function", "function": {"name": "memory_recall", "description": "Recall from memory.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": []}}},
            {"type": "function", "function": {"name": "ask_user", "description": "Ask user for input.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
            {"type": "function", "function": {"name": "task_complete", "description": "Mark task done.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
            {"type": "function", "function": {"name": "spawn_sub_agent", "description": "Spawn specialized sub-agent.", "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "role": {"type": "string"}, "allowed_tools": {"type": "array", "items": {"type": "string"}}, "context": {"type": "string"}}, "required": ["task", "role", "allowed_tools"]}}}
        ]

# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    BeautifulUI.header()
    
    try:
        brain = CentralBrain()
        print(f"\n{Fore.YELLOW}💡 Ready! Type 'exit' to quit.{Style.RESET_ALL}\n")
        
        while True:
            try:
                user_input = BeautifulUI.user_prompt()
                if user_input.lower() in ['exit', 'quit']: break
                if not user_input.strip(): continue
                
                response = brain.chat(user_input)
                BeautifulUI.ai_response(response)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                BeautifulUI.error_box("Runtime Error", str(e))
                
    except Exception as e:
        BeautifulUI.error_box("Initialization Failed", str(e))

if __name__ == "__main__":
    main()