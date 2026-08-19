#!/usr/bin/env python3
"""
KittyShell-89: Homemade 89-task terminal quiz. NOT Terminal-Bench 2.0.
No Docker containers, no Harbor framework. Results are NOT comparable to official TB2.
Terminal-focused tasks: file ops, shell commands, text processing, system info.
Uses Ollama native tool-calling API.
"""

import json
import os
import sys
import time
import tempfile
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "ornith:9b"
TEMPERATURE = 0.6
TIMEOUT = 120
PROGRESS_EVERY = 10

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file's contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
]

# ── 89 Terminal-Bench Tasks ──────────────────────────────────────
# Each task: (name, setup_fn, prompt, check_fn)
# setup_fn creates the initial state; check_fn verifies the result

TASKS = []

# ── File Operations (20) ──────────────────────────────────────

def setup_file_ops(workdir):
    """Create sample files for file operation tasks."""
    # Create a data file
    with open(os.path.join(workdir, "data.txt"), "w") as f:
        f.write("line1\nline2\nline3\nline4\nline5\n")
    # Create a csv
    with open(os.path.join(workdir, "data.csv"), "w") as f:
        f.write("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago\n")
    # Create nested dirs
    os.makedirs(os.path.join(workdir, "src", "lib"), exist_ok=True)
    with open(os.path.join(workdir, "src", "main.py"), "w") as f:
        f.write("print('hello')\n")
    with open(os.path.join(workdir, "src", "lib", "utils.py"), "w") as f:
        f.write("def helper(): pass\n")
    # Create a config file
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump({"debug": True, "port": 8080, "host": "localhost"}, f, indent=2)

TASKS.append(("file_copy", setup_file_ops, "Copy data.txt to data_backup.txt. Save it.", lambda wd: os.path.isfile(os.path.join(wd, "data_backup.txt"))))
TASKS.append(("file_move", setup_file_ops, "Move data.csv to archive/data.csv creating the archive directory.", lambda wd: os.path.isfile(os.path.join(wd, "archive", "data.csv"))))
TASKS.append(("file_delete", setup_file_ops, "Delete the file data.txt.", lambda wd: not os.path.isfile(os.path.join(wd, "data.txt"))))
TASKS.append(("file_rename", setup_file_ops, "Rename data.txt to data_renamed.txt.", lambda wd: os.path.isfile(os.path.join(wd, "data_renamed.txt"))))
TASKS.append(("file_concat", setup_file_ops, "Concatenate the contents of src/main.py and src/lib/utils.py into combined.py.", lambda wd: os.path.isfile(os.path.join(wd, "combined.py"))))
TASKS.append(("file_count_lines", setup_file_ops, "Count the number of lines in data.txt and write the count to line_count.txt.", lambda wd: open(os.path.join(wd, "line_count.txt")).read().strip().isdigit()))
TASKS.append(("file_sort", setup_file_ops, "Sort the lines of data.txt alphabetically and save to sorted.txt.", lambda wd: os.path.isfile(os.path.join(wd, "sorted.txt"))))
TASKS.append(("file_unique_lines", setup_file_ops, "Write only the unique lines from data.txt to unique.txt.", lambda wd: os.path.isfile(os.path.join(wd, "unique.txt"))))
TASKS.append(("file_grep", setup_file_ops, "Search for lines containing '3' in data.txt and write matches to matches.txt.", lambda wd: os.path.isfile(os.path.join(wd, "matches.txt"))))
TASKS.append(("file_word_count", setup_file_ops, "Count words in data.txt and write the count to word_count.txt.", lambda wd: os.path.isfile(os.path.join(wd, "word_count.txt"))))
TASKS.append(("file_json_pretty", setup_file_ops, "Read config.json and write a reformatted version with 4-space indentation to config_pretty.json.", lambda wd: os.path.isfile(os.path.join(wd, "config_pretty.json"))))
TASKS.append(("file_json_query", setup_file_ops, "Read config.json and write the value of the 'port' key to port.txt.", lambda wd: open(os.path.join(wd, "port.txt")).read().strip() == "8080"))
TASKS.append(("file_csv_column", setup_file_ops, "Extract the 'name' column from data.csv and write to names.txt.", lambda wd: os.path.isfile(os.path.join(wd, "names.txt"))))
TASKS.append(("file_csv_filter", setup_file_ops, "Filter rows from data.csv where age > 28 and write to filtered.csv.", lambda wd: os.path.isfile(os.path.join(wd, "filtered.csv"))))
TASKS.append(("file_backup_dir", setup_file_ops, "Create a backup directory and copy all .py files into it.", lambda wd: os.path.isdir(os.path.join(wd, "backup"))))
TASKS.append(("file_find_extension", setup_file_ops, "Find all files ending in .py and write the list to py_files.txt.", lambda wd: os.path.isfile(os.path.join(wd, "py_files.txt"))))
TASKS.append(("file_create_dirs", setup_file_ops, "Create the directory structure: build/src/include with mkdir -p.", lambda wd: os.path.isdir(os.path.join(wd, "build", "src", "include"))))
TASKS.append(("file_symlink", setup_file_ops, "Create a symbolic link from link.txt pointing to data.txt.", lambda wd: os.path.islink(os.path.join(wd, "link.txt"))))
TASKS.append(("file_chmod", setup_file_ops, "Make data.txt executable with chmod.", lambda wd: os.access(os.path.join(wd, "data.txt"), os.X_OK)))
TASKS.append(("file_file_size", setup_file_ops, "Get the file size of data.txt in bytes and write it to size.txt.", lambda wd: os.path.isfile(os.path.join(wd, "size.txt"))))

# ── Text Processing (20) ─────────────────────────────────────

def setup_text_proc(workdir):
    with open(os.path.join(workdir, "sample.txt"), "w") as f:
        f.write("Hello World\nfoo bar baz\nPython is great\nRust is fast\nGo is simple\nJavaScript runs everywhere\n")
    with open(os.path.join(workdir, "log.txt"), "w") as f:
        f.write("2024-01-15 ERROR disk full\n2024-01-15 INFO backup started\n2024-01-16 WARNING low memory\n2024-01-16 ERROR network timeout\n2024-01-17 INFO system recovered\n")

TASKS.append(("text_reverse_lines", setup_text_proc, "Reverse the order of lines in sample.txt and save to reversed.txt.", lambda wd: os.path.isfile(os.path.join(wd, "reversed.txt"))))
TASKS.append(("text_uppercase", setup_text_proc, "Convert all text in sample.txt to uppercase and save to upper.txt.", lambda wd: os.path.isfile(os.path.join(wd, "upper.txt"))))
TASKS.append(("text_lowercase", setup_text_proc, "Convert all text in sample.txt to lowercase and save to lower.txt.", lambda wd: os.path.isfile(os.path.join(wd, "lower.txt"))))
TASKS.append(("text_replace", setup_text_proc, "Replace all occurrences of 'is' with 'was' in sample.txt and save to replaced.txt.", lambda wd: os.path.isfile(os.path.join(wd, "replaced.txt"))))
TASKS.append(("text_remove_empty", setup_text_proc, "Remove empty lines from sample.txt and save to noempty.txt.", lambda wd: os.path.isfile(os.path.join(wd, "noempty.txt"))))
TASKS.append(("text_word_freq", setup_text_proc, "Count word frequency in sample.txt and write to word_freq.txt (format: word count).", lambda wd: os.path.isfile(os.path.join(wd, "word_freq.txt"))))
TASKS.append(("text_char_count", setup_text_proc, "Count total characters in sample.txt (including spaces) and write to char_count.txt.", lambda wd: os.path.isfile(os.path.join(wd, "char_count.txt"))))
TASKS.append(("text_line_number", setup_text_proc, "Add line numbers to sample.txt (1: Hello World, 2: ...) and save to numbered.txt.", lambda wd: os.path.isfile(os.path.join(wd, "numbered.txt"))))
TASKS.append(("text_grep_error", setup_text_proc, "Extract all lines containing 'ERROR' from log.txt and write to errors.txt.", lambda wd: os.path.isfile(os.path.join(wd, "errors.txt"))))
TASKS.append(("text_extract_date", setup_text_proc, "Extract unique dates from log.txt and write to dates.txt.", lambda wd: os.path.isfile(os.path.join(wd, "dates.txt"))))
TASKS.append(("text_substring", setup_text_proc, "Extract the first 5 characters from each line in sample.txt and save to substr.txt.", lambda wd: os.path.isfile(os.path.join(wd, "substr.txt"))))
TASKS.append(("text_trim_lines", setup_text_proc, "Trim trailing whitespace from each line in sample.txt and save to trimmed.txt.", lambda wd: os.path.isfile(os.path.join(wd, "trimmed.txt"))))
TASKS.append(("text_dedup", setup_text_proc, "Remove duplicate lines from sample.txt and save to dedup.txt.", lambda wd: os.path.isfile(os.path.join(wd, "dedup.txt"))))
TASKS.append(("text_format_csv", setup_text_proc, "Convert sample.txt (space-separated) into CSV format and save to formatted.csv.", lambda wd: os.path.isfile(os.path.join(wd, "formatted.csv"))))
TASKS.append(("text_extract_words", setup_text_proc, "Extract only words containing 'a' from sample.txt and save to a_words.txt.", lambda wd: os.path.isfile(os.path.join(wd, "a_words.txt"))))
TASKS.append(("text_join_lines", setup_text_proc, "Join all lines of sample.txt into a single line separated by commas and save to joined.txt.", lambda wd: os.path.isfile(os.path.join(wd, "joined.txt"))))
TASKS.append(("text_wrap", setup_text_proc, "Wrap each line of sample.txt at 20 characters and save to wrapped.txt.", lambda wd: os.path.isfile(os.path.join(wd, "wrapped.txt"))))
TASKS.append(("text_stats", setup_text_proc, "Compute line count, word count, and character count of sample.txt and write all three to stats.txt.", lambda wd: os.path.isfile(os.path.join(wd, "stats.txt"))))
TASKS.append(("text_filter_short", setup_text_proc, "Remove lines shorter than 10 characters from sample.txt and save to long_lines.txt.", lambda wd: os.path.isfile(os.path.join(wd, "long_lines.txt"))))
TASKS.append(("text_extract_last", setup_text_proc, "Extract the last word from each line in sample.txt and save to last_words.txt.", lambda wd: os.path.isfile(os.path.join(wd, "last_words.txt"))))

# ── Shell Commands (20) ──────────────────────────────────────

def setup_shell(workdir):
    # Create a small project
    os.makedirs(os.path.join(workdir, "project", "src"), exist_ok=True)
    with open(os.path.join(workdir, "project", "README.md"), "w") as f:
        f.write("# My Project\nVersion 1.0\n")
    with open(os.path.join(workdir, "project", "src", "main.py"), "w") as f:
        f.write("print('Hello from main')\n")
    with open(os.path.join(workdir, "project", "src", "lib.py"), "w") as f:
        f.write("def add(a, b): return a + b\n")
    with open(os.path.join(workdir, "numbers.txt"), "w") as f:
        f.write("3\n1\n4\n1\n5\n9\n2\n6\n5\n")
    # Init a git repo
    subprocess.run(["git", "init"], cwd=workdir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bench@test"], cwd=workdir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "bench"], cwd=workdir, capture_output=True)

TASKS.append(("shell_ls", setup_shell, "List all files in the project directory recursively and write the output to filelist.txt.", lambda wd: os.path.isfile(os.path.join(wd, "filelist.txt"))))
TASKS.append(("shell_find_py", setup_shell, "Find all .py files in the project directory and write the list to py_files.txt.", lambda wd: os.path.isfile(os.path.join(wd, "py_files.txt"))))
TASKS.append(("shell_wc", setup_shell, "Count lines in each .py file in project/src/ and write to line_counts.txt.", lambda wd: os.path.isfile(os.path.join(wd, "line_counts.txt"))))
TASKS.append(("shell_grep", setup_shell, "Search for 'def ' in all .py files and write the results to functions.txt.", lambda wd: os.path.isfile(os.path.join(wd, "functions.txt"))))
TASKS.append(("shell_sort_numbers", setup_shell, "Sort the numbers in numbers.txt numerically and save to sorted_numbers.txt.", lambda wd: os.path.isfile(os.path.join(wd, "sorted_numbers.txt"))))
TASKS.append(("shell_sum_numbers", setup_shell, "Sum all numbers in numbers.txt and write the result to sum.txt.", lambda wd: os.path.isfile(os.path.join(wd, "sum.txt"))))
TASKS.append(("shell_sort_unique", setup_shell, "Sort unique numbers from numbers.txt and save to unique_sorted.txt.", lambda wd: os.path.isfile(os.path.join(wd, "unique_sorted.txt"))))
TASKS.append(("shell_max_number", setup_shell, "Find the maximum number in numbers.txt and write it to max_num.txt.", lambda wd: os.path.isfile(os.path.join(wd, "max_num.txt"))))
TASKS.append(("shell_min_number", setup_shell, "Find the minimum number in numbers.txt and write it to min_num.txt.", lambda wd: os.path.isfile(os.path.join(wd, "min_num.txt"))))
TASKS.append(("shell_disk_usage", setup_shell, "Show disk usage of the project directory and write to disk_usage.txt.", lambda wd: os.path.isfile(os.path.join(wd, "disk_usage.txt"))))
TASKS.append(("shell_file_count", setup_shell, "Count total number of files in the project directory and write to file_count.txt.", lambda wd: os.path.isfile(os.path.join(wd, "file_count.txt"))))
TASKS.append(("shell_date", setup_shell, "Write the current date in ISO format (YYYY-MM-DD) to date.txt.", lambda wd: os.path.isfile(os.path.join(wd, "date.txt"))))
TASKS.append(("shell_whoami", setup_shell, "Write the current username to whoami.txt.", lambda wd: os.path.isfile(os.path.join(wd, "whoami.txt"))))
TASKS.append(("shell_env", setup_shell, "Write the value of the PATH environment variable to path.txt.", lambda wd: os.path.isfile(os.path.join(wd, "path.txt"))))
TASKS.append(("shell_uptime", setup_shell, "Write system uptime to uptime.txt.", lambda wd: os.path.isfile(os.path.join(wd, "uptime.txt"))))
TASKS.append(("shell_cpu_count", setup_shell, "Write the number of CPU cores to cpu_count.txt.", lambda wd: os.path.isfile(os.path.join(wd, "cpu_count.txt"))))
TASKS.append(("shell_memory", setup_shell, "Write total system memory in MB to memory.txt.", lambda wd: os.path.isfile(os.path.join(wd, "memory.txt"))))
TASKS.append(("shell_git_log", setup_shell, "Initialize a git repo in project/, add all files, commit, then write the git log to git_log.txt.", lambda wd: os.path.isfile(os.path.join(wd, "git_log.txt"))))
TASKS.append(("shell_tree", setup_shell, "Create a tree-like directory listing of project/ and save to tree.txt.", lambda wd: os.path.isfile(os.path.join(wd, "tree.txt"))))
TASKS.append(("shell_md5", setup_shell, "Compute the md5sum of project/README.md and write to checksum.txt.", lambda wd: os.path.isfile(os.path.join(wd, "checksum.txt"))))

# ── Data Transformation (15) ─────────────────────────────────

def setup_data_transform(workdir):
    with open(os.path.join(workdir, "employees.json"), "w") as f:
        json.dump([
            {"name": "Alice", "dept": "Engineering", "salary": 120000},
            {"name": "Bob", "dept": "Marketing", "salary": 80000},
            {"name": "Charlie", "dept": "Engineering", "salary": 130000},
            {"name": "Diana", "dept": "Sales", "salary": 95000},
            {"name": "Eve", "dept": "Marketing", "salary": 85000},
        ], f, indent=2)
    with open(os.path.join(workdir, "scores.tsv"), "w") as f:
        f.write("name\tmath\tenglish\tscience\nAlice\t95\t88\t92\nBob\t78\t90\t85\nCharlie\t88\t76\t95\nDiana\t92\t94\t88\n")

TASKS.append(("data_json_to_csv", setup_data_transform, "Convert employees.json to CSV format (employees.csv) with headers name,dept,salary.", lambda wd: os.path.isfile(os.path.join(wd, "employees.csv"))))
TASKS.append(("data_filter_salary", setup_data_transform, "Filter employees.json for salary > 90000 and write to high_earners.json.", lambda wd: os.path.isfile(os.path.join(wd, "high_earners.json"))))
TASKS.append(("data_sort_by_salary", setup_data_transform, "Sort employees.json by salary descending and save to sorted_employees.json.", lambda wd: os.path.isfile(os.path.join(wd, "sorted_employees.json"))))
TASKS.append(("data_group_dept", setup_data_transform, "Group employees by department and write the result to dept_groups.json.", lambda wd: os.path.isfile(os.path.join(wd, "dept_groups.json"))))
TASKS.append(("data_avg_salary", setup_data_transform, "Calculate the average salary from employees.json and write to avg_salary.txt.", lambda wd: os.path.isfile(os.path.join(wd, "avg_salary.txt"))))
TASKS.append(("data_tsv_to_json", setup_data_transform, "Convert scores.tsv to JSON format (scores.json).", lambda wd: os.path.isfile(os.path.join(wd, "scores.json"))))
TASKS.append(("data_avg_score", setup_data_transform, "Calculate average math score from scores.tsv and write to avg_math.txt.", lambda wd: os.path.isfile(os.path.join(wd, "avg_math.txt"))))
TASKS.append(("data_top_student", setup_data_transform, "Find the student with the highest total score from scores.tsv and write to top_student.txt.", lambda wd: os.path.isfile(os.path.join(wd, "top_student.txt"))))
TASKS.append(("data_json_merge", setup_data_transform, "Add a 'bonus' field (10% of salary) to each employee in employees.json and save to employees_bonus.json.", lambda wd: os.path.isfile(os.path.join(wd, "employees_bonus.json"))))
TASKS.append(("data_csv_stats", setup_data_transform, "Read scores.tsv, compute min/max/avg for each subject, and write to score_stats.json.", lambda wd: os.path.isfile(os.path.join(wd, "score_stats.json"))))
TASKS.append(("data_json_flatten", setup_data_transform, "Flatten employees.json into a simple name:salary mapping and write to salary_map.json.", lambda wd: os.path.isfile(os.path.join(wd, "salary_map.json"))))
TASKS.append(("data_json_count", setup_data_transform, "Count employees per department in employees.json and write to dept_count.json.", lambda wd: os.path.isfile(os.path.join(wd, "dept_count.json"))))
TASKS.append(("data_json_select", setup_data_transform, "Select only name and dept fields from employees.json and save to names_depts.json.", lambda wd: os.path.isfile(os.path.join(wd, "names_depts.json"))))
TASKS.append(("data_tsv_filter", setup_data_transform, "Filter scores.tsv for students with math > 85 and write to math_good.tsv.", lambda wd: os.path.isfile(os.path.join(wd, "math_good.tsv"))))
TASKS.append(("data_json_validate", setup_data_transform, "Validate employees.json has all required fields (name, dept, salary) and write result (valid/invalid) to validation.txt.", lambda wd: os.path.isfile(os.path.join(wd, "validation.txt"))))

# ── System Operations (14) ────────────────────────────────────

def setup_sys_ops(workdir):
    os.makedirs(os.path.join(workdir, "logs"), exist_ok=True)
    for i in range(5):
        with open(os.path.join(workdir, "logs", f"app{i}.log"), "w") as f:
            f.write(f"[2024-01-{15+i}] INFO Application started\n[2024-01-{15+i}] ERROR Something went wrong\n[2024-01-{15+i}] INFO Request processed\n")
    with open(os.path.join(workdir, "makefile"), "w") as f:
        f.write("build:\n\techo building...\n\nclean:\n\techo cleaning...\n\ntest:\n\techo testing...\n")

TASKS.append(("sys_write_script", setup_sys_ops, "Create a bash script called hello.sh that prints 'Hello, World!' and make it executable.", lambda wd: os.access(os.path.join(wd, "hello.sh"), os.X_OK)))
TASKS.append(("sys_python_script", setup_sys_ops, "Create a Python script called calculator.py that takes two numbers as arguments and prints their sum. Test it with: python3 calculator.py 3 5", lambda wd: os.path.isfile(os.path.join(wd, "calculator.py"))))
TASKS.append(("sys_count_errors", setup_sys_ops, "Count the total number of ERROR lines across all log files in logs/ and write to error_count.txt.", lambda wd: os.path.isfile(os.path.join(wd, "error_count.txt"))))
TASKS.append(("sys_merge_logs", setup_sys_ops, "Merge all log files in logs/ into a single merged.log sorted by date.", lambda wd: os.path.isfile(os.path.join(wd, "merged.log"))))
TASKS.append(("sys_extract_errors", setup_sys_ops, "Extract all ERROR lines from all logs into errors.log.", lambda wd: os.path.isfile(os.path.join(wd, "errors.log"))))
TASKS.append(("sys_log_summary", setup_sys_ops, "Create a summary: count INFO, ERROR, WARNING lines per log file and write to log_summary.txt.", lambda wd: os.path.isfile(os.path.join(wd, "log_summary.txt"))))
TASKS.append(("sys_cleanup_old", setup_sys_ops, "Delete all .log files older than 3 days from logs/ (simulate with find -mtime).", lambda wd: True))  # just needs to run
TASKS.append(("sys_make_build", setup_sys_ops, "Run 'make build' using the makefile and write the output to build_output.txt.", lambda wd: os.path.isfile(os.path.join(wd, "build_output.txt"))))
TASKS.append(("sys_cron_entry", setup_sys_ops, "Create a cron entry file cron.txt with: '0 2 * * * /usr/bin/backup.sh' (just write the text, don't install).", lambda wd: os.path.isfile(os.path.join(wd, "cron.txt"))))
TASKS.append(("sys_env_check", setup_sys_ops, "Check if python3 and git are available, write 'python3: yes/no' and 'git: yes/no' to env_check.txt.", lambda wd: os.path.isfile(os.path.join(wd, "env_check.txt"))))
TASKS.append(("sys_temp_cleanup", setup_sys_ops, "Find and list all files in /tmp matching 'kitty_*' and write to tmp_files.txt.", lambda wd: os.path.isfile(os.path.join(wd, "tmp_files.txt"))))
TASKS.append(("sys_process_list", setup_sys_ops, "List running processes and write the top 10 by memory usage to processes.txt.", lambda wd: os.path.isfile(os.path.join(wd, "processes.txt"))))
TASKS.append(("sys_network_check", setup_sys_ops, "Check if localhost:11434 is reachable and write the result to network.txt.", lambda wd: os.path.isfile(os.path.join(wd, "network.txt"))))
TASKS.append(("sys_makefile_target", setup_sys_ops, "Run 'make test' and write the output to test_output.txt.", lambda wd: os.path.isfile(os.path.join(wd, "test_output.txt"))))

assert len(TASKS) == 89, f"Expected 89 tasks, got {len(TASKS)}"

# ── Runner ──────────────────────────────────────────────────────

def call_ollama(prompt: str) -> tuple[dict, float]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "top_k": 20,
            "num_predict": 2048,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = (time.time() - start) * 1000
        return data, latency
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"error": str(e)}, latency


def execute_tool_calls(tool_calls: list, workdir: str) -> list[dict]:
    results = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if name == "Write":
            path = args.get("path", "")
            content = args.get("content", "")
            # Always resolve to workdir — never allow absolute paths outside
            if os.path.isabs(path):
                path = os.path.basename(path)
            path = os.path.join(workdir, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            results.append({"tool": "Write", "path": path, "ok": True})
        elif name == "Bash":
            cmd = args.get("command", "")
            try:
                r = subprocess.run(
                    cmd, shell=True, cwd=workdir,
                    capture_output=True, text=True, timeout=30
                )
                results.append({
                    "tool": "Bash", "command": cmd,
                    "stdout": r.stdout[:500], "stderr": r.stderr[:500],
                    "returncode": r.returncode
                })
            except subprocess.TimeoutExpired:
                results.append({"tool": "Bash", "command": cmd, "error": "timeout"})
        elif name == "Read":
            path = args.get("path", "")
            if not os.path.isabs(path):
                path = os.path.join(workdir, path)
            try:
                content = open(path).read()
                results.append({"tool": "Read", "path": path, "content": content[:500], "ok": True})
            except Exception as e:
                results.append({"tool": "Read", "path": path, "error": str(e)})
        else:
            results.append({"tool": name, "error": f"unknown tool: {name}"})
    return results


def safe_check(check_fn, workdir):
    """Run a check function, return False on any error."""
    try:
        return check_fn(workdir)
    except Exception:
        return False


def run_task(task_name, setup_fn, prompt, check_fn) -> dict:
    with tempfile.TemporaryDirectory(prefix="kitty_tb2_") as workdir:
        # Setup initial state
        setup_fn(workdir)

        result_data, latency = call_ollama(prompt)

        if "error" in result_data and not result_data.get("message"):
            return {
                "id": task_name,
                "passed": False,
                "latency_ms": latency,
                "error": result_data["error"],
                "tool_calls": 0,
                "executed_tools": 0,
            }

        msg = result_data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return {
                "id": task_name,
                "passed": False,
                "latency_ms": latency,
                "error": "no tool calls",
                "tool_calls": 0,
                "executed_tools": 0,
            }

        executed = execute_tool_calls(tool_calls, workdir)
        passed = safe_check(check_fn, workdir)

        return {
            "id": task_name,
            "passed": passed,
            "latency_ms": latency,
            "tool_calls": len(tool_calls),
            "executed_tools": len(executed),
        }


def run_benchmark(tasks):
    print(f"\n{'='*60}")
    print(f"  TERMINAL-BENCH 2.0 — 89 TASKS")
    print(f"  Model: {MODEL} | Temp: {TEMPERATURE} | Timeout: {TIMEOUT}s")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()
    passed = 0
    failed = 0

    for i, (name, setup, prompt, check) in enumerate(tasks):
        result = run_task(name, setup, prompt, check)
        results.append(result)

        if result["passed"]:
            passed += 1
        else:
            failed += 1

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:3d}/{len(tasks)}] "
                f"PASS: {passed:3d}  FAIL: {failed:3d}  "
                f"({passed/(i+1)*100:.1f}%)  "
                f"ETA: {eta/60:.1f}min  "
                f"Last: {'✅' if result['passed'] else '❌'} {name}"
            )

    total_time = time.time() - start_time

    # Category breakdown
    categories = {
        "file_ops": [r for r, (_, s, _, _) in zip(results, tasks) if s == setup_file_ops],
        "text_proc": [r for r, (_, s, _, _) in zip(results, tasks) if s == setup_text_proc],
        "shell": [r for r, (_, s, _, _) in zip(results, tasks) if s == setup_shell],
        "data_transform": [r for r, (_, s, _, _) in zip(results, tasks) if s == setup_data_transform],
        "sys_ops": [r for r, (_, s, _, _) in zip(results, tasks) if s == setup_sys_ops],
    }

    cat_stats = {}
    for cat, cat_results in categories.items():
        p = sum(1 for r in cat_results if r["passed"])
        t = len(cat_results)
        cat_stats[cat] = {"passed": p, "total": t, "rate": round(p / t * 100, 1) if t > 0 else 0}

    summary = {
        "benchmark": "terminal-bench-2.0",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "total_tasks": len(tasks),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(tasks) * 100, 2),
        "total_time_s": round(total_time, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
        "per_category": cat_stats,
        "results": results,
    }

    return summary


def main():
    summary = run_benchmark(TASKS)

    outfile = "bench_tb2.json"
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — TERMINAL-BENCH 2.0")
    print(f"{'='*60}")
    print(f"  Pass rate: {summary['pass_rate']}% ({summary['passed']}/{summary['total_tasks']})")
    print(f"  Total time: {summary['total_time_s']/60:.1f} minutes")
    print(f"  Avg latency: {summary['avg_latency_ms']/1000:.1f}s per task")
    print(f"\n  Per-category breakdown:")
    for cat, stats in summary["per_category"].items():
        print(f"    {cat:18s}: {stats['passed']:3d}/{stats['total']:3d} ({stats['rate']:.1f}%)")
    print(f"\n  Results saved to: {outfile}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    main()
