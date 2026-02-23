# Skill: Computer Terminal Interface

This tutorial covers how to interact with the host system's terminal using MCP Computer tools. This interface provides direct shell access, allowing you to execute commands and read their output.

---

## 1. Toolset Overview

### A. `send_keys` (Simulated Input)
Simulates keystrokes sent to the active terminal session.

*   **Signature**: `send_keys(keys: List[str])`

> [!IMPORTANT]
> **The Newline Requirement**: The terminal will **NOT** execute a command until it receives a `\n` or `Enter`. You must explicitly include `"Enter"` in the keys list.
>
> *   **Incorrect**: `send_keys(keys=["ls -la"])` (Text remains unexecuted on the prompt)
> *   **Correct**: `send_keys(keys=["ls -la", "Enter"])` (Command executes immediately)

#### Control Characters
| Key | Action | Use Case |
| :--- | :--- | :--- |
| `"C-c"` | `SIGINT` | Interrupt/Stop a hanging process or clear the command line. |
| `"C-z"` | `SIGTSTP` | Suspend/Background the current process. |
| `"C-d"` | `EOF` | Signal End-of-File to exit shells or close input streams. |

### B. `read_command_output` (Observation)
Captures the output buffer of the terminal.

*   **Signature**: `read_command_output(wait: int = 1)`

> [!TIP]
> Use a longer `wait` time for processes that perform network operations, heavy I/O, or compilation to ensure the output is captured completely.

---

## 2. Operational Strategies

### Pre-Flight: The Prompt Check
Before sending a new command, verify the terminal state.

1.  **Check Buffer**: Call `read_command_output(wait=0)`.
2.  **Verify Prompt**: Look for the shell prompt (e.g., `$`, `#`, `>`) at the end of the result.
3.  **Act Accordingly**:
    *   **Prompt Found**: Proceed with the next command.
    *   **No Prompt**: Wait longer or send `"C-c"` if the previous command is stuck.

### Secure File Creation (Echo Pattern)
Avoid interactive editors (nano, vim) which are difficult to navigate blindly. Use heredocs for reliability.

```bash
# Safe multi-line file creation
send_keys(keys=["cat <<EOF > script.py", "Enter"])
send_keys(keys=["print('Execution successful')", "Enter"])
send_keys(keys=["EOF", "Enter"])
```

### Handling Massive Output
`read_command_output` has a hard token limit. If a command generates too much data, the tool call will **fail**.

> [!WARNING]
> If a read fails due to size limits, **do not** retry the same call. Implement the Pagination Pattern instead.

**The Pagination Pattern:**
1.  **Capture to File**: Rerun the command with redirection.
    ```bash
    command > output.tmp 2>&1
    ```
2.  **Sample Output**: Read specific sections using `head`, `tail`, or `grep`.
    ```bash
    head -n 50 output.tmp
    tail -n 50 output.tmp
    ```

### Managing Interactive Pagers
Interactive pagers (e.g., `less`, `more`, `man`) block terminal progression until input is received. Use flags or pipes to disable them.

*   **Pipes**: `command | cat`
*   **Git**: `git --no-pager <cmd>`
*   **Emergency Escape**: If trapped in a pager (output ends with `:`), send `keys=["q", "Enter"]`.

---

## 3. Safety Checklists

1.  **Idempotency**: Check if a target directory or file exists before writing.
2.  **Infinite Loops**: Always set limits (e.g., `ping -c 4` instead of `ping`).
3.  **Privilege Management**: Avoid `sudo` unless authorized; password prompts are not natively handled.
4.  **No Blind Deletion**: Always verify your current directory with `pwd` before running `rm`.
