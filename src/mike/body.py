"""MIK-e's computer: run one shell command as the body's user, with a time limit and an output cap."""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time

PROMPT = "$ "
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_]\w*)\1")
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def command_of(line: str) -> str | None:
    """The command on a reply line, or None if the line is speech."""
    stripped = line.strip()
    if stripped.startswith(PROMPT) and stripped[len(PROMPT):].strip():
        return stripped[len(PROMPT):].strip()
    return None


def _unfinished(command: str) -> bool:
    """True while a quote is open or the line ends in a backslash: bash would keep reading."""
    try:
        shlex.split(command, comments=True)
        return False
    except ValueError:
        return True


def _first_command(lines: list[str]):
    """(said, command, index of its last line, complete). Like bash, a command continues past its first line
    while a quote is open, and a `<<WORD` heredoc runs to the line WORD."""
    said = []
    for i, line in enumerate(lines):
        command = command_of(line)
        if command is None:
            said.append(line)
            continue
        marker = HEREDOC.search(command)
        if marker:
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == marker.group(2):
                    return said, "\n".join([command] + lines[i + 1:j + 1]), j, True
            return said, "\n".join([command] + lines[i + 1:]), len(lines) - 1, False
        j = i
        while _unfinished(command) and j + 1 < len(lines):
            j += 1
            command += "\n" + lines[j]
        return said, command, j, not _unfinished(command)
    return said, None, len(lines) - 1, False


def parse_reply(text: str) -> tuple[list[str], str | None, str]:
    """(lines said before the first command, that command or None, the reply kept up to the command's end)."""
    lines = text.splitlines()
    said, command, end, _ = _first_command(lines)
    return said, command, text if command is None else "\n".join(lines[:end + 1])


def has_command(generated: str) -> bool:
    """Early stop: the finished lines of the reply already hold a whole command."""
    if "\n" not in generated:
        return False
    return _first_command(generated[:generated.rfind("\n")].splitlines())[3]


class Body:
    def __init__(self, home: str | None = None, user: str | None = None, timeout: float = 30.0,
                 max_output: int = 2000, grace: float = 1.0):
        import pwd
        self.user = user
        self.uid = self.gid = None
        if user is not None:
            entry = pwd.getpwnam(user)
            self.uid, self.gid = entry.pw_uid, entry.pw_gid
            home = home or entry.pw_dir
        self.home = home or os.path.expanduser("~")
        self.timeout, self.max_output, self.grace = timeout, max_output, grace

    def environment(self) -> dict:
        import pwd
        user = self.user or pwd.getpwuid(os.getuid()).pw_name
        return {"HOME": self.home, "USER": user, "LOGNAME": user, "SHELL": "/bin/bash", "PATH": PATH,
                "LANG": "C.UTF-8", "TERM": "dumb", "PAGER": "cat", "MANPAGER": "cat", "GIT_PAGER": "cat",
                "MANWIDTH": "80", "COLUMNS": "80", "DEBIAN_FRONTEND": "noninteractive"}

    def run(self, command: str) -> dict:
        import select
        drop = {}
        if self.uid is not None and os.geteuid() == 0 and self.uid != 0:
            drop = {"user": self.uid, "group": self.gid,
                    "extra_groups": os.getgrouplist(self.user, self.gid)}
        start = time.monotonic()
        proc = subprocess.Popen(["bash", "-c", command], cwd=self.home, env=self.environment(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, **drop)
        fd, keep = proc.stdout.fileno(), self.max_output * 4
        chunks, kept, total = [], 0, 0
        deadline, exited, timed_out, eof = start + self.timeout, None, False, False
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                break
            if eof:
                try:
                    proc.wait(deadline - now)
                except subprocess.TimeoutExpired:
                    timed_out = True
                break
            if proc.poll() is not None:
                exited = exited or now
                if now - exited >= self.grace:  # a background child still holds the pipe
                    break
            ready, _, _ = select.select([fd], [], [], min(0.1, deadline - now))
            if ready:
                data = os.read(fd, 65536)
                if not data:
                    eof = True
                    continue
                total += len(data)
                if kept < keep:
                    chunks.append(data[:keep - kept])
                    kept += len(chunks[-1])
        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        proc.stdout.close()
        text = b"".join(chunks).decode("utf-8", "replace")
        truncated = len(text) > self.max_output or total > kept
        return {"command": command, "output": text[:self.max_output], "truncated": truncated,
                "output_bytes": total, "exit_code": None if timed_out else proc.returncode,
                "timed_out": timed_out, "seconds": round(time.monotonic() - start, 3)}

    def observation(self, result: dict) -> str:
        """What the kernel hands back to the model after a command."""
        text = result["output"].rstrip("\n") or "(no output)"
        if result["truncated"]:
            text += f"\n[cut: only the first {self.max_output} characters are shown]"
        if result["timed_out"]:
            text += f"\n(stopped after {self.timeout:g} seconds)"
        elif result["exit_code"]:
            text += f"\n(exit status {result['exit_code']})"
        return "output:\n" + text
