# Project Encoding Policy

All agents and contributors working in this repository must use UTF-8 explicitly.

- Files: create and edit text files as UTF-8.
- Python code: pass `encoding="utf-8"` for text `open()`, `Path.read_text()`, `Path.write_text()`, subprocess text I/O, and decoding/encoding boundaries.
- Python runtime: run with `PYTHONUTF8=1`; keep `PYTHONIOENCODING=utf-8` for subprocesses and CLI tests.
- PowerShell: run commands in UTF-8 mode before project work:
  - `$OutputEncoding = [System.Text.UTF8Encoding]::new($false)`
  - `[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)`
  - `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)`
  - `$env:PYTHONUTF8 = "1"`
  - `$env:PYTHONIOENCODING = "utf-8"`
- Git: keep repository text as UTF-8; use `.gitattributes` and local Git UTF-8 display settings.
- Do not rely on system-default encodings such as Windows ANSI/GBK or locale-dependent Python defaults.
- If a tool cannot specify encoding, wrap it with a script or environment that forces UTF-8 before using it.
