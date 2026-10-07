# Tampered Template Examples

Each file here is a **chat template attack specimen** — safe to read, never to install.
These are the patterns a hostile GGUF's `tokenizer.chat_template` uses. The `benign/`
folder has the honest equivalents for contrast. Load none of these into a model.

| Specimen | Attack class | What it would do |
|---|---|---|
| `01-ssti-class-escape.j2` | Sandbox escape | Walks Python's class hierarchy from an empty string to reach `subprocess` and run shell commands when the template renders |
| `02-exfil-system-prompt.j2` | Prompt/secret theft | Silently copies the system prompt (and anything containing "key"/"token") into a tool-call the model will emit, smuggling data out of the session |
| `03-invisible-instructions.j2` | Hidden instruction injection | Zero-width characters carry a hidden instruction that's invisible in any visual review but alters rendered output |
| `04-homoglyph-smuggle.j2` | Review evasion | Cyrillic/Greek look-alikes make malicious text survive human inspection and keyword filters |
| `05-attribute-traversal.j2` | Sandbox escape (filter abuse) | Uses only Jinja's "safe" `|attr` filter plus string-building to reach forbidden objects without ever writing the obvious keywords |
| `06-benign-reference.j2` | — (in `benign/`) | A normal Qwen-style template: loops messages, renders roles, nothing else |

**How to read a specimen:** every `{%- ... -%}` is code, everything else is text the model
sees. Anything in `{{ ... }}` gets evaluated. Honesty check: a clean template only ever
touches `messages`, roles, content, and the special tokens — nothing else.
