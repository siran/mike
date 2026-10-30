"""Copy a GGUF's chat template, letting it think: the empty <think></think> only when enable_thinking is false."""
import sys
from gguf import GGUFReader

gguf, out = sys.argv[1], sys.argv[2]
field = GGUFReader(gguf).fields["tokenizer.chat_template"]
template = bytes(field.parts[field.data[0]]).decode()
forced = "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
assert forced in template, "no forced empty think block in this template"
optional = ("{{- '<|im_start|>assistant\\n' }} {%- if enable_thinking is defined and enable_thinking is false %} "
            "{{- '<think>\\n\\n</think>\\n\\n' }} {%- endif %}")
open(out, "w").write(template.replace(forced, optional))
print("wrote", out)
