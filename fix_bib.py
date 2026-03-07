import re

with open("paper/main.tex", "r") as f:
    text = f.read()

# Make sure main.tex cleanly references the bib items.
if "gemmateam2024gemma" not in text:
    text = text.replace("Gemma-2 \\cite{team2024gemma}", "Gemma-2 \\cite{gemmateam2024gemma}")
    text = text.replace("Gemma-2 \\cite{gemma2024gemma}", "Gemma-2 \\cite{gemmateam2024gemma}")

if "zhang2024tinyllama" not in text:
    text = text.replace("TinyLlama \\cite{zhang2024tinyllama}", "TinyLlama \\cite{zhang2024tinyllama}")

if "dubey2024llama3" not in text:
    text = text.replace("Llama-3.2 \\cite{llama32024}", "Llama-3.2 \\cite{dubey2024llama3}")

with open("paper/main.tex", "w") as f:
    f.write(text)

