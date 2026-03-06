import re
from pathlib import Path

tex_file = "paper/main.tex"
with open(tex_file, "r") as f:
    content = f.read()

models = [
    ("Qwen3-0.6B", "Qwen_Qwen3-0.6B"),
    ("Qwen2.5-0.5B-Instruct", "Qwen_Qwen2.5-0.5B-Instruct"),
    ("TinyLlama-1.1B", "TinyLlama_TinyLlama-1.1B-Chat-v1.0"),
    ("Llama-3.2-1B-Instruct", "unsloth_Llama-3.2-1B-Instruct"),
    ("Gemma-2-2B-Instruct", "unsloth_gemma-2-2b-it")
]

new_appendix = r"""\section{Per-Model Detailed Results}\label{app:permodel}

This appendix presents the full per-model experimental results, including layer-wise encoding profiles, orthogonality matrices, causal localization maps, and steering trade-off curves for each individual model.

"""

for display_name, file_key in models:
    new_appendix += f"\\subsection{{{display_name}}}\n\n"
    
    layer_fig = Path(f"paper/figures/layer_profile_{file_key}_openness.png")
    if layer_fig.exists():
        new_appendix += f"""\\begin{{figure}}[ht!]\n\\centering\n\\includegraphics[width=\\textwidth]{{figures/layer_profile_{file_key}_openness.png}}\n\\caption{{\\textbf{{{display_name}: Layer-wise personality encoding profile for Openness.}}\nLinear probe accuracy, Mean difference norm, and PCA explained variance.}}\n\\label{{fig:{file_key}_layer}}\n\\end{{figure}}\n\n"""
        
    causal_fig = Path(f"paper/figures/causal_loc_{file_key}_openness.png")
    if causal_fig.exists():
        new_appendix += f"""\\begin{{figure}}[ht!]\n\\centering\n\\includegraphics[width=\\textwidth]{{figures/causal_loc_{file_key}_openness.png}}\n\\caption{{\\textbf{{{display_name}: Token-localized and component causal importance for Openness.}}}}\n\\label{{fig:{file_key}_causal}}\n\\end{{figure}}\n\n"""
        
    ortho_fig = Path(f"paper/figures/ortho_{file_key}.png")
    if ortho_fig.exists():
        new_appendix += f"""\\begin{{figure}}[ht!]\n\\centering\n\\includegraphics[width=0.6\\textwidth]{{figures/ortho_{file_key}.png}}\n\\caption{{\\textbf{{{display_name}: Cosine similarity matrix between persona vectors.}}}}\n\\label{{fig:{file_key}_ortho}}\n\\end{{figure}}\n\n"""

# Inject back
pattern = r"\\section\{Per-Model Detailed Results\}.*?(?=\\subsection\{Extended Cross-Model Analytics\}|\\section\{Algorithm Pseudocode\}|\\end\{document\})"
content = re.sub(pattern, new_appendix.replace('\\', '\\\\'), content, flags=re.DOTALL)

with open(tex_file, "w") as f:
    f.write(content)

print("Updated Appendix B successfully.")
