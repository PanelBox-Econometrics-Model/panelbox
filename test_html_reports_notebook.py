"""
Script para testar execução do notebook de HTML reports.
Executa célula por célula e identifica erros.
"""

import json
import sys
import traceback
from pathlib import Path

# Adicionar panelbox ao path
sys.path.insert(0, "/home/guhaase/projetos/panelbox")


def execute_notebook_cells(notebook_path):
    """Executa células do notebook e identifica erros."""

    # Ler notebook
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    print(f"📓 Testando notebook: {notebook_path}")
    print(f"📊 Total de células: {len(nb['cells'])}\n")

    # Namespace global para execução
    global_ns = {}

    errors = []
    successes = 0

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        # Pular células vazias
        if not source.strip():
            continue

        # Pular células que são apenas comentários
        if all(line.strip().startswith("#") or not line.strip() for line in source.split("\n")):
            continue

        print(f"Célula {i}: ", end="", flush=True)

        try:
            # Executar célula
            exec(source, global_ns)
            print("✓")
            successes += 1
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"❌ {error_type}: {error_msg}")

            # Obter traceback completo
            tb = traceback.format_exc()

            errors.append(
                {
                    "cell": i,
                    "type": error_type,
                    "message": error_msg,
                    "traceback": tb,
                    "source": source[:300] + "..." if len(source) > 300 else source,
                }
            )

            # Parar em alguns erros críticos
            if error_type in ["SyntaxError", "IndentationError"]:
                print(f"\n⚠️  Erro crítico de sintaxe na célula {i}. Parando.")
                break

    print(f"\n{'='*70}")
    print(f"📊 RESUMO:")
    print(f"  ✓ Sucessos: {successes}")
    print(f"  ❌ Erros: {len(errors)}")
    print(f"{'='*70}\n")

    if errors:
        print("📋 ERROS ENCONTRADOS:\n")
        for err in errors:
            print(f"Célula {err['cell']}: {err['type']}")
            print(f"  Mensagem: {err['message']}")
            print(f"  Código:\n{err['source']}")
            print(f"\n  Traceback:\n{err['traceback']}")
            print("-" * 70)
            print()

    return errors


if __name__ == "__main__":
    notebook_path = "examples/jupyter/08_html_reports_complete_guide.ipynb"

    print("=" * 70)
    print("TESTE DE EXECUÇÃO DO NOTEBOOK - HTML REPORTS")
    print("=" * 70)
    print()

    errors = execute_notebook_cells(notebook_path)

    if not errors:
        print("✅ NOTEBOOK SEM ERROS!")
    else:
        print(f"⚠️  {len(errors)} ERRO(S) ENCONTRADO(S)")
        print("\nVer detalhes acima para correção.")

    sys.exit(len(errors))
