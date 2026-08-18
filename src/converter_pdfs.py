"""
Utilitário: converte PDFs em .txt

Pega todos os PDFs da pasta 'documentos/', extrai o texto
(reaproveitando o carregador do projeto, que já faz OCR quando
o PDF é escaneado) e salva cada um como .txt na pasta 'txt_convertidos/'.

Por que usar:
- PDF escaneado só precisa passar por OCR UMA vez. Depois você usa
  o .txt (leitura instantânea) em vez de refazer o OCR toda hora.
- Você pode abrir o .txt e corrigir na mão erros que o OCR cometeu.

Como rodar (a partir da RAIZ do projeto):
    python src/converter_pdfs.py
"""

import sys
from pathlib import Path

# permite importar o carregador que está na mesma pasta src/
sys.path.append(str(Path(__file__).parent))

from carregador import ler_pdf


def converter_todos(pasta_entrada: str = "documentos", pasta_saida: str = "txt_convertidos"):
    """
    Lê todos os PDFs de 'pasta_entrada' e salva o texto extraído
    como .txt em 'pasta_saida'.
    """
    entrada = Path(pasta_entrada)
    saida = Path(pasta_saida)
    saida.mkdir(exist_ok=True)  # cria a pasta de saída se não existir

    pdfs = list(entrada.glob("*.pdf"))

    if not pdfs:
        print(f"Nenhum PDF encontrado em '{pasta_entrada}/'.")
        return

    print(f"Encontrados {len(pdfs)} PDF(s). Convertendo...\n")

    for pdf in pdfs:
        print(f"Processando: {pdf.name}")
        try:
            texto = ler_pdf(str(pdf))

            if not texto.strip():
                print(f"  [AVISO] Nenhum texto extraído de '{pdf.name}'. Pulando.\n")
                continue

            # o .txt tem o mesmo nome do PDF, só muda a extensão
            destino = saida / (pdf.stem + ".txt")
            destino.write_text(texto, encoding="utf-8")
            print(f"  -> salvo em: {destino}  ({len(texto)} caracteres)\n")

        except Exception as e:
            print(f"  [ERRO] Falha ao converter '{pdf.name}': {e}\n")

    print("Concluído! Os .txt estão na pasta 'txt_convertidos/'.")
    print("Se quiser usá-los no RAG, mova-os para a pasta 'documentos/'")
    print("(e, se quiser, tire os PDFs escaneados de lá para não refazer OCR).")


if __name__ == "__main__":
    converter_todos()