"""
O PIPELINE COMPLETO do RAG, juntando as 5 peças.

Fluxo:
  INDEXAÇÃO (feita 1 vez, quando você adiciona documentos):
    documento -> chunks -> embeddings -> guarda no vector DB

  CONSULTA (feita a cada pergunta):
    pergunta -> busca chunks parecidos -> injeta no prompt -> LLM responde

Se você entender este arquivo, você entende RAG. É só isto.
"""

from pathlib import Path

from carregador import ler_pdf, ler_txt, quebrar_em_chunks
from base_vetorial import criar_base, indexar_chunks, buscar
from gerador import responder


def indexar_pasta(pasta: str = "documentos", nome_base: str = "documentos"):
    """
    Lê todos os PDFs e TXTs de uma pasta, quebra em chunks e indexa.
    Roda uma vez (ou sempre que adicionar documentos novos).
    """
    colecao = criar_base(nome_base)

    for arquivo in Path(pasta).iterdir():
        if arquivo.suffix.lower() == ".pdf":
            texto = ler_pdf(str(arquivo))
        elif arquivo.suffix.lower() == ".txt":
            texto = ler_txt(str(arquivo))
        else:
            continue  # ignora outros tipos

        chunks = quebrar_em_chunks(texto, tamanho=500, overlap=50)
        indexar_chunks(colecao, chunks, origem=arquivo.name)

    return colecao


def perguntar(colecao, pergunta: str, k: int = 6) -> str:
    """
    O ciclo de consulta completo: busca + geração.
    """
    # 1. RETRIEVAL: acha os chunks relevantes
    chunks_relevantes = buscar(colecao, pergunta, k=k)
   
    # 2. GERAÇÃO: manda pro LLM com o contexto
    resposta = responder(pergunta, chunks_relevantes)

    return resposta


if __name__ == "__main__":
    print("Indexando documentos da pasta 'documentos/'...")
    colecao = indexar_pasta()

    print("\nPronto! Faça perguntas (digite 'sair' para encerrar).\n")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        resposta = perguntar(colecao, pergunta)
        print(f"\nAssistente: {resposta}\n")
