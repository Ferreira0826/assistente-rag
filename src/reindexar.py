"""
REINDEXAÇÃO COMPLETA DA BASE.

Rode este script sempre que mudar:
  - a função de embedding (PROVEDOR_EMBEDDING no .env), ou
  - a estratégia de chunking (carregador.py)

POR QUE APAGAR A COLEÇÃO ANTES:

1. Embeddings de modelos diferentes vivem em espaços vetoriais
   diferentes. Misturar vetores do MiniLM com vetores do
   text-embedding-3-small produz distâncias sem significado — a busca
   não quebra, ela simplesmente devolve lixo. É o pior tipo de bug:
   silencioso.

2. `colecao.add()` com IDs repetidos não substitui o conteúdo antigo.
   O `indexar_pasta()` original rodava a cada execução do app e ia
   empilhando os mesmos IDs. Recriar a coleção resolve isso de vez.

USO (a partir da raiz do projeto):
    python src/reindexar.py
"""

import shutil
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

# load_dotenv() ANTES dos imports do projeto, de propósito: se viesse
# depois, qualquer variável lida em tempo de import (nível de módulo)
# enxergaria o ambiente ainda sem o .env carregado.
load_dotenv()

from base_vetorial import criar_base, indexar_chunks, provedor_atual  # noqa: E402
from carregador import (  # noqa: E402
    OVERLAP_PADRAO,
    TAMANHO_PADRAO,
    ler_pdf,
    ler_txt,
    quebrar_em_chunks,
)

# Duas pastas, com propósitos diferentes:
#   documentos/          -> ignorado pelo git. Arquivos locais e privados.
#   documentos_exemplo/  -> versionado. Documentos públicos, para que quem
#                           clonar o repositório consiga rodar o projeto
#                           sem precisar arrumar material próprio.
# Ambas são indexadas na mesma coleção. Pasta ausente é simplesmente pulada.
PASTAS_DOCUMENTOS = ["documentos", "documentos_exemplo"]

NOME_COLECAO = "documentos"
CAMINHO_CHROMA = "./chroma_db"


def apagar_colecao(nome: str):
    """Remove a coleção antiga, se existir."""
    cliente = chromadb.PersistentClient(path=CAMINHO_CHROMA)
    existentes = [getattr(c, "name", c) for c in cliente.list_collections()]

    if nome in existentes:
        cliente.delete_collection(nome)
        print(f"Coleção '{nome}' apagada.")
    else:
        print(f"Coleção '{nome}' não existia — nada a apagar.")


PASTA_CACHE = Path("cache_texto")


def carregar_arquivo(caminho: Path) -> str:
    """
    Lê PDF (com OCR automático se preciso) ou TXT, com cache em disco.

    POR QUE O CACHE:
    OCR é caro — um PDF escaneado de 32 páginas leva minutos. Sem cache,
    toda reindexação refaz esse trabalho, mesmo quando o que mudou foi só
    o chunking ou o modelo de embedding (que é o caso comum enquanto se
    ajusta o RAG).

    A chave do cache inclui o tamanho e a data de modificação do arquivo:
    se o PDF for trocado, o cache é invalidado sozinho.

    A pasta cache_texto/ contém o texto dos documentos — inclua no
    .gitignore junto com documentos/ e chroma_db/.
    """
    if caminho.suffix.lower() in (".txt", ".md"):
        return ler_txt(str(caminho))

    if caminho.suffix.lower() != ".pdf":
        return ""

    PASTA_CACHE.mkdir(exist_ok=True)
    stat = caminho.stat()
    arquivo_cache = PASTA_CACHE / f"{caminho.stem}_{stat.st_size}_{int(stat.st_mtime)}.txt"

    if arquivo_cache.exists():
        print("  (texto recuperado do cache — OCR não será refeito)")
        return arquivo_cache.read_text(encoding="utf-8")

    texto = ler_pdf(str(caminho))
    if texto.strip():
        arquivo_cache.write_text(texto, encoding="utf-8")
    return texto


def main():
    arquivos = []
    for nome_pasta in PASTAS_DOCUMENTOS:
        pasta = Path(nome_pasta)
        if not pasta.exists():
            continue
        arquivos.extend(
            a for a in sorted(pasta.iterdir())
            if a.is_file() and a.suffix.lower() in (".pdf", ".txt", ".md")
        )

    if not arquivos:
        print(
            "ERRO: nenhum PDF/TXT encontrado em "
            f"{' nem em '.join(repr(p) for p in PASTAS_DOCUMENTOS)}."
        )
        sys.exit(1)

    print("=" * 60)
    print("REINDEXAÇÃO DA BASE")
    print("=" * 60)
    print(f"Embedding : {provedor_atual()}")
    print(f"Chunk     : {TAMANHO_PADRAO} caracteres, overlap {OVERLAP_PADRAO}")
    print(f"Arquivos  : {len(arquivos)}")
    for a in arquivos:
        print(f"            {a.parent.name}/{a.name}")
    print()

    # Backup do banco atual — barato, e evita choro se algo der errado
    origem_bkp = Path(CAMINHO_CHROMA)
    if origem_bkp.exists():
        destino_bkp = Path(f"{CAMINHO_CHROMA}_backup")
        if destino_bkp.exists():
            shutil.rmtree(destino_bkp)
        shutil.copytree(origem_bkp, destino_bkp)
        print(f"Backup salvo em '{destino_bkp}'.\n")

    apagar_colecao(NOME_COLECAO)

    # A coleção nova já nasce com a função de embedding correta
    colecao = criar_base(NOME_COLECAO)
    print()

    total_chunks = 0
    for arquivo in arquivos:
        print(f"Processando '{arquivo.name}'...")

        texto = carregar_arquivo(arquivo)
        if not texto.strip():
            print("  AVISO: nenhum texto extraído. Pulando.")
            continue

        chunks = quebrar_em_chunks(texto)
        if not chunks:
            print("  AVISO: nenhum chunk gerado. Pulando.")
            continue

        indexar_chunks(colecao, chunks, origem=arquivo.name)
        total_chunks += len(chunks)

    print()
    print("=" * 60)
    print(f"Concluído: {total_chunks} chunks em '{NOME_COLECAO}'.")
    print(f"Total na coleção: {colecao.count()}")
    print("=" * 60)

    # Teste de fumaça: a base sabe responder sobre SISAB agora?
    from base_vetorial import buscar_com_fontes

    print("\nTeste rápido — busca por 'SISAB':")
    for item in buscar_com_fontes(colecao, "SISAB", k=3):
        texto = item["texto"].replace("\n", " ")[:90]
        print(f"  dist={item['distancia']:.3f} | {item['origem']}")
        print(f"    {texto}...")


if __name__ == "__main__":
    main()