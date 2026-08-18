"""
Peças 2 e 3 do RAG: embeddings + banco vetorial.

--- ALTERAÇÕES DA PARADA TÉCNICA ---
`criar_base()` agora aceita uma FUNÇÃO DE EMBEDDING configurável.

Motivo: o ChromaDB usa por padrão o `all-MiniLM-L6-v2`, treinado
majoritariamente em inglês. Com documentos técnicos em português cheios
de sigla (SISAB, e-SUS APS, CBO, SAPS, LEDI), a representação semântica
fica ruim e a busca não encontra conteúdo que ESTÁ no documento — foi
exatamente o que aconteceu na validação do agente.

Também foi adicionada `buscar_com_fontes()`, usada pela tool do agente.
"""

import os

import chromadb
from chromadb.utils import embedding_functions


def _sanear_variaveis_vazias():
    """
    Remove do ambiente variáveis de configuração que estão VAZIAS.

    POR QUE ISSO EXISTE (armadilha que custou caro):
    Uma linha `OPENAI_BASE_URL=` no .env coloca uma string vazia no
    ambiente. O SDK da OpenAI testa essa variável com `is None`, não com
    teste de verdade/falsidade — então a string vazia passa no teste e
    ele tenta usá-la como endereço da API. O httpx então falha com
    "Request URL is missing an 'http://' or 'https://' protocol", um erro
    que não diz absolutamente nada sobre .env.

    No agente dá pra contornar passando base_url explícito. Aqui não:
    quem instancia o cliente é a OpenAIEmbeddingFunction do Chroma, por
    dentro. A única saída é limpar o ambiente antes.

    Chamada em tempo de execução (não de import) para rodar depois de
    qualquer load_dotenv().
    """
    for variavel in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        valor = os.environ.get(variavel)
        if valor is not None and not valor.strip():
            del os.environ[variavel]

# Qual embedding usar. Definido no .env:
#   openai (padrão) — text-embedding-3-small, via API. Multilíngue e bom
#                     em português. Custa ~US$ 0,02 por milhão de tokens:
#                     indexar uma base inteira sai por centavos.
#   local           — modelo multilíngue rodando na máquina, sem API e
#                     sem custo. Baixa ~120 MB na primeira execução.
#   padrao          — o all-MiniLM-L6-v2 original do Chroma (inglês).
#                     Mantido só para comparação.
def provedor_atual() -> str:
    """Lê o provedor NA HORA DO USO. Se fosse constante de módulo, seria
    avaliada no import — antes do load_dotenv() de quem importa — e o
    valor do .env seria ignorado."""
    return (os.getenv("PROVEDOR_EMBEDDING") or "openai").strip().lower()

MODELO_LOCAL = "paraphrase-multilingual-MiniLM-L12-v2"


def obter_funcao_embedding(provedor: str | None = None):
    """
    Devolve a função de embedding conforme o provedor escolhido.

    IMPORTANTE: embeddings de modelos diferentes NÃO são compatíveis
    entre si. Trocar de provedor exige reindexar do zero — por isso
    existe o script `reindexar.py`.
    """
    _sanear_variaveis_vazias()
    provedor = (provedor or provedor_atual()).strip().lower()

    if provedor == "openai":
        chave = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not chave:
            raise RuntimeError(
                "PROVEDOR_EMBEDDING=openai exige OPENAI_API_KEY no .env."
            )
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=chave,
            model_name="text-embedding-3-small",
        )

    if provedor == "local":
        # Baixa o modelo na primeira execução (precisa de acesso ao
        # huggingface.co). Depois roda offline.
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODELO_LOCAL
        )

    # "padrao" — deixa o Chroma usar o modelo dele.
    return None


def criar_base(nome: str = "documentos", provedor: str | None = None):
    """
    Cria (ou abre) uma coleção no ChromaDB.

    PersistentClient salva em ./chroma_db, então os dados sobrevivem ao
    fechamento do programa.

    ATENÇÃO: a função de embedding usada aqui precisa ser a MESMA da
    indexação. Se não for, a busca retorna resultado sem sentido.
    """
    cliente = chromadb.PersistentClient(path="./chroma_db")
    funcao = obter_funcao_embedding(provedor)

    if funcao is None:
        return cliente.get_or_create_collection(name=nome)

    return cliente.get_or_create_collection(name=nome, embedding_function=funcao)


def indexar_chunks(colecao, chunks: list[str], origem: str):
    """
    Joga os chunks no banco. O Chroma gera o embedding de cada um.

    Cada item precisa de um ID único: nome da origem + índice.
    'metadatas' guarda de qual documento veio o chunk, o que permite
    citar a fonte na resposta.
    """
    ids = [f"{origem}_{i}" for i in range(len(chunks))]
    metadados = [{"origem": origem} for _ in chunks]

    colecao.add(documents=chunks, ids=ids, metadatas=metadados)
    print(f"  Indexados {len(chunks)} chunks de '{origem}'.")


def buscar(colecao, pergunta: str, k: int = 3) -> list[str]:
    """
    Busca os `k` chunks mais parecidos com a pergunta. Retorna só os textos.
    Mantida como está para não quebrar `rag.py` (modo RAG puro).
    """
    resultado = colecao.query(query_texts=[pergunta], n_results=k)
    return resultado["documents"][0]


def buscar_com_fontes(colecao, pergunta: str, k: int = 5) -> list[dict]:
    """
    Mesma busca, mas devolvendo também a ORIGEM do trecho.

    O `buscar()` original descartava os metadados, então o RAG não tinha
    como citar a fonte. O agente precisa dessa informação: resposta com
    fonte citada é o que separa um assistente confiável de um que parece
    inventar.

    Retorna lista de dicts com 'texto', 'origem' e 'distancia'
    (distância menor = mais parecido com a pergunta).
    """
    resultado = colecao.query(
        query_texts=[pergunta],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "texto": texto,
            "origem": (meta or {}).get("origem", "desconhecida"),
            "distancia": distancia,
        }
        for texto, meta, distancia in zip(
            resultado["documents"][0],
            resultado["metadatas"][0],
            resultado["distances"][0],
        )
    ]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print(f"Provedor de embedding: {provedor_atual()}\n")

    colecao = criar_base("teste_embedding")

    exemplos = [
        "O SISAB consolida as informações da Atenção Básica enviadas pelo e-SUS APS.",
        "A Nissan March 2015 usa óleo 5W30 na revisão de 10 mil quilômetros.",
        "Os dados enviados ao Centralizador Nacional passam por validação.",
        "Python é ótimo para automação de planilhas com pandas.",
    ]
    try:
        indexar_chunks(colecao, exemplos, origem="teste")
    except Exception:
        print("  (chunks já indexados, seguindo)")

    pergunta = "Qual sistema consolida os dados da atenção básica?"
    print(f"\nPergunta: {pergunta}\n")
    for item in buscar_com_fontes(colecao, pergunta, k=3):
        print(f"  dist={item['distancia']:.3f} | {item['texto'][:80]}")