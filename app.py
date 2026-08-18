"""
Interface web do assistente, com Streamlit.

Rode a partir da RAIZ do projeto:
    streamlit run app.py

MUDANÇA EM RELAÇÃO À VERSÃO ANTERIOR
------------------------------------
Antes esta interface chamava o RAG puro: toda pergunta ia direto ao
retrieval, sem decisão nenhuma. Agora ela conversa com o AGENTE, que
escolhe entre as ferramentas disponíveis.

A diferença fica visível na tela: o painel de ferramentas mostra o que o
agente decidiu usar em cada pergunta. É o que transforma "confie em mim,
é um agente" em algo demonstrável.

Também foi removida a reindexação automática. A versão anterior chamava
`indexar_pasta()` a cada carga do app, o que reinsere os mesmos IDs no
Chroma. Indexar é operação de manutenção e agora vive em
`src/reindexar.py`, onde deve estar.
"""

import sys
from pathlib import Path

# deixa o Python enxergar a pasta src/
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

from agente import Conversa, criar_cliente
from base_vetorial import criar_base, provedor_atual

load_dotenv()

st.set_page_config(page_title="Assistente de Documentos", page_icon="📄")


# =====================================================================
# Recursos carregados uma vez por sessão
# =====================================================================

@st.cache_resource
def carregar_recursos():
    """Abre a coleção existente e cria o cliente da API."""
    return criar_base("documentos"), criar_cliente()


try:
    colecao, cliente = carregar_recursos()
except Exception as erro:
    st.error(f"Não foi possível iniciar o assistente: {erro}")
    st.info(
        "Verifique se o arquivo `.env` existe na raiz do projeto e contém "
        "`OPENAI_API_KEY`."
    )
    st.stop()

total_chunks = colecao.count()


# =====================================================================
# Barra lateral
# =====================================================================

with st.sidebar:
    st.subheader("Base de conhecimento")

    if total_chunks == 0:
        st.warning("A base está vazia.")
        st.caption("Coloque seus arquivos em `documentos/` e rode `python src/reindexar.py`.")
    else:
        dados = colecao.get(include=["metadatas"])
        origens = sorted({(m or {}).get("origem", "?") for m in dados["metadatas"]})

        st.metric("Trechos indexados", total_chunks)
        st.caption("Documentos:")
        for origem in origens:
            st.caption(f"· {origem}")

    st.divider()
    st.subheader("Ferramentas do agente")
    st.caption("· buscar_documentos")
    st.caption("· calcular")
    st.caption("· data_atual")
    st.caption("· dias_entre_datas")
    st.caption(f"Embeddings: {provedor_atual()}")

    st.divider()
    mostrar_ferramentas = st.toggle("Mostrar decisões do agente", value=True)

    if st.button("Nova conversa", use_container_width=True):
        st.session_state.conversa = Conversa(cliente, colecao, verbose=False)
        st.session_state.historico = []
        st.rerun()


# =====================================================================
# Estado da sessão
# =====================================================================
#
# A instância de Conversa guarda a memória entre perguntas. Ela precisa
# viver no session_state: o Streamlit reexecuta o script inteiro a cada
# interação, então qualquer variável comum seria recriada do zero — e a
# conversa perderia o contexto a cada mensagem.

if "conversa" not in st.session_state:
    st.session_state.conversa = Conversa(cliente, colecao, verbose=False)

if "historico" not in st.session_state:
    st.session_state.historico = []


# =====================================================================
# Tela
# =====================================================================

st.title("Assistente de Documentos")
st.caption(
    "Pergunte sobre os documentos indexados. O agente decide quando buscar "
    "no acervo, quando calcular e quando consultar datas."
)


def desenhar_ferramentas(eventos):
    """Mostra o que o agente decidiu usar."""
    if not eventos or not mostrar_ferramentas:
        return

    rotulo = f"{len(eventos)} ferramenta{'s' if len(eventos) > 1 else ''} usada"
    if len(eventos) > 1:
        rotulo += "s"

    with st.expander(rotulo, expanded=False):
        for i, evento in enumerate(eventos, 1):
            st.markdown(f"**{i}. `{evento['nome']}`**")
            if evento["argumentos"]:
                st.json(evento["argumentos"], expanded=False)
            trecho = str(evento["resultado"])
            if len(trecho) > 500:
                trecho = trecho[:500] + "…"
            st.text(trecho)


# histórico já existente
for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            desenhar_ferramentas(msg.get("ferramentas"))
        st.write(msg["conteudo"])


# Teto para o tamanho da pergunta. Sem isso, um texto colado de 500 mil
# caracteres vai inteiro para a API — uma única mensagem pode custar caro.
# Toda entrada de usuário que vira token precisa de limite.
TAMANHO_MAXIMO_PERGUNTA = 2000

if pergunta := st.chat_input("Faça uma pergunta"):
    if len(pergunta) > TAMANHO_MAXIMO_PERGUNTA:
        st.warning(
            f"A pergunta tem {len(pergunta)} caracteres. "
            f"O limite é {TAMANHO_MAXIMO_PERGUNTA}. Resuma e tente de novo."
        )
        st.stop()

    st.session_state.historico.append({"role": "user", "conteudo": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        eventos = []

        # O callback registra cada ferramenta usada durante o loop do
        # agente. No terminal bastava o print; aqui capturamos os eventos
        # para desenhar na tela.
        def registrar(nome, argumentos, resultado):
            eventos.append(
                {"nome": nome, "argumentos": argumentos, "resultado": resultado}
            )

        st.session_state.conversa.ao_usar_ferramenta = registrar

        with st.spinner("Pensando..."):
            try:
                resposta = st.session_state.conversa.perguntar(pergunta)
            except Exception as erro:
                # Mensagem genérica na tela, detalhe no console: um traceback
                # exposto na interface entrega caminhos de arquivo, versões de
                # biblioteca e estrutura interna a quem estiver do outro lado.
                print(f"[erro] {type(erro).__name__}: {erro}")
                resposta = (
                    "Não consegui processar essa pergunta. "
                    "Tente reformular ou comece uma nova conversa."
                )

        desenhar_ferramentas(eventos)
        st.write(resposta)

    st.session_state.historico.append(
        {"role": "assistant", "conteudo": resposta, "ferramentas": eventos}
    )