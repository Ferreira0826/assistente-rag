"""
ETAPA 4 — O MESMO AGENTE, FORMALIZADO COMO GRAFO (LangGraph).

POR QUE ESTE ARQUIVO EXISTE JUNTO COM O agente.py
-------------------------------------------------
O `agente.py` implementa o loop de function calling na mão, com um
`for` e um `if`. Ele funciona e continua no projeto de propósito.

Este arquivo faz EXATAMENTE A MESMA COISA usando LangGraph. A comparação
é o ponto: um framework de agentes não faz mágica nenhuma, ele dá
estrutura ao loop que você escreveria de qualquer jeito. Quem entende o
loop manual sabe o que o framework está fazendo — e sabe depurar quando
o framework se comporta de forma inesperada.

O LOOP DO agente.py                    VIRA ESTE GRAFO
-------------------                    ---------------
    for iteracao in range(...):            START
        resposta = modelo(...)               |
        if not tool_calls:                   v
            return resposta            +-----------+
        for chamada in tool_calls:     |  modelo   |<----+
            executa e devolve          +-----------+     |
                                             |           |
                                       tem tool_calls?   |
                                        /          \\     |
                                     sim           não   |
                                      |             |    |
                                      v             v    |
                               +------------+      END   |
                               | ferramentas|-----------+
                               +------------+

O QUE O FRAMEWORK ACRESCENTA DE VERDADE
---------------------------------------
1. Estado explícito e tipado, em vez de uma lista solta.
2. Checkpointer: a memória de conversa (Etapa 3) vira infraestrutura,
   com `thread_id` separando conversas diferentes.
3. Limite de recursão embutido, no lugar do `max_iteracoes` manual.
4. O grafo é inspecionável — dá pra exportar o diagrama (usado no README).

O que NÃO muda: as ferramentas (`ferramentas.py`) e o cliente da OpenAI
são reaproveitados sem nenhuma alteração.
"""

import json
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agente import INSTRUCAO_SISTEMA, criar_cliente
from ferramentas import FERRAMENTAS, criar_executor

load_dotenv()


# =====================================================================
# O ESTADO
# =====================================================================
#
# No agente.py o estado era uma lista de dicionários passeando entre
# funções. Aqui ele é um tipo declarado.
#
# `Annotated[list, operator.add]` define um REDUCER: quando um nó
# devolve {"mensagens": [x]}, o LangGraph SOMA essa lista ao estado em
# vez de substituí-lo. Sem o reducer, cada nó apagaria o histórico
# inteiro ao retornar — e o agente perderia a memória a cada passo.

class EstadoAgente(TypedDict):
    mensagens: Annotated[list, operator.add]


# =====================================================================
# OS NÓS
# =====================================================================
#
# Um nó é uma função que recebe o estado e devolve o que mudou.

def criar_no_modelo(cliente, modelo: str):
    """Nó que chama o modelo. Equivale ao PASSO 1 do loop manual."""

    def no_modelo(estado: EstadoAgente) -> dict:
        resposta = cliente.chat.completions.create(
            model=modelo,
            messages=estado["mensagens"],
            tools=FERRAMENTAS,
            tool_choice="auto",
        )
        mensagem = resposta.choices[0].message
        # O reducer cuida de anexar isto ao histórico.
        return {"mensagens": [mensagem.model_dump(exclude_none=True)]}

    return no_modelo


def criar_no_ferramentas(executar_ferramenta, verbose: bool = True, ao_usar_ferramenta=None):
    """Nó que executa as ferramentas. Equivale aos PASSOS 3 e 4."""

    def no_ferramentas(estado: EstadoAgente) -> dict:
        ultima = estado["mensagens"][-1]
        chamadas = ultima.get("tool_calls") or []

        resultados = []
        for chamada in chamadas:
            funcao = chamada["function"]
            nome = funcao["name"]

            try:
                argumentos = json.loads(funcao.get("arguments") or "{}")
            except json.JSONDecodeError:
                argumentos = {}

            if verbose:
                print(f"  [grafo] ferramenta: {nome}({argumentos})")

            resultado = executar_ferramenta(nome, argumentos)

            if ao_usar_ferramenta:
                ao_usar_ferramenta(nome, argumentos, resultado)

            resultados.append({
                "role": "tool",
                "tool_call_id": chamada["id"],
                "content": str(resultado),
            })

        return {"mensagens": resultados}

    return no_ferramentas


def decidir_proximo(estado: EstadoAgente) -> str:
    """
    Aresta condicional: o `if not tool_calls` do loop manual.

    É aqui que mora a decisão do agente. Se o modelo pediu ferramenta,
    vai pro nó de ferramentas e depois VOLTA pro modelo — esse ciclo é
    o que permite encadear chamadas (data_atual e depois dias_entre_datas).
    Se não pediu, a resposta está pronta e o grafo termina.
    """
    ultima = estado["mensagens"][-1]
    return "ferramentas" if ultima.get("tool_calls") else END


# =====================================================================
# O GRAFO
# =====================================================================

def construir_grafo(cliente, colecao, modelo: str | None = None, verbose: bool = True, ao_usar_ferramenta=None):
    """Monta e compila o grafo do agente."""
    modelo = modelo or os.getenv("MODELO_AGENTE", "gpt-4o-mini")
    executar_ferramenta = criar_executor(colecao)

    grafo = StateGraph(EstadoAgente)

    grafo.add_node("modelo", criar_no_modelo(cliente, modelo))
    grafo.add_node(
        "ferramentas",
        criar_no_ferramentas(executar_ferramenta, verbose, ao_usar_ferramenta),
    )

    grafo.add_edge(START, "modelo")

    # Do modelo, o caminho depende da decisão dele
    grafo.add_conditional_edges(
        "modelo",
        decidir_proximo,
        {"ferramentas": "ferramentas", END: END},
    )

    # Executou a ferramenta, volta pro modelo. ESTA é a aresta que fecha
    # o ciclo — sem ela não existe agente, só uma chamada única.
    grafo.add_edge("ferramentas", "modelo")

    # O checkpointer é a memória (Etapa 3) virando infraestrutura: o
    # LangGraph guarda o estado por thread_id, então basta enviar a
    # mensagem NOVA a cada turno em vez do histórico inteiro.
    return grafo.compile(checkpointer=MemorySaver())


class ConversaGrafo:
    """Mesma interface da classe Conversa do agente.py, agora sobre o grafo."""

    def __init__(
        self,
        cliente,
        colecao,
        thread_id: str = "principal",
        verbose: bool = True,
        ao_usar_ferramenta=None,
    ):
        self.grafo = construir_grafo(
            cliente, colecao, verbose=verbose, ao_usar_ferramenta=ao_usar_ferramenta
        )
        self.thread_id = thread_id
        self.iniciada = False

    def perguntar(self, pergunta: str) -> str:
        # A instrução de sistema entra uma vez só: o checkpointer guarda
        # o resto do histórico.
        novas = []
        if not self.iniciada:
            novas.append({"role": "system", "content": INSTRUCAO_SISTEMA})
            self.iniciada = True
        novas.append({"role": "user", "content": pergunta})

        resultado = self.grafo.invoke(
            {"mensagens": novas},
            config={
                "configurable": {"thread_id": self.thread_id},
                # Equivale ao max_iteracoes do loop manual: impede que um
                # agente em loop rode para sempre gastando API.
                "recursion_limit": 12,
            },
        )
        return resultado["mensagens"][-1].get("content") or ""

    def limpar(self):
        """Nova thread = conversa nova, sem apagar a anterior."""
        self.thread_id = f"{self.thread_id}_novo"
        self.iniciada = False

    def exportar_diagrama(self, caminho: str = "diagrama_agente.png") -> str | None:
        """Gera o PNG do grafo para o README."""
        try:
            png = self.grafo.get_graph().draw_mermaid_png()
            with open(caminho, "wb") as arquivo:
                arquivo.write(png)
            return caminho
        except Exception as erro:
            print(f"  (não foi possível gerar o PNG: {erro})")
            print("  Diagrama em Mermaid, para colar no README:\n")
            print(self.grafo.get_graph().draw_mermaid())
            return None


if __name__ == "__main__":
    from base_vetorial import criar_base

    colecao = criar_base("documentos")
    cliente = criar_cliente()
    conversa = ConversaGrafo(cliente, colecao)

    print("Agente em LangGraph. Mesmo comportamento do agente.py.")
    print("Comandos: 'limpar' | 'diagrama' gera o PNG do grafo | 'sair'\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        if pergunta.lower() == "limpar":
            conversa.limpar()
            print("Conversa reiniciada.\n")
            continue
        if pergunta.lower() == "diagrama":
            caminho = conversa.exportar_diagrama()
            if caminho:
                print(f"Diagrama salvo em '{caminho}'.\n")
            continue

        print(f"\nAssistente: {conversa.perguntar(pergunta)}\n")