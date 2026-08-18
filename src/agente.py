"""
ETAPAS 2 e 3 — O LOOP DO AGENTE (function calling) + MEMÓRIA DE CONVERSA.

ESTE ARQUIVO É O AGENTE. Todo o resto é infraestrutura.

O CICLO, EM 5 PASSOS
--------------------
    1. Mando a conversa + a lista de ferramentas pro modelo.
    2. O modelo responde uma de duas coisas:
         (a) uma resposta em texto  -> acabou, devolvo pro usuário
         (b) um pedido de tool call -> ele quer usar uma ferramenta
    3. Se for (b): EU executo a função no meu código.
    4. Devolvo o resultado pro modelo como mensagem de role "tool".
    5. Volto pro passo 1. O modelo agora vê o resultado e decide de novo.

O modelo NUNCA executa nada. Ele só diz "quero chamar buscar_documentos
com query='SISAB'". Quem executa é você. O modelo é o cérebro que decide;
seu código é o braço que age.

A MEMÓRIA (Etapa 3)
-------------------
A API é SEM ESTADO. Cada chamada é independente: o modelo não lembra de
nada da chamada anterior. A "memória" de um chatbot é ilusão — ele
reenvia a conversa inteira toda vez.

Por isso a memória aqui é só uma lista que não é descartada entre as
perguntas. A classe `Conversa` cuida disso, e resolve os dois problemas
que aparecem quando essa lista cresce (custo e poda segura).
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from ferramentas import FERRAMENTAS, criar_executor

load_dotenv()


INSTRUCAO_SISTEMA = """Você é um assistente especializado em documentos de saúde pública.

Regras:
- Para perguntas sobre o conteúdo dos documentos, SEMPRE use a ferramenta buscar_documentos antes de responder. Nunca responda de memória sobre o conteúdo deles.
- Se a busca não retornar nada relevante, diga claramente que a informação não consta nos documentos indexados. Não invente.
- Sempre cite a fonte (o nome do arquivo) que aparece nos trechos retornados.
- Para qualquer conta, use a ferramenta calcular em vez de calcular de cabeça.
- Você não sabe a data de hoje: use data_atual quando precisar dela.
- Para diferença entre datas, use dias_entre_datas em vez de montar a conta manualmente.
- Não expanda siglas por conta própria. Se uma sigla aparecer nos trechos recuperados mas o significado dela não estiver escrito ali, use apenas a sigla e informe que a definição não consta nos trechos consultados. Nunca deduza o significado a partir de títulos de seção próximos.
- Considere o histórico da conversa: se o usuário disser "e sobre isso?" ou "detalhe melhor", entenda a que ele se refere pelas mensagens anteriores.
- Responda em português, de forma completa e bem organizada."""


def _sanear_variaveis_vazias():
    """
    Remove do ambiente variáveis de configuração VAZIAS.

    Uma linha `OPENAI_BASE_URL=` no .env coloca string vazia no ambiente.
    O SDK da OpenAI testa essa variável com `is None`, então a string
    vazia passa e ele tenta usá-la como endereço da API — o httpx quebra
    com "Request URL is missing an 'http://' or 'https://' protocol",
    erro que não diz nada sobre .env.
    """
    for variavel in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        valor = os.environ.get(variavel)
        if valor is not None and not valor.strip():
            del os.environ[variavel]


def criar_cliente() -> OpenAI:
    """Cria o cliente da API."""
    _sanear_variaveis_vazias()

    chave = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not chave:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada. Crie um arquivo .env na raiz do "
            "projeto com OPENAI_API_KEY=sk-..."
        )

    # base_url explícito (nunca None) para o SDK não consultar o ambiente.
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    return OpenAI(
        api_key=chave,
        base_url=base_url or "https://api.openai.com/v1",
    )


# =====================================================================
# O LOOP DO AGENTE (Etapa 2)
# =====================================================================

def executar_agente(
    cliente: OpenAI,
    mensagens: list[dict],
    executar_ferramenta,
    modelo: str | None = None,
    max_iteracoes: int = 5,
    verbose: bool = True,
    ao_usar_ferramenta=None,
) -> str:
    """
    Roda o ciclo até o modelo produzir uma resposta final.

    `mensagens` é MODIFICADA no lugar, de propósito: quem chamou fica com
    o histórico completo, inclusive as chamadas de ferramenta. É isso que
    permite a memória da Etapa 3 sem reescrever nada aqui.

    `ao_usar_ferramenta` é um callback opcional chamado como
    (nome, argumentos, resultado) a cada execução de ferramenta. Serve
    para a interface mostrar o que o agente está fazendo — no terminal
    basta o print, mas no Streamlit precisamos capturar os eventos.
    """
    modelo = modelo or os.getenv("MODELO_AGENTE", "gpt-4o-mini")

    # Trava de segurança: um agente pode entrar em loop, chamando a mesma
    # ferramenta pra sempre. Sem limite, isso é uma conta de API subindo
    # sozinha. Todo loop de agente em produção tem esse teto.
    for iteracao in range(1, max_iteracoes + 1):

        # PASSO 1 — manda a conversa + as ferramentas disponíveis
        resposta = cliente.chat.completions.create(
            model=modelo,
            messages=mensagens,
            tools=FERRAMENTAS,
            tool_choice="auto",   # o modelo decide. É o ponto do agente.
        )

        mensagem_modelo = resposta.choices[0].message

        # A resposta entra no histórico ANTES de qualquer coisa: sem isso,
        # a próxima chamada teria um tool_call_id órfão e a API rejeita.
        mensagens.append(mensagem_modelo.model_dump(exclude_none=True))

        # PASSO 2 — o modelo pediu ferramenta, ou já respondeu?
        if not mensagem_modelo.tool_calls:
            if verbose:
                print(f"  [iteração {iteracao}] resposta final")
            return mensagem_modelo.content or ""

        # PASSOS 3 e 4 — executo cada ferramenta e devolvo o resultado.
        # O modelo pode pedir várias de uma vez, por isso é um for.
        for chamada in mensagem_modelo.tool_calls:
            nome = chamada.function.name

            # Os argumentos vêm como STRING JSON, e o modelo às vezes
            # gera JSON inválido — daí o try.
            try:
                argumentos = json.loads(chamada.function.arguments or "{}")
            except json.JSONDecodeError:
                argumentos = {}

            if verbose:
                print(f"  [iteração {iteracao}] tool: {nome}({argumentos})")

            resultado = executar_ferramenta(nome, argumentos)

            if ao_usar_ferramenta:
                ao_usar_ferramenta(nome, argumentos, resultado)

            mensagens.append({
                "role": "tool",              # role específica de resultado
                "tool_call_id": chamada.id,  # amarra o resultado ao pedido
                "content": str(resultado),
            })

        # PASSO 5 — volta pro topo; o modelo agora vê o resultado.

    return (
        "Não consegui concluir a resposta dentro do limite de "
        f"{max_iteracoes} iterações. Tente reformular a pergunta."
    )


# =====================================================================
# MEMÓRIA DE CONVERSA (Etapa 3)
# =====================================================================

class Conversa:
    """
    Mantém o histórico entre perguntas.

    Sem isso, cada pergunta nasce do zero e "e detalhe melhor esse
    segundo ponto" não significa nada. Com isso, o agente acompanha o
    fio da conversa.
    """

    def __init__(
        self,
        cliente: OpenAI,
        colecao,
        instrucao_sistema: str = INSTRUCAO_SISTEMA,
        max_mensagens: int = 20,
        verbose: bool = True,
        ao_usar_ferramenta=None,
    ):
        self.cliente = cliente
        self.executar_ferramenta = criar_executor(colecao)
        self.verbose = verbose
        self.max_mensagens = max_mensagens
        self.ao_usar_ferramenta = ao_usar_ferramenta

        # A mensagem de sistema fica SEMPRE na posição 0 e nunca é podada.
        # Se ela sumir, o agente esquece as próprias regras no meio da
        # conversa — e o sintoma (respostas piorando com o tempo) é bem
        # difícil de diagnosticar depois.
        self.mensagens: list[dict] = [
            {"role": "system", "content": instrucao_sistema}
        ]

    def perguntar(self, pergunta: str) -> str:
        """Faz uma pergunta mantendo todo o contexto anterior."""
        self.mensagens.append({"role": "user", "content": pergunta})

        resposta = executar_agente(
            self.cliente,
            self.mensagens,      # a MESMA lista, turno após turno
            self.executar_ferramenta,
            verbose=self.verbose,
            ao_usar_ferramenta=self.ao_usar_ferramenta,
        )

        self._podar()
        return resposta

    def _podar(self):
        """
        Corta o histórico antigo quando ele fica grande demais.

        POR QUE PODAR:
        Você reenvia a conversa inteira a cada chamada. Sem poda, o custo
        e a latência crescem sem parar, e em algum ponto estoura o limite
        de contexto do modelo.

        A SUTILEZA QUE QUEBRA TUDO:
        Não dá pra cortar em qualquer ponto. Uma mensagem do assistente
        com `tool_calls` e as mensagens `tool` correspondentes formam um
        BLOCO INDIVISÍVEL. Se o corte cair no meio, sobra um tool_call
        sem resultado (ou um resultado sem pedido) e a API rejeita a
        requisição inteira com erro 400.

        A regra segura: só cortar imediatamente antes de uma mensagem
        "user". Todo bloco de ferramenta vive entre duas mensagens de
        usuário, então esse ponto de corte nunca parte um bloco.
        """
        if len(self.mensagens) <= 1:
            return

        sistema = self.mensagens[0]
        corpo = self.mensagens[1:]

        if len(corpo) <= self.max_mensagens:
            return

        corte = len(corpo) - self.max_mensagens

        # Empurra o corte pra frente até cair numa mensagem de usuário
        while corte < len(corpo) and corpo[corte].get("role") != "user":
            corte += 1

        # Sem fronteira segura à frente? Melhor não cortar nada.
        if corte >= len(corpo):
            return

        descartadas = corte
        self.mensagens = [sistema] + corpo[corte:]

        if self.verbose and descartadas:
            print(f"  [memória] {descartadas} mensagens antigas descartadas")

    def limpar(self):
        """Recomeça a conversa, mantendo só a instrução de sistema."""
        self.mensagens = self.mensagens[:1]

    def resumo(self) -> str:
        """Quantas mensagens de cada tipo estão na memória agora."""
        contagem: dict[str, int] = {}
        for m in self.mensagens:
            papel = m.get("role", "?")
            contagem[papel] = contagem.get(papel, 0) + 1
        partes = [f"{papel}={qtd}" for papel, qtd in sorted(contagem.items())]
        return f"{len(self.mensagens)} mensagens ({', '.join(partes)})"


def responder(cliente, colecao, pergunta: str, verbose: bool = True) -> str:
    """Atalho para uma pergunta ISOLADA, sem memória. Útil em testes."""
    mensagens = [
        {"role": "system", "content": INSTRUCAO_SISTEMA},
        {"role": "user", "content": pergunta},
    ]
    return executar_agente(
        cliente, mensagens, criar_executor(colecao), verbose=verbose
    )


# =====================================================================
# Teste pela linha de comando
# =====================================================================
#
# Rode a partir da RAIZ do projeto (o ChromaDB usa caminho relativo):
#     python src/agente.py
#
if __name__ == "__main__":
    from base_vetorial import criar_base

    colecao = criar_base("documentos")
    cliente = criar_cliente()
    conversa = Conversa(cliente, colecao)

    print("Agente com memória de conversa.")
    print("Comandos: 'limpar' zera o histórico | 'memoria' mostra o estado | 'sair'")
    print("\nPra testar a MEMÓRIA, faça perguntas encadeadas:")
    print("  1. 'o que os documentos dizem sobre o SISAB?'")
    print("  2. 'detalhe melhor o processo de validação'   <- 'melhor' o quê?")
    print("  3. 'e quem tem acesso a esses relatórios?'     <- 'esses' quais?\n")

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
            print("Histórico zerado.\n")
            continue
        if pergunta.lower() in ("memoria", "memória"):
            print(f"  {conversa.resumo()}\n")
            continue

        resposta = conversa.perguntar(pergunta)
        print(f"\nAssistente: {resposta}\n")