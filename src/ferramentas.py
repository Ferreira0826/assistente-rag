"""
ETAPA 1 — As FERRAMENTAS (tools) do agente.

DIFERENÇA ENTRE RAG PURO E AGENTE
---------------------------------
No `rag.py`, o retrieval é chamado SEMPRE, direto no código:

    chunks = buscar(colecao, pergunta)   # acontece toda vez, sem escolha

Aqui, o retrieval vira UMA DAS OPÇÕES que o modelo pode escolher chamar.
Se o usuário perguntar "quanto é 15% de 320?", o modelo NÃO vai buscar nos
documentos — vai chamar `calcular`. Se perguntar "o que diz o documento
sobre o SISAB?", aí sim ele chama `buscar_documentos`.

Essa DECISÃO é o que caracteriza um agente. Por isso são 3 ferramentas e
não 1: com uma só, não existe decisão a tomar.

O QUE TEM NESTE ARQUIVO
-----------------------
1. As implementações Python das 3 ferramentas.
2. A lista FERRAMENTAS — o schema JSON que descreve cada uma para o modelo.
3. `criar_executor(colecao)` — a ponte entre o nome que o modelo devolve
   ("buscar_documentos") e a função Python de verdade.
"""

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from base_vetorial import buscar_com_fontes


# =====================================================================
# FERRAMENTA 1 — calcular
# =====================================================================
#
# NOTA DE SEGURANÇA (isso conta ponto em entrevista):
# A implementação óbvia seria `eval(expressao)`. NÃO faça isso. `eval`
# executa qualquer código Python — e a expressão aqui vem de um LLM, que
# por sua vez recebe texto do usuário. Alguém digitando a coisa certa
# consegue ler arquivos ou rodar comandos na sua máquina.
#
# A solução: parsear a expressão com `ast` e percorrer a árvore
# permitindo APENAS nós de aritmética. Qualquer outra coisa (chamada de
# função, nome de variável, import) levanta erro antes de executar.

_OPERADORES_PERMITIDOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,   # número negativo: -5
    ast.UAdd: operator.pos,   # número positivo explícito: +5
}


# Limites contra exaustão de recursos. O parser com ast impede EXECUÇÃO de
# código, mas não impede que uma expressão legítima consuma toda a memória
# da máquina. "9**9**9**9" é aritmética pura e trava o processo.
TAMANHO_MAXIMO_EXPRESSAO = 200   # caracteres
EXPOENTE_MAXIMO = 1000
MAGNITUDE_MAXIMA = 10 ** 100     # teto do resultado e dos operandos
PROFUNDIDADE_MAXIMA = 20         # aninhamento de parênteses


def _avaliar_no(no, profundidade: int = 0):
    """Percorre a árvore sintática, aceitando só aritmética pura."""
    if profundidade > PROFUNDIDADE_MAXIMA:
        raise ValueError("Expressão aninhada demais.")
    # Um número literal
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        if abs(no.value) > MAGNITUDE_MAXIMA:
            raise ValueError("Número grande demais.")
        return no.value

    # Operação binária: 2 + 3, 10 * 4, 2 ** 8
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES_PERMITIDOS:
        esquerda = _avaliar_no(no.left, profundidade + 1)
        direita = _avaliar_no(no.right, profundidade + 1)

        # A potenciação é o vetor perigoso: um expoente pequeno na tela
        # gera um número gigante na memória. Checamos ANTES de calcular.
        if isinstance(no.op, ast.Pow):
            if abs(direita) > EXPOENTE_MAXIMO:
                raise ValueError("Expoente grande demais.")
            if abs(esquerda) > 10 ** 6 and abs(direita) > 100:
                raise ValueError("Potência resultaria em número grande demais.")

        resultado = _OPERADORES_PERMITIDOS[type(no.op)](esquerda, direita)

        if isinstance(resultado, (int, float)) and abs(resultado) > MAGNITUDE_MAXIMA:
            raise ValueError("Resultado grande demais.")
        return resultado

    # Operação unária: -7
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES_PERMITIDOS:
        return _OPERADORES_PERMITIDOS[type(no.op)](
            _avaliar_no(no.operand, profundidade + 1)
        )

    raise ValueError("Expressão contém elementos não permitidos.")


def calcular(expressao: str) -> str:
    """
    Avalia uma expressão aritmética simples com segurança.

    Aceita: + - * / // % ** e parênteses.
    Rejeita: qualquer outra coisa, além de expressões que consumiriam
    memória ou tempo excessivos.
    """
    if len(expressao) > TAMANHO_MAXIMO_EXPRESSAO:
        return "Erro: expressão longa demais."

    arvore = ast.parse(expressao, mode="eval")
    resultado = _avaliar_no(arvore.body)
    return str(resultado)


# =====================================================================
# FERRAMENTA 2 — data_atual
# =====================================================================
#
# Parece trivial, e é — de propósito. Ela existe por dois motivos:
#
# 1. O modelo NÃO sabe que dia é hoje. O conhecimento dele foi congelado
#    no treinamento. Sem essa ferramenta, ele chuta ou inventa.
# 2. É a ferramenta mais barata possível de demonstrar o modelo ESCOLHENDO
#    entre opções — e ainda permite perguntas compostas, que forçam ele a
#    encadear duas chamadas. Ex.: "quantos dias faltam pro fim do ano?"
#    exige data_atual E calcular, em sequência.

def data_atual() -> str:
    """Retorna a data e hora atuais no fuso de Brasília."""
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))

    dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
            "sexta-feira", "sábado", "domingo"]
    dia_semana = dias[agora.weekday()]

    return (
        f"Data e hora atuais (fuso de Brasília): "
        f"{agora.strftime('%d/%m/%Y %H:%M')} — {dia_semana}."
    )


# =====================================================================
# FERRAMENTA 3 — dias_entre_datas
# =====================================================================
#
# ESTA FERRAMENTA NASCEU DE UM BUG OBSERVADO.
#
# Perguntado "quantos dias faltam pro fim do ano?", o agente chamou
# data_atual (certo), e depois montou esta expressão pra calcular:
#
#     (31 - 8) + (30 * 4) + (31 * 3) + (30 + 31)  -> 297
#
# Ele usou o NÚMERO DO MÊS (8) onde devia usar o DIA (17), e somou meses
# que não faltavam. A resposta certa era 136. A ferramenta `calcular`
# fez a conta corretamente — o erro foi de RACIOCÍNIO, não de aritmética.
#
# A lição: tool use conserta o cálculo, não o pensamento. Se o modelo
# precisa decompor um problema em aritmética manual, ele pode decompor
# errado. A solução é elevar o nível da ferramenta: em vez de pedir que
# ele monte a conta, dar uma ferramenta que já resolve o problema inteiro.

def dias_entre_datas(data_inicio: str, data_fim: str) -> str:
    """
    Calcula quantos dias separam duas datas no formato DD/MM/AAAA.

    Usa a aritmética de datas do próprio Python, que já sabe de meses
    com tamanhos diferentes e anos bissextos.
    """
    formato = "%d/%m/%Y"
    try:
        inicio = datetime.strptime(data_inicio.strip(), formato).date()
        fim = datetime.strptime(data_fim.strip(), formato).date()
    except ValueError:
        return (
            "Erro: as datas devem estar no formato DD/MM/AAAA "
            f"(recebi '{data_inicio}' e '{data_fim}')."
        )

    diferenca = (fim - inicio).days
    sentido = "depois" if diferenca >= 0 else "antes"
    return (
        f"De {data_inicio} a {data_fim} há {abs(diferenca)} dias "
        f"({data_fim} vem {sentido} de {data_inicio})."
    )


# =====================================================================
# O SCHEMA DAS FERRAMENTAS (formato da API OpenAI)
# =====================================================================
#
# Esta é a parte que o modelo realmente "lê". Ele nunca vê seu código
# Python — ele vê só esta descrição e decide, a partir dela, qual função
# chamar e com quais argumentos.
#
# POR ISSO A DESCRIPTION É CÓDIGO, NÃO COMENTÁRIO. Description ruim =
# agente que escolhe a ferramenta errada. É o ponto que mais gente erra
# ao começar com function calling: capricha aqui, não no docstring.
#
# Estrutura de cada item:
#   type: sempre "function"
#   function.name: tem que bater EXATAMENTE com a chave no executor
#   function.description: quando usar esta ferramenta (e quando não usar)
#   function.parameters: JSON Schema dos argumentos

FERRAMENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_documentos",
            "description": (
                "Busca trechos relevantes na base de documentos indexados "
                "(normas, manuais e documentos técnicos de saúde pública). "
                "Use SEMPRE que a pergunta for sobre o conteúdo dos documentos, "
                "sobre sistemas de saúde como SISAB, e-SUS APS ou DATASUS, ou "
                "sobre qualquer informação institucional específica. "
                "NÃO use para conhecimento geral, contas ou datas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "O termo de busca semântica. Reformule a pergunta do "
                            "usuário destacando os conceitos-chave, em vez de copiar "
                            "a frase inteira. Ex.: para 'me explica direitinho como "
                            "funciona o envio pro SISAB', busque 'envio de dados e-SUS "
                            "APS para o SISAB'."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": (
                            "Quantos trechos retornar. Use 5 por padrão. "
                            "Aumente até 8 se a pergunta for ampla."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular",
            "description": (
                "Avalia uma expressão aritmética e retorna o resultado exato. "
                "Use para QUALQUER conta — percentuais, somas, médias, diferenças. "
                "Modelos de linguagem erram aritmética; esta ferramenta não erra. "
                "Não use para perguntas conceituais sobre matemática."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expressao": {
                        "type": "string",
                        "description": (
                            "Expressão aritmética em sintaxe Python, usando apenas "
                            "números e os operadores + - * / // % ** e parênteses. "
                            "Ex.: '320 * 0.15' ou '(1250 + 890) / 2'."
                        ),
                    }
                },
                "required": ["expressao"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dias_entre_datas",
            "description": (
                "Calcula quantos dias existem entre duas datas. "
                "Use SEMPRE que a pergunta envolver prazo, contagem de dias, "
                "ou quanto falta para uma data. NÃO monte esse cálculo com a "
                "ferramenta calcular: decompor calendário em aritmética manual "
                "é fonte de erro (meses têm tamanhos diferentes e há anos "
                "bissextos). Se precisar da data de hoje, consulte data_atual antes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_inicio": {
                        "type": "string",
                        "description": "Data inicial no formato DD/MM/AAAA.",
                    },
                    "data_fim": {
                        "type": "string",
                        "description": "Data final no formato DD/MM/AAAA.",
                    },
                },
                "required": ["data_inicio", "data_fim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_atual",
            "description": (
                "Retorna a data e a hora de hoje no fuso de Brasília. "
                "Use sempre que a pergunta envolver 'hoje', 'agora', 'este mês', "
                "'este ano' ou qualquer cálculo de prazo. Você não sabe a data "
                "atual por conta própria — sempre consulte esta ferramenta."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# =====================================================================
# O EXECUTOR — a ponte entre o nome e a função
# =====================================================================
#
# PROBLEMA DE DESIGN QUE ISTO RESOLVE:
# `buscar_documentos` precisa da `colecao` do ChromaDB. Mas o modelo só
# devolve os argumentos que estão no schema (query, k) — ele não conhece
# e nem deve conhecer a coleção.
#
# Solução: uma FACTORY. `criar_executor(colecao)` devolve uma função que
# já "carrega" a coleção por closure. Sem variável global, e fácil de
# testar (é só passar uma coleção falsa).

def criar_executor(colecao):
    """
    Recebe a coleção do ChromaDB e devolve a função que executa
    ferramentas pelo nome.
    """

    K_MAXIMO = 10
    TAMANHO_MAXIMO_QUERY = 500

    def buscar_documentos(query: str, k: int = 5) -> str:
        """Encapsula o retrieval que já existe, formatando pro modelo ler."""
        # O modelo escolhe o k, então ele é ENTRADA NÃO CONFIÁVEL. Sem teto,
        # um k=100000 puxaria a base inteira para o contexto — lento e caro.
        # Toda tool deve validar os próprios argumentos: quem os produziu foi
        # um LLM que pode ter sido manipulado pela pergunta do usuário.
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = 5
        k = max(1, min(k, K_MAXIMO))

        query = str(query)[:TAMANHO_MAXIMO_QUERY]

        achados = buscar_com_fontes(colecao, query, k=k)

        if not achados:
            return (
                "Nenhum trecho relevante encontrado na base de documentos. "
                "Informe ao usuário que a informação não consta nos documentos indexados."
            )

        # Devolvemos a ORIGEM junto com o texto. Isso permite que o agente
        # cite a fonte na resposta final — coisa que o RAG puro perdia,
        # porque o `buscar()` original descartava os metadados.
        blocos = [
            f"[Trecho {i} | fonte: {item['origem']}]\n{item['texto']}"
            for i, item in enumerate(achados, 1)
        ]
        return "\n\n".join(blocos)

    # O nome da chave TEM que ser idêntico ao "name" no schema acima.
    ferramentas_disponiveis = {
        "buscar_documentos": buscar_documentos,
        "calcular": calcular,
        "data_atual": data_atual,
        "dias_entre_datas": dias_entre_datas,
    }

    def executar(nome: str, argumentos: dict) -> str:
        """
        Executa a ferramenta pedida e devolve o resultado como string.

        DETALHE IMPORTANTE: erros viram TEXTO em vez de exceção.
        Se a ferramenta explodir e você deixar a exceção subir, o agente
        morre. Devolvendo a mensagem de erro pro modelo, ele consegue se
        recuperar — tentar outra ferramenta, corrigir o argumento, ou
        avisar o usuário. Isso é tratamento de erro de agente.
        """
        funcao = ferramentas_disponiveis.get(nome)

        if funcao is None:
            return f"Erro: a ferramenta '{nome}' não existe."

        try:
            return str(funcao(**argumentos))
        except Exception as erro:
            return f"Erro ao executar '{nome}': {erro}"

    return executar


# =====================================================================
# Teste isolado (não precisa de API key nem de ChromaDB)
# =====================================================================
if __name__ == "__main__":
    print("Testando 'calcular':")
    print("  320 * 0.15 =", calcular("320 * 0.15"))
    print("  (1250 + 890) / 2 =", calcular("(1250 + 890) / 2"))

    print("\nTestando as proteções do 'calcular':")
    for tentativa in [
        "__import__('os').listdir('.')",   # execução de código
        "9**9**9**9",                      # exaustão de memória
        "10**5000",                        # expoente absurdo
        "1+" * 30 + "1",                   # aninhamento profundo de verdade
        "5" * 300,                         # expressão longa demais
    ]:
        try:
            saida = calcular(tentativa)
            if str(saida).startswith("Erro:"):
                print(f"  Bloqueado: {saida}")
            else:
                print(f"  FALHOU: '{tentativa[:30]}' passou!")
        except ValueError as e:
            print(f"  Bloqueado: {e}")

    print("\nTestando 'data_atual':")
    print(" ", data_atual())

    print("\nTestando 'dias_entre_datas':")
    print(" ", dias_entre_datas("18/08/2026", "31/12/2026"))
    print(" ", dias_entre_datas("data errada", "31/12/2026"))

    print(f"\n{len(FERRAMENTAS)} ferramentas definidas:")
    for f in FERRAMENTAS:
        print(f"  - {f['function']['name']}")