"""
Peças 4 e 5 do RAG: montar o prompt com contexto + chamar o LLM.
VERSÃO LOCAL COM OLLAMA — ajustada para respostas mais completas.

Mudanças em relação à versão anterior:
- Prompt pede explicitamente uma resposta detalhada e bem desenvolvida.
- temperature um pouco maior (0.4) dá mais "fôlego" ao modelo,
  sem deixar de ser fiel ao contexto.
- num_predict alto garante espaço para respostas longas.
"""

import ollama

MODELO = "llama3.2"


def montar_prompt(pergunta: str, contexto: list[str]) -> str:
    """
    Monta o prompt final. Agora pede uma resposta COMPLETA e
    DETALHADA, aproveitando todo o contexto relevante — mas ainda
    proibindo inventar o que não está lá (anti-alucinação).
    """
    blocos = "\n\n".join(f"[Trecho {i+1}]\n{c}" for i, c in enumerate(contexto))

    prompt = f"""Você é um assistente que responde perguntas usando APENAS o contexto abaixo.

Instruções:
- Responda em português, de forma completa, clara e bem desenvolvida.
- Use TODAS as informações relevantes do contexto para elaborar a resposta.
- Se houver vários pontos, organize a resposta cobrindo cada um deles.
- Se a resposta não estiver no contexto, diga que não encontrou a informação nos documentos.
- Não invente dados que não estejam no contexto.

CONTEXTO:
{blocos}

PERGUNTA: {pergunta}

RESPOSTA (completa e detalhada):"""
    return prompt


def responder(pergunta: str, contexto: list[str]) -> str:
    """Chama o LLM local via Ollama com parâmetros para respostas ricas."""
    prompt = montar_prompt(pergunta, contexto)

    resposta = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 1500,   # bastante espaço para respostas longas
            "temperature": 0.4,    # um pouco mais de elaboração
        },
    )
    return resposta["message"]["content"]


if __name__ == "__main__":
    contexto_teste = [
        "O SISAB é o Sistema de Informação em Saúde para a Atenção Básica.",
        "O e-SUS APS envia os dados de produção para o SISAB.",
        "O SISAB serve de base para o financiamento da Atenção Primária.",
    ]
    print(responder("O que é o SISAB e para que serve?", contexto_teste))