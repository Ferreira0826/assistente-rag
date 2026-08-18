"""
Peça 1 do RAG (versão com OCR): carregar documentos e quebrar em chunks.

Carregador híbrido de PDF: para cada arquivo, primeiro tenta extrair
texto com pypdf (rápido). Se o PDF for escaneado (pypdf devolve vazio),
aciona o OCR automaticamente com Tesseract.

--- ALTERAÇÃO DA PARADA TÉCNICA ---
`quebrar_em_chunks()` foi reescrita. A versão anterior fatiava o texto
por contagem bruta de caracteres (`texto[inicio:fim]`), o que cortava
palavras e frases no meio e degradava a qualidade dos embeddings.
A nova versão respeita fronteiras de parágrafo e de frase.
Nada mais no arquivo mudou.
"""

import os
import re
import shutil
from pathlib import Path

from pypdf import PdfReader

# Bibliotecas de OCR (só usadas quando o PDF é escaneado)
import pytesseract
from pdf2image import convert_from_path

# ==========================================================
# CONFIGURAÇÃO DO OCR
# ==========================================================
#
# Os caminhos vêm do .env, NÃO ficam fixos no código. Dois motivos:
#
# 1. Caminho fixo no código deixa o projeto impossível de rodar em outra
#    máquina sem editar o fonte — e um repositório de portfólio precisa
#    funcionar para quem clonar.
# 2. Um caminho como C:\Users\<matrícula>\... publica o login do autor
#    no repositório. Configuração de máquina vai no .env; o .env não é
#    versionado.
#
# Se as variáveis não estiverem definidas, tentamos achar os executáveis
# no PATH do sistema — que é como fica numa instalação Linux ou macOS.

TESSERACT_CMD = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
POPPLER_PATH = os.getenv("POPPLER_PATH") or None

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def ler_pdf(caminho: str) -> str:
    """
    Lê um PDF tentando primeiro o modo rápido (pypdf).
    Se não vier texto, cai para OCR.
    """
    leitor = PdfReader(caminho)
    texto = ""
    for pagina in leitor.pages:
        texto += (pagina.extract_text() or "") + "\n"

    if len(texto.strip()) > 20:
        return texto

    if not TESSERACT_CMD:
        print(
            f"  [AVISO] '{Path(caminho).name}' parece escaneado, mas o Tesseract "
            "não foi encontrado. Defina TESSERACT_CMD no .env. Pulando o arquivo."
        )
        return ""

    print(f"  [OCR] '{Path(caminho).name}' parece escaneado. Rodando OCR (pode demorar)...")
    return _ler_pdf_com_ocr(caminho)


def _ler_pdf_com_ocr(caminho: str) -> str:
    """
    Converte cada página do PDF em imagem e roda o Tesseract.
    'lang=por' usa o pacote de idioma português.
    """
    # poppler_path só é passado no Windows; no Linux/macOS o Poppler
    # está no PATH e o argumento deve ficar de fora.
    if POPPLER_PATH:
        paginas = convert_from_path(caminho, poppler_path=POPPLER_PATH)
    else:
        paginas = convert_from_path(caminho)

    texto = ""
    for i, imagem in enumerate(paginas, 1):
        print(f"    OCR na página {i}/{len(paginas)}...")
        texto += pytesseract.image_to_string(imagem, lang="por") + "\n"
    return texto


def ler_txt(caminho: str) -> str:
    """Lê um arquivo .txt simples."""
    return Path(caminho).read_text(encoding="utf-8")


# =====================================================================
# CHUNKING
# =====================================================================
#
# POR QUE A VERSÃO ANTERIOR ERA RUIM:
#
#     fim = inicio + tamanho
#     chunk = texto[inicio:fim]     # corta no caractere 500, doa a quem doer
#
# Isso produzia chunks como "mpos podem ser consultados no LEDI..." —
# começando no meio da palavra "campos". Um pedaço truncado assim gera
# um embedding que representa mal o conteúdo: o modelo vetoriza um texto
# que não é uma unidade de sentido. Resultado prático: a busca não
# encontra o que está no documento.
#
# ESTRATÉGIA NOVA (chunking recursivo por fronteiras):
#   1. Quebra o texto em parágrafos (linha em branco).
#   2. Parágrafo grande demais vira frases (pontuação final).
#   3. Frase grande demais é cortada em palavras — nunca no meio de uma.
#   4. Agrupa essas unidades até chegar perto do tamanho alvo.
#   5. O overlap entre chunks também respeita palavra inteira.
#
# É a mesma ideia do RecursiveCharacterTextSplitter do LangChain, escrita
# à mão. Vale escrever à mão aqui: em entrevista, "usei o splitter do
# LangChain" e "implementei chunking respeitando fronteiras semânticas
# porque o corte cego degradava o embedding" são respostas diferentes.

# Tamanho maior que os 500 originais. Documento técnico tem parágrafo
# longo, e chunk curto demais perde o contexto em volta do termo buscado.
TAMANHO_PADRAO = 900
OVERLAP_PADRAO = 150


def _quebrar_frase_longa(frase: str, tamanho: int) -> list[str]:
    """Última instância: corta por palavras, nunca no meio de uma."""
    palavras = frase.split()
    partes, atual = [], ""

    for palavra in palavras:
        # Caso extremo: uma "palavra" maior que o chunk inteiro — URL
        # gigante, string base64, lixo de OCR. Aqui o corte bruto é o
        # certo, porque não existe fronteira melhor.
        if len(palavra) > tamanho:
            if atual:
                partes.append(atual)
                atual = ""
            for i in range(0, len(palavra), tamanho):
                partes.append(palavra[i:i + tamanho])
            continue

        if not atual:
            atual = palavra
        elif len(atual) + 1 + len(palavra) <= tamanho:
            atual += " " + palavra
        else:
            partes.append(atual)
            atual = palavra

    if atual:
        partes.append(atual)
    return partes


def _quebrar_em_frases(paragrafo: str, tamanho: int) -> list[str]:
    """Divide um parágrafo grande em frases, pela pontuação final."""
    # Corta DEPOIS de . ! ? ou ; seguidos de espaço — o lookbehind
    # preserva a pontuação no fim da frase.
    frases = re.split(r"(?<=[.!?;])\s+", paragrafo)

    unidades = []
    for frase in frases:
        frase = frase.strip()
        if not frase:
            continue
        if len(frase) <= tamanho:
            unidades.append(frase)
        else:
            unidades.extend(_quebrar_frase_longa(frase, tamanho))
    return unidades


def _cauda_por_palavras(texto: str, limite: int) -> str:
    """Pega o final do texto para servir de overlap, sem partir palavra."""
    if len(texto) <= limite:
        return texto

    trecho = texto[-limite:]
    # descarta o primeiro pedaço, que provavelmente é uma palavra cortada
    espaco = trecho.find(" ")
    return trecho[espaco + 1:] if espaco != -1 else trecho


def quebrar_em_chunks(
    texto: str,
    tamanho: int = TAMANHO_PADRAO,
    overlap: int = OVERLAP_PADRAO,
) -> list[str]:
    """
    Quebra o texto em chunks respeitando fronteiras de sentido.

    Cada chunk tem no máximo `tamanho` caracteres e começa repetindo os
    últimos `overlap` caracteres do anterior — assim uma informação que
    cai na fronteira aparece inteira em pelo menos um dos dois.
    """
    if not texto or not texto.strip():
        return []

    # Normaliza espaçamento: 3+ quebras de linha viram parágrafo simples
    texto = re.sub(r"\n{3,}", "\n\n", texto.strip())

    # PASSO 1 e 2 — parágrafos, depois frases se necessário
    unidades = []
    for paragrafo in texto.split("\n\n"):
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue
        if len(paragrafo) <= tamanho:
            unidades.append(paragrafo)
        else:
            unidades.extend(_quebrar_em_frases(paragrafo, tamanho))

    # PASSO 4 — agrupa as unidades até encher o chunk
    chunks, atual = [], ""

    for unidade in unidades:
        if not atual:
            atual = unidade
        elif len(atual) + 1 + len(unidade) <= tamanho:
            atual += "\n" + unidade
        else:
            chunks.append(atual)
            # PASSO 5 — o próximo chunk começa com a cauda do anterior.
            # Só aplicamos o overlap se ele couber: cauda + unidade pode
            # estourar o limite, e chunk maior que o teto quebra a premissa
            # de tamanho previsível (e pode passar do limite do embedding).
            cauda = _cauda_por_palavras(atual, overlap)
            if cauda and len(cauda) + 1 + len(unidade) <= tamanho:
                atual = f"{cauda}\n{unidade}"
            else:
                atual = unidade

    if atual:
        chunks.append(atual)

    return [c.strip() for c in chunks if c.strip()]


if __name__ == "__main__":
    exemplo = (
        "A Atenção Primária à Saúde é a porta de entrada do SUS. "
        "O e-SUS APS registra a produção assistencial das equipes.\n\n"
        "Os dados enviados ao Centralizador Nacional são submetidos a "
        "validação. Os campos podem ser consultados no LEDI de integração, "
        "disponível no link. O SISAB consolida as informações da atenção básica."
    ) * 3

    pedacos = quebrar_em_chunks(exemplo, tamanho=400, overlap=80)
    print(f"Gerou {len(pedacos)} chunks a partir de {len(exemplo)} caracteres.\n")

    for i, c in enumerate(pedacos, 1):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c[:120] + ("..." if len(c) > 120 else ""))
        print()

    # Verificação: nenhum chunk pode começar ou terminar no meio de palavra
    print("Nenhum chunk começa com palavra cortada:",
          all(c[0].isupper() or c[0].isdigit() or not c[0].isalpha() for c in pedacos))