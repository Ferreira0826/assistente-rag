#!/usr/bin/env bash
# ==============================================================================
# auditoria.sh — Varredura de segurança dos seus repositórios
# Rode DENTRO da pasta de cada projeto:  bash auditoria.sh
# Ou aponte para um repo:                bash auditoria.sh /caminho/do/repo
# ==============================================================================
set -uo pipefail

REPO="${1:-.}"
cd "$REPO" || { echo "Pasta não encontrada: $REPO"; exit 1; }

echo "════════════════════════════════════════════════════════════"
echo "  AUDITORIA DE SEGURANÇA — $(basename "$(pwd)")"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════"

# ------------------------------------------------------------------
echo -e "\n▶ 1. Arquivos sensíveis versionados (NÃO deveriam estar no Git)"
echo "------------------------------------------------------------"
git ls-files 2>/dev/null | grep -iE '\.env($|\.)|\.env\.(local|prod|production)|credentials|secret|\.pem$|\.key$|\.pfx$|id_rsa|\.sqlite3?$|\.pgpass|serviceAccount.*\.json|\.npmrc|\.pypirc' \
  && echo "  ⚠  ACIMA: arquivos que precisam sair do versionamento" \
  || echo "  ✓ Nenhum arquivo sensível óbvio versionado"

# ------------------------------------------------------------------
echo -e "\n▶ 2. node_modules / venv / dist versionados (peso + risco)"
echo "------------------------------------------------------------"
git ls-files 2>/dev/null | grep -iE '(^|/)(node_modules|\.venv|venv|__pycache__|dist|build)/' | head -5 \
  && echo "  ⚠  ACIMA: diretórios que deveriam estar no .gitignore" \
  || echo "  ✓ Limpo"

# ------------------------------------------------------------------
echo -e "\n▶ 3. Segredos no CÓDIGO (padrões de chaves e tokens)"
echo "------------------------------------------------------------"
# Padrões comuns: OpenAI, AWS, Google, Supabase, senhas hardcoded, JWT, connection strings
grep -rInE \
  'sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|ya29\.[0-9A-Za-z_\-]+|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|(password|senha|passwd|secret|api[_-]?key|token)\s*[=:]\s*["'"'"'][^"'"'"']{6,}|postgres(ql)?://[^:\s]+:[^@\s]+@|mongodb(\+srv)?://[^:\s]+:[^@\s]+@|https://[a-z0-9]+\.supabase\.co' \
  --include='*.py' --include='*.js' --include='*.jsx' --include='*.ts' --include='*.tsx' \
  --include='*.env*' --include='*.json' --include='*.yml' --include='*.yaml' \
  --include='*.md' --include='*.txt' --include='*.ipynb' \
  . 2>/dev/null | grep -vE 'node_modules|\.git/|package-lock' | head -40 \
  && echo "  ⚠  REVISE cada linha acima" \
  || echo "  ✓ Nenhum padrão de segredo hardcoded encontrado"

# ------------------------------------------------------------------
echo -e "\n▶ 4. Segredos no HISTÓRICO do Git (mesmo que já removidos dos arquivos)"
echo "------------------------------------------------------------"
echo "  Procurando em TODOS os commits..."
git log -p --all 2>/dev/null | grep -InE \
  'sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|postgres(ql)?://[^:\s]+:[^@\s]+@|\.supabase\.co|(password|senha|secret|api[_-]?key)\s*[=:]\s*["'"'"'][^"'"'"']{6,}' \
  | head -20 \
  && echo "  ⚠  ATENÇÃO: segredos no histórico exigem ROTAÇÃO da credencial + limpeza do histórico" \
  || echo "  ✓ Nenhum segredo aparente no histórico"

# ------------------------------------------------------------------
echo -e "\n▶ 5. E-mails / dados pessoais em código e logs"
echo "------------------------------------------------------------"
grep -rInE '[a-zA-Z0-9._%+-]+@(gmail|hotmail|outlook|yahoo|saude\.df\.gov)\.[a-z.]+' \
  --include='*.py' --include='*.js' --include='*.md' --include='*.txt' --include='*.log' --include='*.ipynb' \
  . 2>/dev/null | grep -vE 'node_modules|\.git/' | head -15 \
  && echo "  ⚠  REVISE: e-mails reais expostos (LGPD)" \
  || echo "  ✓ Nenhum e-mail pessoal óbvio"

# ------------------------------------------------------------------
echo -e "\n▶ 6. Dependências vulneráveis"
echo "------------------------------------------------------------"
if [ -f package.json ]; then
  echo "  Node detectado → rodando npm audit..."
  npm audit 2>/dev/null | tail -8
elif [ -f requirements.txt ]; then
  echo "  Python detectado → instale e rode: pip install pip-audit && pip-audit -r requirements.txt"
else
  echo "  (sem package.json/requirements.txt neste repo)"
fi

# ------------------------------------------------------------------
echo -e "\n▶ 7. .gitignore presente e cobrindo o básico?"
echo "------------------------------------------------------------"
if [ -f .gitignore ]; then
  for p in ".env" "node_modules" "venv" "__pycache__" "*.sqlite3"; do
    grep -q -- "$p" .gitignore && echo "  ✓ ignora $p" || echo "  ⚠  NÃO ignora $p"
  done
else
  echo "  ⚠  SEM .gitignore — crie um urgente"
fi

echo -e "\n════════════════════════════════════════════════════════════"
echo "  Fim. Reveja cada ⚠ acima. Nada aqui sai da sua máquina."
echo "════════════════════════════════════════════════════════════"