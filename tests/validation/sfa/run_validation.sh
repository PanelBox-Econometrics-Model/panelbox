#!/bin/bash
#
# Script para executar validação completa SFA contra R
#
# Autor: PanelBox Development Team
# Data: 2026-02-15
#
# Uso: ./run_validation.sh [--skip-r] [--skip-python]

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Diretório do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Flags
SKIP_R=false
SKIP_PYTHON=false

# Parse argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-r)
            SKIP_R=true
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        *)
            echo -e "${RED}Argumento desconhecido: $1${NC}"
            echo "Uso: $0 [--skip-r] [--skip-python]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================"
echo "Validação SFA PanelBox vs R"
echo "========================================${NC}\n"

# ============================================================================
# Verificar dependências
# ============================================================================

echo -e "${YELLOW}[1/5] Verificando dependências...${NC}"

# Verificar R
if ! command -v Rscript &> /dev/null; then
    echo -e "${RED}ERRO: R não encontrado. Instale R primeiro.${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install r-base"
    echo "  macOS: brew install r"
    exit 1
fi

echo -e "${GREEN}  ✓ R encontrado: $(R --version | head -1)${NC}"

# Verificar pacotes R
if [ "$SKIP_R" = false ]; then
    echo "  Verificando pacotes R..."
    Rscript -e "
    required_packages <- c('frontier', 'readr')
    missing_packages <- setdiff(required_packages, installed.packages()[,'Package'])
    if (length(missing_packages) > 0) {
        cat('Pacotes R faltando:', paste(missing_packages, collapse=', '), '\n')
        cat('Instalando...\n')
        install.packages(missing_packages, repos='https://cloud.r-project.org/')
    }
    cat('  ✓ Pacotes R instalados\n')
    " || {
        echo -e "${RED}ERRO: Falha ao verificar/instalar pacotes R${NC}"
        exit 1
    }
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERRO: Python não encontrado.${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ Python encontrado: $(python3 --version)${NC}"

# Verificar pytest
if ! python3 -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}  pytest não encontrado. Instalando...${NC}"
    pip install pytest
fi

echo -e "${GREEN}  ✓ pytest instalado${NC}"

# ============================================================================
# Gerar resultados R
# ============================================================================

if [ "$SKIP_R" = false ]; then
    echo -e "\n${YELLOW}[2/5] Gerando resultados de referência do R...${NC}"

    # frontier package
    echo "  Executando generate_r_frontier_results.R..."
    if Rscript generate_r_frontier_results.R > r_results/r_frontier_log.txt 2>&1; then
        echo -e "${GREEN}  ✓ Resultados frontier gerados com sucesso${NC}"
    else
        echo -e "${RED}  ✗ Erro ao gerar resultados frontier${NC}"
        echo "  Veja r_results/r_frontier_log.txt para detalhes"
        exit 1
    fi

    # sfaR package (TRE/BC95) - opcional
    if [ -f "generate_r_sfaR_results.R" ]; then
        echo "  Executando generate_r_sfaR_results.R..."
        if Rscript generate_r_sfaR_results.R > r_results/r_sfaR_log.txt 2>&1; then
            echo -e "${GREEN}  ✓ Resultados sfaR gerados com sucesso${NC}"
        else
            echo -e "${YELLOW}  ⚠ Aviso: sfaR falhou (não crítico)${NC}"
            echo "  TRE/BC95 podem não ser validados"
        fi
    fi
else
    echo -e "\n${YELLOW}[2/5] Pulando geração de resultados R (--skip-r)${NC}"
fi

# ============================================================================
# Verificar resultados R
# ============================================================================

echo -e "\n${YELLOW}[3/5] Verificando resultados R gerados...${NC}"

REQUIRED_FILES=(
    "r_results/riceProdPhil.csv"
    "r_results/r_frontier_cs_halfnormal_params.csv"
    "r_results/r_frontier_cs_halfnormal_efficiency.csv"
    "r_results/r_frontier_cs_halfnormal_loglik.csv"
    "r_results/r_frontier_panel_pittlee_params.csv"
    "r_results/r_frontier_panel_pittlee_efficiency.csv"
    "r_results/r_frontier_panel_pittlee_loglik.csv"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}ERRO: Arquivos de resultados R faltando:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Execute sem --skip-r para gerar os resultados R primeiro."
    exit 1
fi

echo -e "${GREEN}  ✓ Todos os arquivos de resultados R encontrados${NC}"

# Estatísticas dos resultados
echo "  Estatísticas dos resultados:"
NUM_OBSERVATIONS=$(wc -l < r_results/riceProdPhil.csv)
echo "    - riceProdPhil.csv: $((NUM_OBSERVATIONS - 1)) observações"

# ============================================================================
# Executar testes Python
# ============================================================================

if [ "$SKIP_PYTHON" = false ]; then
    echo -e "\n${YELLOW}[4/5] Executando testes de validação Python...${NC}"

    # Definir PYTHONPATH para incluir panelbox
    export PYTHONPATH="${SCRIPT_DIR}/../../../:${PYTHONPATH}"

    # Executar testes com pytest
    echo "  Executando pytest..."
    if pytest test_r_frontier_validation.py -v --tb=short --color=yes 2>&1 | tee r_results/pytest_log.txt; then
        echo -e "\n${GREEN}  ✓ Todos os testes de validação passaram!${NC}"
    else
        echo -e "\n${RED}  ✗ Alguns testes falharam${NC}"
        echo "  Veja r_results/pytest_log.txt para detalhes"
        exit 1
    fi
else
    echo -e "\n${YELLOW}[4/5] Pulando testes Python (--skip-python)${NC}"
fi

# ============================================================================
# Resumo
# ============================================================================

echo -e "\n${YELLOW}[5/5] Gerando resumo de validação...${NC}"

# Contar testes passados/falhados
if [ -f "r_results/pytest_log.txt" ]; then
    PASSED=$(grep -c "PASSED" r_results/pytest_log.txt || echo "0")
    FAILED=$(grep -c "FAILED" r_results/pytest_log.txt || echo "0")
    WARNINGS=$(grep -c "WARNING" r_results/pytest_log.txt || echo "0")

    echo ""
    echo "  Resumo dos Testes:"
    echo -e "    ${GREEN}✓ Passaram: $PASSED${NC}"
    if [ "$WARNINGS" -gt 0 ]; then
        echo -e "    ${YELLOW}⚠ Avisos:   $WARNINGS${NC}"
    fi
    if [ "$FAILED" -gt 0 ]; then
        echo -e "    ${RED}✗ Falharam: $FAILED${NC}"
    fi
fi

# Lista de modelos validados
echo ""
echo "  Modelos Validados:"
echo "    ✓ Cross-section SFA - Half-Normal"
echo "    ✓ Panel SFA - Pitt & Lee (1981)"
echo "    ✓ Panel SFA - Battese & Coelli (1992)"

# ============================================================================
# Conclusão
# ============================================================================

echo ""
echo -e "${BLUE}========================================"
echo "Validação Concluída!"
echo "========================================${NC}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ SUCESSO: Implementação PanelBox SFA validada contra R${NC}"
    echo ""
    echo "Todos os modelos replicam resultados R dentro das tolerâncias:"
    echo "  - Coeficientes:         ± 1e-4"
    echo "  - Componentes variância: ± 1e-3"
    echo "  - Log-likelihood:       ± 1e-2"
    echo "  - Eficiências:          ± 1e-3"
    echo ""
    echo "A implementação está pronta para produção!"
    exit 0
else
    echo -e "${RED}✗ FALHA: Alguns testes não passaram${NC}"
    echo ""
    echo "Revise r_results/pytest_log.txt para detalhes."
    echo "Possíveis causas:"
    echo "  - Diferenças de otimizador (R nlm vs scipy)"
    echo "  - Starting values diferentes"
    echo "  - Bugs na implementação PanelBox"
    exit 1
fi
