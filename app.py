import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from math import floor
from sklearn.linear_model import LinearRegression

# =====================================================================
# 1. MOTOR DE DADOS E TREINAMENTO (COM CACHE DE MEMÓRIA)
# =====================================================================
@st.cache_resource(ttl=86400, show_spinner="Construindo inteligência do motor (conectando ao BCB)...")
def preparar_motor():
    # Coleta de Dados
    def coletar_dados_sgs(codigo_bcb, data_inicio, data_fim):
        url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_bcb}/dados?formato=json&dataInicial={data_inicio}&dataFinal={data_fim}'
        response = requests.get(url)
        df = pd.DataFrame(response.json())
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = df['valor'].astype(float)
        df.set_index('data', inplace=True)
        return df

    series_sgs = {
        'selic_mensal': 4390, 'inpc_mensal': 188, 'taxa_desocupação': 24369,
        'taxa_juros_mercado': 20744, 'prazo_medio_mercado': 20881,
        'taxa_consignado_inss': 25468 
    }

    dfs = []
    for nome, codigo in series_sgs.items():
        df_temp = coletar_dados_sgs(codigo, '01/01/2007', '31/05/2026')
        df_temp.rename(columns={'valor': nome}, inplace=True)
        dfs.append(df_temp)

    df_base = pd.concat(dfs, axis=1).ffill().dropna()

    # Engenharia de Atributos
    df_base['meses_desde_inicio'] = np.arange(len(df_base))
    df_base['peso_temporal'] = np.exp(0.15 * (df_base['meses_desde_inicio'] - df_base['meses_desde_inicio'].max()))
    df_base['fim_intermediacao_rh'] = np.where(df_base.index >= '2025-04-01', 1, 0)

    # Treinamento WLS
    features = ['selic_mensal', 'inpc_mensal', 'taxa_desocupação', 'prazo_medio_mercado', 'fim_intermediacao_rh']
    X = df_base[features]
    y = df_base['taxa_juros_mercado']
    
    modelo = LinearRegression()
    modelo.fit(X, y, sample_weight=df_base['peso_temporal'])
    
    estado_atual = df_base.iloc[-1]
    prazo_maximo = int(df_base.loc['2025-04-01':]['prazo_medio_mercado'].max())
    
    return modelo, estado_atual, prazo_maximo

# Executa o cache no carregamento
modelo_base, estado, prazo_maximo_mercado = preparar_motor()

# =====================================================================
# 2. LÓGICA DO ALGORITMO GENÉTICO
# =====================================================================
def avaliar_oferta_mercado(prazo_meses, garantia_fgts_ativa, saldo_fgts, volume, salario_liquido):
    X_simulacao = pd.DataFrame([{
        'selic_mensal': estado['selic_mensal'],
        'inpc_mensal': estado['inpc_mensal'],
        'taxa_desocupação': estado['taxa_desocupação'],
        'prazo_medio_mercado': prazo_meses,
        'fim_intermediacao_rh': estado['fim_intermediacao_rh']
    }])

    taxa_anual_mercado = max(modelo_base.predict(X_simulacao)[0], 0.8)

    if garantia_fgts_ativa == 1 and saldo_fgts > 0:
        taxa_anual_fgts = min(taxa_anual_mercado * 0.55, ((1 + estado['taxa_consignado_inss']/100)**12 - 1)*100)
        if saldo_fgts >= volume:
            taxa_anual_final = taxa_anual_fgts
        else:
            taxa_anual_final = ((saldo_fgts * taxa_anual_fgts) + ((volume - saldo_fgts) * taxa_anual_mercado)) / volume
    else:
        taxa_anual_final = taxa_anual_mercado

    taxa_mensal = ((1 + taxa_anual_final / 100) ** (1 / 12)) - 1
    pmt = volume * (taxa_mensal * (1 + taxa_mensal)**prazo_meses) / ((1 + taxa_mensal)**prazo_meses - 1)
    
    margem = salario_liquido * 0.35
    penalidade = max(0, (pmt - margem)) * 10000
    
    return taxa_mensal, pmt, ((pmt * prazo_meses) - volume) + penalidade

def otimizar_contrato_ag(usa_fgts, saldo_fgts, volume, salario_liquido):
    populacao = np.clip(np.random.normal(int(estado['prazo_medio_mercado']), 12, 15), 6, 84).astype(int)

    for _ in range(30): # 30 Gerações
        custos = [avaliar_oferta_mercado(p, usa_fgts, saldo_fgts, volume, salario_liquido)[2] for p in populacao]
        nova_pop = [int(populacao[i]) for i in np.argsort(custos)[:2]]
        
        while len(nova_pop) < 15:
            p1 = populacao[random.choice(range(15))]
            p2 = populacao[random.choice(range(15))]
            filho = int((p1 + p2) / 2)
            if random.random() < 0.20: filho += random.choice([-6, -3, 3, 6])
            nova_pop.append(max(6, min(84, filho)))
            
        populacao = np.array(nova_pop)

    custos_finais = [avaliar_oferta_mercado(p, usa_fgts, saldo_fgts, volume, salario_liquido)[2] for p in populacao]
    melhor_prazo = int(populacao[np.argmin(custos_finais)])
    taxa_m, pmt_f, _ = avaliar_oferta_mercado(melhor_prazo, usa_fgts, saldo_fgts, volume, salario_liquido)
    
    return melhor_prazo, taxa_m * 100, pmt_f

# =====================================================================
# 3. INTERFACE DE USUÁRIO (FRONTEND STREAMLIT)
# =====================================================================
st.set_page_config(page_title="Motor WLS/AG", layout="centered")
st.title("Simulador de Recomendação Financeira")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1: valor_emprestimo = st.number_input("Valor Solicitado (R$)", value=50000.0, step=1000.0)
with col2: saldo_fgts_usuario = st.number_input("Saldo FGTS (R$)", value=50000.0, step=1000.0)
with col3: salario_liquido_usuario = st.number_input("Salário Líquido (R$)", value=7200.0, step=500.0)

if st.button("Executar Simulação Preditiva", type="primary", use_container_width=True):
    
    margem_maxima = salario_liquido_usuario * 0.35
    taxa_ref = estado['taxa_consignado_inss'] / 100
    teto_credito = (margem_maxima * ((1 + taxa_ref)**prazo_maximo_mercado - 1)) / (taxa_ref * (1 + taxa_ref)**prazo_maximo_mercado)
    
    if valor_emprestimo > teto_credito:
        st.error(f"⚠️ Operação Recusada! O valor solicitado ultrapassa a margem consignável legal. Teto máximo permitido: R$ {teto_credito:,.2f}")
    else:
        with st.spinner("Processando Algoritmo Genético..."):
            p_sem, t_sem, pmt_sem = otimizar_contrato_ag(0, 0, valor_emprestimo, salario_liquido_usuario)
            p_com, t_com, pmt_com = otimizar_contrato_ag(1, saldo_fgts_usuario, valor_emprestimo, salario_liquido_usuario)
            
            st.subheader("🔴 Cenário A: Mercado Livre (Sem Garantia)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prazo Ideal", f"{p_sem} meses")
            c2.metric("Taxa Estimada", f"{t_sem:.2f}% a.m.")
            c3.metric("Parcela", f"R$ {pmt_sem:,.2f}")
            c4.metric("Juros Totais", f"R$ {(pmt_sem*p_sem)-valor_emprestimo:,.2f}")
            
            st.markdown("---")
            
            if saldo_fgts_usuario >= valor_emprestimo:
                st.subheader("🟢 Cenário B: Garantia Integral do FGTS")
            elif saldo_fgts_usuario > 0:
                st.subheader(f"🟡 Cenário B: Garantia Híbrida (FGTS R$ {saldo_fgts_usuario:,.2f})")
                
            if saldo_fgts_usuario > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prazo Ideal", f"{p_com} meses", f"{p_com - p_sem} meses", delta_color="inverse")
                c2.metric("Taxa Estimada", f"{t_com:.2f}% a.m.", f"{t_com - t_sem:.2f}%", delta_color="inverse")
                c3.metric("Parcela", f"R$ {pmt_com:,.2f}", f"R$ {pmt_com - pmt_sem:,.2f}", delta_color="inverse")
                
                economia = ((pmt_sem*p_sem)-valor_emprestimo) - ((pmt_com*p_com)-valor_emprestimo)
                c4.metric("Economia Juros", f"R$ {economia:,.2f}", "Dinheiro salvo!")
                
                st.success("Recomendação gerada pela Fronteira de Eficiência de Pareto.")
            else:
                st.info("Cenário B Indisponível: Saldo do FGTS zerado.")
