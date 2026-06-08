import streamlit as st
# aqui você importa a sua função real depois que ela estiver pronta
# from src.motor_inferencia import rodar_motor_ag

st.set_page_config(page_title="Motor de Recomendação", layout="centered")

st.title("Simulador de Recomendação Financeira")
st.write("Insira os parâmetros abaixo para executar a simulação.")

valor_emprestimo = st.number_input("Valor do empréstimo desejado", min_value=0.0, value=50000.00, step=1000.00)
saldo_fgts_usuario = st.number_input("Saldo do FGTS", min_value=0.0, value=50000.00, step=1000.00)
salario_liquido_usuario = st.number_input("Salário líquido", min_value=0.0, value=7200.00, step=500.00)

if st.button("Executar Motor"):
    with st.spinner("Processando WLS e Algoritmo Genético..."):
        
        # mockup temporário para testar se o Streamlit subiu corretamente
        melhor_prazo = 36
        melhor_taxa = 1.85
        parcela = 1950.00
        
        st.subheader("Resultados da Otimização")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Prazo Ideal", f"{melhor_prazo} meses")
        col2.metric("Taxa Estimada", f"{melhor_taxa}% a.m.")
        col3.metric("Parcela", f"R$ {parcela:.2f}")
        
        st.success("Recomendação gerada com sucesso pela fronteira de Pareto.")
