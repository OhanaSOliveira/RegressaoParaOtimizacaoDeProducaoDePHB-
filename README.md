# **Análise de Produção de PHB por Cepas**  

## **1. Contexto**  
O polihidroxibutirato (PHB) é um biopolímero biodegradável produzido naturalmente por bactérias como reserva de carbono e energia. Por ser uma alternativa sustentável aos plásticos derivados de petróleo, há um grande interesse em otimizar sua produção industrial. No entanto, a síntese natural de PHB apresenta limitações, como desvio de carbono para subprodutos e dependência de condições ideais de fermentação.  

Este projeto tem como objetivo **analisar e identificar condições de fermentação que maximizem a produção de PHB**, comparando diferentes cepas e condições (pH, temperatura e oxigenação) usando dados simulados baseados na literatura científica.

---

## **2. Conjunto de Dados**  
Os dados simulam medições de **6 cepas bacterianas** (WT, C1, C2, C3, C4 e C5) ao longo de 72 horas, em intervalos de 2 horas, considerando as seguintes variáveis:  
- **Tempo_h:** tempo do cultivo (h).  
- **Glicose_gL:** concentração de glicose (g/L).  
- **PHB_gL:** concentração acumulada de PHB (g/L).  
- **OD600:** densidade óptica, que indica biomassa celular.  
- **pH, Temperatura_C, Oxigenacao_%:** condições de fermentação.  

---

## **3. Principais Funções e Métricas**  

### **3.1 Função T90**  
Calcula o tempo mínimo para uma cepa atingir 90% do valor máximo de PHB, permitindo avaliar a rapidez do processo.  

### **3.2 Função de Métricas**  
Agrupa os dados por cepa e calcula:  
- **PHB_final_gL:** produção final de PHB.  
- **Glicose_consumida_gL:** diferença entre glicose inicial e final.  
- **Rendimento_PHB_por_gGlicose:** eficiência de conversão (g PHB/g glicose).  
- **OD_final:** biomassa final.  
- **T90_h:** tempo para atingir 90% do PHB máximo.  
- **Produtividade_media_gL_h:** média de produção por hora.  

As cepas são ranqueadas de acordo com o PHB final.  


## **4. Análises Estatísticas**  

### **4.1 Correlação**  
O heatmap de correlação mostrou que poucas variáveis têm relação linear forte com o PHB final, indicando a natureza multifatorial do processo. Oxigenação e rendimento de glicose apresentam impacto moderado, enquanto pH e temperatura exibem baixa correlação.  

### **4.2 Regressão Linear**  
A regressão múltipla apresentou a seguinte equação:  
PHB_gL = 12.62 - 0.17 * pH - 0.02 * Temperatura_C + 0.03 * Oxigenacao_% - 0.24 * Glicose_gL + 1.84 * OD600

- **Oxigenação** foi o fator mais positivo.  
- **Temperatura** teve impacto negativo leve.  
- **pH** mostrou efeito intermediário, com melhor produção em valores neutros a levemente alcalinos.  

O modelo prevê bem valores baixos e médios de PHB, mas subestima valores altos, sugerindo a necessidade de modelos não lineares.  

---

## **5. Resultados e Conclusões**  
- **Condições ideais:**  
  - pH entre **7.0 e 7.5**.  
  - Temperatura entre **30°C e 35°C**, com tolerância até 40°C.  
  - Oxigenação constante em **60%**.  
  - Glicose entre **48 e 50 g/L**.  
- **Cepa mais produtiva:** **C5**, com produtividade média de **0,0759 g PHB/h**.  
- A análise confirma que a otimização simultânea de parâmetros é essencial e que métodos de modelagem mais complexos poderiam melhorar a previsão da produção.  

