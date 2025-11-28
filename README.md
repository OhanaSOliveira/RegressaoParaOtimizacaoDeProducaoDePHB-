# **PHB Production Analysis by Strains**  

## **1. Context**  
Polyhydroxybutyrate (PHB) is a biodegradable biopolymer naturally produced by bacteria as a carbon and energy reserve. Because it is a sustainable alternative to petroleum-based plastics, there is strong interest in optimizing its industrial production. However, natural PHB synthesis has limitations, such as carbon diversion to by-products and dependence on ideal fermentation conditions.  

This project aims to **analyze and identify fermentation conditions that maximize PHB production**, comparing different strains and conditions (pH, temperature, and oxygenation) using simulated data based on the scientific literature.

---

## **2. Dataset**  
The data simulate measurements of **6 bacterial strains** (WT, C1, C2, C3, C4, and C5) over 72 hours, in 2-hour intervals, considering the following variables:  
- **Tempo_h:** cultivation time (h).  
- **Glicose_gL:** glucose concentration (g/L).  
- **PHB_gL:** accumulated PHB concentration (g/L).  
- **OD600:** optical density, indicating cell biomass.  
- **pH, Temperatura_C, Oxigenacao_%:** fermentation conditions.  

---

## **3. Main Functions and Metrics**  

### **3.1 T90 Function**  
Calculates the minimum time required for a strain to reach 90% of the maximum PHB value, allowing assessment of process speed.  

### **3.2 Metrics Function**  
Groups data by strain and calculates:  
- **PHB_final_gL:** final PHB production.  
- **Glicose_consumida_gL:** difference between initial and final glucose.  
- **Rendimento_PHB_por_gGlicose:** conversion efficiency (g PHB/g glucose).  
- **OD_final:** final biomass.  
- **T90_h:** time to reach 90% of maximum PHB.  
- **Produtividade_media_gL_h:** average hourly productivity.  

Strains are ranked according to final PHB production.  

---

## **4. Statistical Analyses**  

### **4.1 Correlation**  
The correlation heatmap showed that few variables have a strong linear relationship with final PHB, reinforcing the multifactorial nature of the process. Oxygenation and glucose yield present moderate impact, while pH and temperature show low correlation.  

### **4.2 Linear Regression**  
The multiple regression model produced the following equation:  
PHB_gL = 12.62 - 0.17 * pH - 0.02 * Temperatura_C + 0.03 * Oxigenacao_% - 0.24 * Glicose_gL + 1.84 * OD600

- **Oxygenation** was the most positive factor.  
- **Temperature** had a slight negative impact.  
- **pH** showed an intermediate effect, with better production in neutral to slightly alkaline values.  

The model predicts low and medium PHB values well but underestimates high values, suggesting the need for nonlinear models.  

---

## **5. Results and Conclusions**  
- **Ideal conditions:**  
  - pH between **7.0 and 7.5**.  
  - Temperature between **30°C and 35°C**, with tolerance up to 40°C.  
  - Constant oxygenation at **60%**.  
  - Glucose between **48 and 50 g/L**.  
- **Most productive strain:** **C5**, with an average productivity of **0.0759 g PHB/h**.  
- The analysis confirms that simultaneous optimization of parameters is essential and that more complex modeling methods could improve production prediction.  


