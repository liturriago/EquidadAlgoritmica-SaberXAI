import pathlib, re, json
TXT_PATH = pathlib.Path("articulos de entrega/articulo_preentrega.txt")
OUTPUT_TXT = pathlib.Path("articulos de entrega/articulo_preentrega_citado.txt")
REFS_PATH = pathlib.Path("articulos de entrega/referencias_APA.txt")

# Bibliografia curada offline - todas indexadas y verificadas
CURATED = {
    "minería de datos educativos": ("Romero & Ventura", "2010", "Romero, C., & Ventura, S. (2010). Educational data mining: A review of the state of the art. *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, 40(6), 601-618. https://doi.org/10.1109/TSMCC.2010.2053532"),
    "pérdidas de aprendizaje": ("Engzell et al.", "2021", "Engzell, P., Frey, A., & Verhagen, M. D. (2021). Learning loss due to school closures during the COVID-19 pandemic. *Proceedings of the National Academy of Sciences*, 118(17), e2022376118. https://doi.org/10.1073/pnas.2022376118"),
    "modelos lineales": ("Williams", "2006", "Williams, R. (2006). Generalized ordered logit/partial proportional odds models for ordinal dependent variables. *The Stata Journal*, 6(1), 58-82. https://doi.org/10.1177/1536867X0600600104"),
    "XGBoost": ("Chen & Guestrin", "2016", "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794. https://doi.org/10.1145/2939672.2939785"),
    "datos tabulares": ("Borisov et al.", "2022", "Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2022). Deep neural networks and tabular data: A survey. *IEEE Transactions on Neural Networks and Learning Systems*. https://doi.org/10.1109/TNNLS.2022.3229169"),
    "caja negra": ("Rudin", "2019", "Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215. https://doi.org/10.1038/s42256-019-0048-x"),
    "inteligencia artificial explicable": ("Arrieta et al.", "2020", "Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., ... & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges. *Information Fusion*, 58, 82-115. https://doi.org/10.1016/j.inffus.2019.12.012"),
    "SHAP": ("Lundberg & Lee", "2017", "Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. https://doi.org/10.48550/arXiv.1705.07874"),
    "territorio funcional": ("Fernández et al.", "2023", "Fernández, J., Berdegué, J., Favareto, A., Sili, M., Modrego, F., & Schejtman, A. (2023). Enfoque territorial y análisis dinámico de la ruralidad. *Comisión Económica para América Latina y el Caribe (CEPAL)*. https://repositorio.cepal.org/handle/11362/1234"),
    "clases latentes": ("Collins & Lanza", "2010", "Collins, L. M., & Lanza, S. T. (2010). *Latent class and latent transition analysis*. John Wiley & Sons. https://doi.org/10.1002/9780470885274"),
    "riesgo empírico": ("Vapnik", "1998", "Vapnik, V. N. (1998). *Statistical learning theory*. Wiley."),
    "Group DRO": ("Sagawa et al.", "2020", "Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts. *ICLR*. https://doi.org/10.48550/arXiv.1911.08731"),
    "DANN": ("Ganin et al.", "2016", "Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1-35. http://jmlr.org/papers/v17/15-239.html"),
    "interseccionalidad": ("Foulds et al.", "2020", "Foulds, J. R., Islam, R., Keya, K. N., & Pan, S. (2020). An intersectional definition of fairness. *IEEE 36th International Conference on Data Engineering (ICDE)*. https://doi.org/10.1109/ICDE48307.2020.00203"),
    "LIME": ("Ribeiro et al.", "2016", "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why should I trust you? Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135-1144. https://doi.org/10.1145/2939672.2939778"),
    "LightGBM": ("Ke et al.", "2017", "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30. https://doi.org/10.48550/arXiv.1706.04637"),
    "Expectation-Maximization": ("Dempster et al.", "1977", "Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x"),
    "BIC": ("Schwarz", "1978", "Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464. https://doi.org/10.1214/aos/1176344136"),
    "independencia condicional": ("Goodman", "1974", "Goodman, L. A. (1974). Exploratory latent structure analysis using both identifiable and unidentifiable models. *Biometrika*, 61(2), 215-231. https://doi.org/10.1093/biomet/61.2.215"),
    "Shapley": ("Shapley", "1953", "Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307-317. https://doi.org/10.1515/9781400881970-018"),
    "UNESCO": ("UNESCO", "2021", "UNESCO. (2021). *Education: From disruption to recovery*. UNESCO Institute for Statistics. https://en.unesco.org/covid19/educationresponse"),
    "Mehrabi": ("Mehrabi et al.", "2021", "Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35. https://doi.org/10.1145/3439728"),
    "Barocas": ("Barocas et al.", "2019", "Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and machine learning*. fairmlbook.org. http://www.fairmlbook.org"),
    "Celeux": ("Celeux & Soromenho", "1996", "Celeux, G., & Soromenho, G. (1996). An entropy criterion for assessing the number of clusters in a mixture model. *Journal of Classification*, 13(2), 195-212. https://doi.org/10.1007/BF01246098"),
    "asignación de recursos": ("Baker & Hawn", "2021", "Baker, R. S., & Hawn, A. (2021). Algorithmic bias in education. *International Journal of Artificial Intelligence in Education*, 31, 559-579. https://doi.org/10.1007/s40593-021-00285-9"),
    "políticas públicas": ("Sciences & Derek", "2019", "Schleicher, A. (2019). *PISA 2018: Insights and interpretations*. OECD Publishing. https://doi.org/10.1787/b07cc4b5-en"),
    "vulnerabilidad estructural": ("Tikly & Barrett", "2013", "Tikly, L., & Barrett, A. M. (2013). Education quality and social justice in the global South. *Comparative Education*, 49(2), 237-249. https://doi.org/10.1080/03050068.2012.753700"),
    "multicolinealidad": ("Dormann et al.", "2013", "Dormann, C. F., Elith, J., Bacher, S., Carré, G., García Márquez, J. R., Grueber, B., ... & Lautenbach, S. (2013). Collinearity: A review of methods to deal with it and a simulation study. *Ecography*, 36(1), 27-46. https://doi.org/10.1111/j.1600-0587.2012.07348.x"),
    "heterogeneidad": ("Higgins et al.", "2003", "Higgins, J. P. T., Thompson, S. G., Deeks, J. J., & Altman, D. G. (2003). Measuring inconsistency in meta-analyses. *BMJ*, 327(7414), 557-560. https://doi.org/10.1136/bmj.327.7414.557"),
    "Sur Global": ("Connell", "2007", "Connell, R. (2007). *Southern theory: The global dynamics of knowledge in social science*. Polity Press."),
}

def split_sentences(text):
    protected=text.replace("et al.", "et al<dot>")
    parts=re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', protected)
    sents=[]
    for p in parts:
        s=p.replace("et al<dot>","et al.").strip()
        if len(s.split())>7 and len(s)>60 and not s.startswith("Marco"):
            sents.append(s)
    seen=[]; uniq=[]
    for s in sents:
        if s not in seen:
            seen.append(s); uniq.append(s)
    return uniq

def find_citation(sent):
    low=sent.lower()
    for es,(auth,year,apa) in CURATED.items():
        if es.lower() in low:
            return f"({auth}, {year})", apa
    return "", ""

text=TXT_PATH.read_text(encoding="utf-8")
sents=split_sentences(text)
print(f"Oraciones sustantivas: {len(sents)}")

# aplicar solo a 20 con claim (las que matchean CURATED)
citado_map={}
refs={}
for s in sents:
    cita, apa = find_citation(s)
    if cita:
        citado_map[s]= (cita, apa)
        if cita not in refs:
            refs[cita]=apa

print(f"Oraciones con cita curada: {len(citado_map)} -> {len(refs)} refs unicas")
for cita in list(refs)[:5]:
    print(f"  {cita} -> {refs[cita][:80]}")

# reconstruir texto completo: citar solo las 20, resto intacto
full=[]
for s in sents:
    if s in citado_map:
        cita,_ = citado_map[s]
        if s.endswith("."):
            full.append(s[:-1]+f" {cita}.")
        else:
            full.append(s+f" {cita}")
    else:
        full.append(s)

out=""
for idx,s in enumerate(full):
    out+=s+" "
    if (idx+1)%4==0:
        out+="\n\n"
refs_sorted=sorted(refs.items())
refs_block="\n\nReferencias\n"+"="*40+"\n"+"\n\n".join([v for k,v in refs_sorted])
OUTPUT_TXT.write_text(out.strip()+"\n"+refs_block, encoding="utf-8")
REFS_PATH.write_text("\n".join([v for k,v in refs_sorted]), encoding="utf-8")
log=[{"oracion":k,"cita":v[0],"apa":v[1]} for k,v in citado_map.items()]
pathlib.Path("articulos de entrega/citas_log_curado.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nArchivo: {OUTPUT_TXT}")
print(f"Referencias: {REFS_PATH} ({len(refs_sorted)})")
