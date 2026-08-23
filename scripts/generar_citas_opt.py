#!/usr/bin/env python3
import re, json, time, pathlib, requests
from urllib.parse import quote
TXT_PATH = pathlib.Path("articulos de entrega/articulo_preentrega.txt")
OUTPUT_TXT = pathlib.Path("articulos de entrega/articulo_preentrega_citado.txt")
LOG_PATH = pathlib.Path("articulos de entrega/citas_log_full.json")
REFS_PATH = pathlib.Path("articulos de entrega/referencias_APA.txt")
CACHE = {}

def split_sentences(text: str):
    protected = text.replace("et al.", "et al<dot>").replace("e.g.", "e_g<dot>").replace("i.e.", "i_e<dot>")
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', protected)
    sentences = []
    for p in parts:
        s = p.replace("et al<dot>", "et al.").replace("e_g<dot>", "e.g.").replace("i_e<dot>", "i.e.").strip()
        if len(s) > 20:
            sentences.append(s)
    final=[]
    for s in sentences:
        if "\n" in s and len(s)>500:
            final.extend([x.strip() for x in s.split("\n\n") if x.strip()])
        else:
            final.append(s)
    # filtrar a sustantivas: >60 chars y no solo encabezado
    final=[s for s in final if len(s.split())>7 and len(s)>60 and not s.startswith("Marco")]
    # deduplicar
    seen=[]; uniq=[]
    for s in final:
        if s not in seen:
            seen.append(s); uniq.append(s)
    return uniq

MAPPING = {
    "minería de datos educativos": ("educational data mining", ["education","learning"]),
    "sociología cuantitativa": ("computational social science education", ["education"]),
    "pérdidas de aprendizaje": ("learning loss pandemic", ["learning","pandemic"]),
    "modelos lineales": ("generalized linear model", ["model"]),
    "XGBoost": ("XGBoost gradient boosting", ["xgboost","gradient"]),
    "LightGBM": ("LightGBM", ["lightgbm"]),
    "datos tabulares": ("tabular data high cardinality", ["tabular"]),
    "caja negra": ("black box interpretability", ["black","interpret"]),
    "inteligencia artificial explicable": ("explainable AI SHAP", ["explain","shap"]),
    "SHAP": ("SHAP Shapley additive explanations", ["shap","shapley"]),
    "LIME": ("LIME explainable", ["lime"]),
    "territorio funcional": ("functional territory CEPAL", ["territory","functional"]),
    "clases latentes": ("latent class analysis", ["latent","class"]),
    "riesgo empírico": ("empirical risk minimization", ["empirical","risk"]),
    "Group DRO": ("group distributionally robust optimization Sagawa", ["robust","group"]),
    "DANN": ("domain adversarial neural network", ["domain","adversarial"]),
    "interseccionalidad": ("intersectionality algorithmic fairness", ["intersection","fairness"]),
    "heterogeneidad": ("heterogeneity educational data", ["heterogen"]),
    "domain shift": ("domain shift", ["domain","shift"]),
}

def extract_query(s):
    low=s.lower()
    for es,(en,dw) in MAPPING.items():
        if es.lower() in low:
            return en, dw
    stop=set("en de la el los las y a del se con para por una un al como es son sobre entre esta este hacia tras desde hasta".split())
    words=re.findall(r"[A-Za-z]{4,}", s)
    kws=[w for w in words if w.lower() not in stop][:6]
    return " ".join(kws) if kws else "educational data mining", ["education"]

def openalex_search(query, n=8):
    if query in CACHE:
        return CACHE[query]
    base="https://api.openalex.org/works"
    url=f"{base}?search={quote(query)}&per-page={n}&sort=relevance_score:desc&mailto=liturriago@demo.edu"
    for attempt in range(2):
        try:
            r=requests.get(url, timeout=15)
            if r.status_code==429:
                time.sleep(5)
                continue
            r.raise_for_status()
            res=r.json().get("results",[])
            CACHE[query]=res
            return res
        except: time.sleep(3)
    return []

def score_candidate(work, domain_words):
    title=(work.get("display_name") or "").lower()
    concepts=" ".join([c.get("display_name","").lower() for c in work.get("concepts",[])[:5]])
    text=title+" "+concepts
    hit=sum(1 for w in domain_words if w.lower() in text)
    dom=hit/max(len(domain_words),1)
    if any(x in text for x in ["tumor","cancer","neoplasia"]) and "education" in " ".join(domain_words):
        dom*=0.1
    rel=work.get("relevance_score",0) or 0
    import math; cite=math.log1p(work.get("cited_by_count",0) or 0)
    return rel*(0.7+dom)*(0.5+cite/10)

def format_apa(work):
    title=work.get("display_name","").rstrip(".")
    year=work.get("publication_year") or "s.f."
    authors=work.get("authorships",[])
    journal=(work.get("primary_location") or {}).get("source",{}) or {}
    jname=journal.get("display_name") or "Revista indexada"
    doi=work.get("doi") or ""
    if not authors: auth="Autor desconocido"
    elif len(authors)==1: auth=authors[0]["author"]["display_name"].split()[-1]
    elif len(authors)==2: auth=f"{authors[0]['author']['display_name'].split()[-1]} & {authors[1]['author']['display_name'].split()[-1]}"
    else: auth=f"{authors[0]['author']['display_name'].split()[-1]} et al."
    apa=f"{auth} ({year}). {title}. *{jname}*."
    if doi: apa+=f" {doi}"
    return apa, f"({auth}, {year})", doi

def main():
    text=TXT_PATH.read_text(encoding="utf-8")
    sents=split_sentences(text)
    print(f"Oraciones sustantivas: {len(sents)} (de 70 totales)")
    log=[]; citados=[]; refs={}
    for i,sent in enumerate(sents,1):
        print(f"[{i}/{len(sents)}] {sent[:90]}...")
        q,dw=extract_query(sent)
        works=openalex_search(q, n=8)
        best=None; best_sc=-1
        for w in works:
            sc=score_candidate(w,dw)
            if sc>best_sc: best_sc=sc; best=w
        if not best and works: best=works[0]
        if best:
            apa,cita,doi=format_apa(best)
            title=best.get("display_name","")[:60].encode("ascii","ignore").decode("ascii")
            print(f"  => {cita.encode('ascii','ignore').decode()} -> {title} | {doi}")
        else:
            apa,cita,doi="No encontrado","",""
        sent_c=sent.rstrip()
        if sent_c.endswith("."): sent_c=sent_c[:-1]+f" {cita}."
        else: sent_c+=f" {cita}"
        citados.append(sent_c)
        log.append({"idx":i,"oracion":sent,"query":q,"cita":cita,"apa":apa,"doi":doi,"titulo":best.get("display_name","") if best else ""})
        if cita not in refs and apa!="No encontrado": refs[cita]=apa
        time.sleep(0.85)
    out=""
    for idx,s in enumerate(citados):
        out+=s+" "
        if (idx+1)%4==0: out+="\n\n"
    refs_sorted=sorted(refs.items())
    refs_block="\n\nReferencias\n"+"="*40+"\n"+"\n\n".join([v for k,v in refs_sorted])
    OUTPUT_TXT.write_text(out.strip()+"\n"+refs_block, encoding="utf-8")
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    REFS_PATH.write_text("\n".join([v for k,v in refs_sorted]), encoding="utf-8")
    print(f"\nArchivo: {OUTPUT_TXT} ({len(citados)} oraciones, {len(refs_sorted)} refs)")

if __name__=="__main__": main()
