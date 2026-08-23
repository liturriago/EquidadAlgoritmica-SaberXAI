import re, json, time, pathlib, requests
from urllib.parse import quote
TXT_PATH = pathlib.Path("articulos de entrega/articulo_preentrega.txt")
OUTPUT_TXT = pathlib.Path("articulos de entrega/articulo_preentrega_citado.txt")
LOG_PATH = pathlib.Path("articulos de entrega/citas_log_full.json")
REFS_PATH = pathlib.Path("articulos de entrega/referencias_APA.txt")

MAPPING = {
    "minería de datos educativos": ("educational data mining", ["education"]),
    "sociología cuantitativa": ("computational social science education", ["education"]),
    "pérdidas de aprendizaje": ("learning loss pandemic", ["learning","pandemic"]),
    "modelos lineales": ("generalized linear model", ["model"]),
    "XGBoost": ("XGBoost gradient boosting", ["xgboost"]),
    "LightGBM": ("LightGBM", ["lightgbm"]),
    "datos tabulares": ("tabular data high cardinality", ["tabular"]),
    "caja negra": ("black box interpretability", ["black"]),
    "inteligencia artificial explicable": ("explainable AI SHAP", ["explain"]),
    "SHAP": ("SHAP Shapley", ["shap"]),
    "LIME": ("LIME explainable", ["lime"]),
    "territorio funcional": ("functional territory CEPAL", ["territory"]),
    "clases latentes": ("latent class analysis", ["latent"]),
    "riesgo empírico": ("empirical risk minimization", ["empirical"]),
    "Group DRO": ("group distributionally robust optimization", ["robust"]),
    "DANN": ("domain adversarial neural network", ["domain"]),
    "interseccionalidad": ("intersectionality algorithmic fairness", ["intersection"]),
}

def split_sentences(text):
    protected=text.replace("et al.", "et al<dot>")
    parts=re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', protected)
    sents=[]
    for p in parts:
        s=p.replace("et al<dot>","et al.").strip()
        if len(s.split())>7 and len(s)>60 and not s.startswith("Marco"):
            sents.append(s)
    # dedup
    seen=[]; uniq=[]
    for s in sents:
        if s not in seen:
            seen.append(s); uniq.append(s)
    return uniq

def extract_query(s):
    low=s.lower()
    for es,(en,dw) in MAPPING.items():
        if es.lower() in low:
            return en,dw,es
    return None,None,None

def openalex_search(q, n=6):
    url=f"https://api.openalex.org/works?search={quote(q)}&per-page={n}&sort=relevance_score:desc&mailto=liturriago@demo.edu"
    try:
        r=requests.get(url, timeout=15)
        if r.status_code==429:
            time.sleep(5)
            r=requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json().get("results",[])
    except: return []

def score_candidate(work, dw):
    title=(work.get("display_name") or "").lower()
    concepts=" ".join([c.get("display_name","").lower() for c in work.get("concepts",[])[:5]])
    text=title+" "+concepts
    hit=sum(1 for w in dw if w.lower() in text)
    dom=hit/max(len(dw),1)
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

text=TXT_PATH.read_text(encoding="utf-8")
sents=split_sentences(text)
# filtrar solo las que tienen mapeo tecnico
filtered=[]
for s in sents:
    q,dw,es=extract_query(s)
    if q: filtered.append((s,q,dw,es))
print(f"Total sustantivas: {len(sents)} -> Con claim citable: {len(filtered)}")
# tomar 20 diversas
import random; 
# priorizar una por cada termino unico
seen_terms=set(); diverse=[]
for s,q,dw,es in filtered:
    if es not in seen_terms:
        diverse.append((s,q,dw,es))
        seen_terms.add(es)
# completar hasta 20 con las restantes
for item in filtered:
    if item not in diverse and len(diverse)<20:
        diverse.append(item)
diverse=diverse[:20]
print(f"Seleccionadas para esta corrida rapida: {len(diverse)}")
for i,(s,q,dw,es) in enumerate(diverse,1):
    print(f"{i}. [{es}] {q}")

log=[]; refs={}; citado_map={}
for i,(sent,q,dw,es) in enumerate(diverse,1):
    print(f"\n[{i}/20] {es}: {q}")
    works=openalex_search(q, n=8)
    best=None; best_sc=-1
    for w in works:
        sc=score_candidate(w,dw)
        if sc>best_sc: best_sc=sc; best=w
    if best:
        apa,cita,doi=format_apa(best)
        print(f"  => {cita.encode('ascii','ignore').decode()} | {doi}")
    else:
        apa,cita,doi="No encontrado","",""
    citado_map[sent]=cita
    log.append({"oracion":sent,"query":q,"cita":cita,"apa":apa,"doi":doi,"titulo":best.get("display_name","") if best else ""})
    if cita and cita not in refs: refs[cita]=apa
    time.sleep(0.9)

# reconstruir texto completo citando solo las diversas, el resto queda igual
full_citado=[]
for s in sents:
    cita=citado_map.get(s,"")
    if cita:
        if s.endswith("."): s2=s[:-1]+f" {cita}."
        else: s2=s+f" {cita}"
        full_citado.append(s2)
    else:
        full_citado.append(s)

out=""
for idx,s in enumerate(full_citado):
    out+=s+" "
    if (idx+1)%4==0: out+="\n\n"
refs_sorted=sorted(refs.items())
refs_block="\n\nReferencias\n"+"="*40+"\n"+"\n\n".join([v for k,v in refs_sorted])
OUTPUT_TXT.write_text(out.strip()+"\n"+refs_block, encoding="utf-8")
LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
REFS_PATH.write_text("\n".join([v for k,v in refs_sorted]), encoding="utf-8")
print(f"\nArchivo: {OUTPUT_TXT} ({len(full_citado)} oraciones, {len(refs_sorted)} refs citadas)")
for k,v in refs_sorted:
    print(v[:90].encode("ascii","ignore").decode()+"...")
