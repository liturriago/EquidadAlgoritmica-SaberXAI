#!/usr/bin/env python3
"""
generar_citas_full.py - Corrida completa 45 oraciones con filtro de relevancia
"""
import re, json, time, pathlib, requests
from urllib.parse import quote

TXT_PATH = pathlib.Path("articulos de entrega/articulo_preentrega.txt")
OUTPUT_TXT = pathlib.Path("articulos de entrega/articulo_preentrega_citado.txt")
REFS_PATH = pathlib.Path("articulos de entrega/referencias_APA.txt")
LOG_PATH = pathlib.Path("articulos de entrega/citas_log_full.json")

def split_sentences(text: str):
    protected = text.replace("et al.", "et al<dot>").replace("e.g.", "e_g<dot>").replace("i.e.", "i_e<dot>")
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', protected)
    sentences = []
    for p in parts:
        s = p.replace("et al<dot>", "et al.").replace("e_g<dot>", "e.g.").replace("i_e<dot>", "i.e.").strip()
        if len(s) > 20:
            sentences.append(s)
    final = []
    for s in sentences:
        if "\n" in s and len(s) > 500:
            final.extend([x.strip() for x in s.split("\n\n") if x.strip()])
        else:
            final.append(s)
    final = [s for s in final if len(s.split()) > 6 and not s.startswith("Marco")]
    return final

# Mapeo ES -> EN + filtro de dominio
MAPPING = {
    "minería de datos educativos": ("educational data mining", ["education","learning","student"]),
    "sociología cuantitativa": ("computational social science education", ["education","social"]),
    "pérdidas de aprendizaje": ("learning loss pandemic Latin America", ["learning","pandemic","education"]),
    "modelos lineales generalizados": ("generalized linear model education", ["model","regression"]),
    "XGBoost": ("XGBoost gradient boosting tabular", ["xgboost","gradient","tree"]),
    "LightGBM": ("LightGBM tabular", ["lightgbm"]),
    "datos tabulares": ("tabular data high cardinality", ["tabular","categorical"]),
    "caja negra": ("black box interpretability machine learning", ["black","interpret","explain"]),
    "inteligencia artificial explicable": ("explainable artificial intelligence SHAP", ["explain","shap"]),
    "SHAP": ("SHAP Shapley additive explanations", ["shap","shapley"]),
    "LIME": ("LIME explainable AI", ["lime"]),
    "territorio funcional": ("functional territory CEPAL", ["territory","functional","region"]),
    "clases latentes": ("latent class analysis polytomous", ["latent","class"]),
    "riesgo empírico": ("empirical risk minimization", ["empirical","risk"]),
    "Group DRO": ("group distributionally robust optimization Sagawa", ["robust","group","dro"]),
    "DANN": ("domain adversarial neural network Ganin", ["domain","adversarial"]),
    "interseccionalidad": ("intersectionality algorithmic fairness", ["intersection","fairness"]),
    "heterogeneidad": ("heterogeneity educational data", ["heterogen"]),
    "domain shift": ("domain shift covariate", ["domain","shift"]),
    "multicolinealidad": ("multicollinearity", ["multicollinear"]),
}

def extract_query(sentence: str):
    low = sentence.lower()
    for es, (en, domain_words) in MAPPING.items():
        if es.lower() in low:
            return en, domain_words
    # fallback: extraer 5 palabras clave ingles/espanol y mapear a EN generico
    stop = set("en de la el los las y a del se con para por una un al como es son sobre entre esta este hacia tras desde hasta".split())
    words = re.findall(r"[A-Za-z]{4,}", sentence)
    kws = [w for w in words if w.lower() not in stop][:6]
    return " ".join(kws) if kws else "educational data mining", ["education","learning"]

def openalex_search(query: str, n=10, retries=3):
    base = "https://api.openalex.org/works"
    q = quote(query)
    url = f"{base}?search={q}&per-page={n}&sort=relevance_score:desc&mailto=liturriago@demo.edu"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [429 rate limit] esperando {wait}s (intento {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                wait = 10 * (attempt + 1)
                print(f"  [429] backoff {wait}s")
                time.sleep(wait)
                continue
            print(f"  [OpenAlex error] {e}")
            return []
        except Exception as e:
            print(f"  [OpenAlex error] {e}")
            return []
    print(f"  [OpenAlex] agotados reintentos para: {query}")
    return []

def score_candidate(work, domain_words, query):
    title = (work.get("display_name") or "").lower()
    abs_txt = (work.get("abstract") or "").lower() if isinstance(work.get("abstract"), str) else ""
    # algunos works no traen abstract, usar conceptos
    concepts = " ".join([c.get("display_name","").lower() for c in work.get("concepts",[])[:5]])
    text = title + " " + abs_txt + " " + concepts
    domain_hit = sum(1 for w in domain_words if w.lower() in text)
    domain_score = domain_hit / max(len(domain_words),1)
    # penalizar si es de medicina pura cuando buscamos educacion
    if any(x in text for x in ["tumor","cancer","neoplasia"]) and "education" in query.lower():
        domain_score *= 0.1
    relevance = work.get("relevance_score", 0) or 0
    cited = work.get("cited_by_count", 0) or 0
    # score combinado: relevancia * (0.5 + domain) * log citado
    import math
    cite_w = math.log1p(cited)
    return (relevance * (0.7 + domain_score) * (0.5 + cite_w/10))

def format_apa(work):
    title = work.get("display_name","").rstrip(".")
    year = work.get("publication_year") or "s.f."
    authors = work.get("authorships", [])
    journal = (work.get("primary_location") or {}).get("source", {}) or {}
    journal_name = journal.get("display_name") or "Revista indexada"
    doi = work.get("doi") or ""
    # volumen/paginas si estan
    bib = work.get("biblio") or {}
    vol = bib.get("volume")
    # autores APA
    if not authors:
        auth_str = "Autor desconocido"
        auth_full = "Autor desconocido"
    elif len(authors) == 1:
        last = authors[0]["author"]["display_name"].split()[-1]
        auth_str = last
        auth_full = f"{last}, {authors[0]['author']['display_name'].split()[0][0]}."
    elif len(authors) == 2:
        a1 = authors[0]["author"]["display_name"].split()[-1]
        a2 = authors[1]["author"]["display_name"].split()[-1]
        auth_str = f"{a1} & {a2}"
        auth_full = f"{a1} & {a2}"
    else:
        a1 = authors[0]["author"]["display_name"].split()[-1]
        auth_str = f"{a1} et al."
        auth_full = f"{a1} et al."
    # referencia completa APA 7 simplificada
    apa = f"{auth_full} ({year}). {title}. *{journal_name}*"
    if vol:
        apa += f", {vol}"
    if doi:
        apa += f". {doi}"
    else:
        # fallback a openalex id
        apa += f". {work.get('id','')}"
    return apa, f"({auth_str}, {year})", doi

def main():
    text = TXT_PATH.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    print(f"Total oraciones: {len(sentences)}")
    # eliminar duplicados exactos por seguridad
    seen = []
    uniq = []
    for s in sentences:
        if s not in seen:
            seen.append(s)
            uniq.append(s)
    sentences = uniq
    print(f"Oraciones únicas: {len(sentences)}")

    log = []
    citado_sentences = []
    apa_refs = {}  # cita corta -> apa completa (deduplicada por doi)

    for i, sent in enumerate(sentences, 1):
        print(f"\n[{i}/{len(sentences)}] {sent[:110]}...")
        query, domain_words = extract_query(sent)
        print(f"  Query: {query} | dominio: {domain_words}")
        works = openalex_search(query, n=10)
        if not works:
            print("  -> sin resultados, fallback generico")
            works = openalex_search("educational data mining", n=10)
        # elegir mejor por score de dominio
        best = None
        best_score = -1
        for w in works:
            sc = score_candidate(w, domain_words, query)
            if sc > best_score:
                best_score = sc
                best = w
        if best:
            apa, cita, doi = format_apa(best)
            title = (best.get("display_name","")[:70]).encode("ascii","ignore").decode("ascii")
            safe_cita = cita.encode("ascii","ignore").decode("ascii")
            print(f"  => {safe_cita} -> {title}... | score {best_score:.1f} | {doi}")
        else:
            apa, cita, doi = "No encontrado", "", ""
            print("  => No encontrado")
        sent_citado = sent.rstrip()
        # evitar doble punto si la oracion ya termina con .
        if sent_citado.endswith("."):
            sent_citado = sent_citado[:-1] + f" {cita}."
        else:
            sent_citado = sent_citado + f" {cita}"
        citado_sentences.append(sent_citado)
        log.append({
            "idx": i,
            "oracion_original": sent,
            "query": query,
            "oracion_citada": sent_citado,
            "cita_corta": cita,
            "apa": apa,
            "doi": doi,
            "titulo": best.get("display_name","") if best else "",
            "year": best.get("publication_year","") if best else "",
            "score": best_score if best else 0
        })
        if cita not in apa_refs and apa != "No encontrado":
            apa_refs[cita] = apa
        time.sleep(1.2)  # respetar rate limit OpenAlex (10 req/s max)

    # reconstruir texto con parrafos (cada 3-4 oraciones un parrafo)
    # para mantener legibilidad, agrupar de 4
    out_text = ""
    for idx, s in enumerate(citado_sentences):
        out_text += s + " "
        if (idx+1) % 4 == 0:
            out_text += "\n\n"
    out_text = out_text.strip()

    # agregar seccion referencias
    refs_sorted = sorted(apa_refs.items())  # alfabetico por cita
    refs_block = "\n\nReferencias\n" + "="*40 + "\n"
    for cita, apa in refs_sorted:
        refs_block += apa + "\n\n"

    OUTPUT_TXT.write_text(out_text + "\n" + refs_block, encoding="utf-8")
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    REFS_PATH.write_text("\n".join([v for k,v in refs_sorted]), encoding="utf-8")
    print(f"\n\nArchivo citado: {OUTPUT_TXT} ({len(citado_sentences)} oraciones)")
    print(f"Referencias unicas: {len(refs_sorted)}")
    print(f"Log: {LOG_PATH}")
    for cita, apa in refs_sorted[:5]:
        safe = apa[:100].encode("ascii","ignore").decode("ascii")
        print(f"  {safe}...")

if __name__ == "__main__":
    main()
