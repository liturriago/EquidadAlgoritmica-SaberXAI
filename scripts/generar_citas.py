#!/usr/bin/env python3
"""
generar_citas.py - Genera citas verificadas por oracion via OpenAlex/Crossref
Muestra de 5 oraciones para validacion de formato.
"""
import re
import json
import time
import pathlib
import requests
from urllib.parse import quote

TXT_PATH = pathlib.Path("articulos de entrega/articulo_preentrega.txt")
OUTPUT_TXT = pathlib.Path("articulos de entrega/articulo_preentrega_citado_demo.txt")
LOG_PATH = pathlib.Path("articulos de entrega/citas_log_demo.json")

# --- 1. Split robusto evitando cortes en "et al." y "e.g." ---
def split_sentences(text: str):
    # proteger et al. y etc.
    protected = text.replace("et al.", "et al<dot>").replace("e.g.", "e_g<dot>").replace("i.e.", "i_e<dot>")
    # split por .!? seguido de espacio + mayuscula o salto
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', protected)
    sentences = []
    for p in parts:
        s = p.replace("et al<dot>", "et al.").replace("e_g<dot>", "e.g.").replace("i_e<dot>", "i.e.").strip()
        # restaurar saltos de parrafo como oraciones separadas si quedan muy largas con \n\n
        if len(s) > 20:
            sentences.append(s)
    # dividir parrafos muy largos por \n\n si contienen multiples oraciones pegadas
    final = []
    for s in sentences:
        # si el texto aun contiene "\n\nMarco" etc, separar
        if "\n" in s and len(s) > 500:
            final.extend([x.strip() for x in s.split("\n\n") if x.strip()])
        else:
            final.append(s)
    # filtrar encabezados sin punto
    final = [s for s in final if len(s.split()) > 6]
    return final

def extract_keywords(sentence: str, max_words=8):
    # Mapeo de conceptos ES -> query EN para OpenAlex
    mapping = {
        "minería de datos educativos": "educational data mining",
        "sociología cuantitativa": "computational social science",
        "pérdidas de aprendizaje": "learning loss pandemic",
        "modelos lineales generalizados": "generalized linear model",
        "XGBoost": "XGBoost gradient boosting",
        "LightGBM": "LightGBM",
        "datos tabulares": "tabular data high cardinality",
        "caja negra": "black box algorithmic",
        "inteligencia artificial explicable": "explainable artificial intelligence SHAP",
        "SHAP": "SHAP Shapley",
        "LIME": "LIME explainable",
        "territorio funcional": "functional territory",
        "clases latentes": "latent class analysis",
        "riesgo empírico": "empirical risk minimization",
        "Group DRO": "group distributionally robust optimization",
        "DANN": "domain adversarial neural network",
        "interseccionalidad": "intersectionality algorithmic fairness",
    }
    # buscar mapeos presentes en la oracion
    found = []
    low = sentence.lower()
    for es, en in mapping.items():
        if es.lower() in low:
            found.append(en)
    if found:
        return " ".join(found[:2])
    # fallback: stopwords ES simple
    stop = set("en de la el los las y a del se con para por una un al como es son sobre entre esta este hacia tras desde hasta".split())
    words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{4,}", sentence)
    keywords = [w for w in words if w.lower() not in stop]
    return " ".join(keywords[:max_words]) if keywords else "educational data mining"

def openalex_search(keywords: str, n=5):
    base = "https://api.openalex.org/works"
    q = quote(keywords)
    # sin filtro type muy restrictivo; luego filtramos en codigo
    url = f"{base}?search={q}&per-page={n}&sort=cited_by_count:desc&mailto=liturriago@demo.edu"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        # filtrar solo articulos de revista indexada si hay primary_location
        filtered = [w for w in results if w.get("primary_location") and w.get("type") in ("article","journal-article", None) or True]
        return filtered[:n] if filtered else results[:n]
    except Exception as e:
        print(f"  [OpenAlex error] {e} | query={keywords}")
        return []

def format_apa(work: dict):
    # work es dict de OpenAlex
    title = work.get("display_name","").rstrip(".")
    year = work.get("publication_year","s.f.")
    authors = work.get("authorships", [])
    journal = (work.get("primary_location") or {}).get("source", {}) or {}
    journal_name = journal.get("display_name","Revista indexada")
    doi = work.get("doi") or work.get("ids",{}).get("doi","")
    # autores APA
    if not authors:
        auth_str = "Autor desconocido"
    elif len(authors) == 1:
        last = authors[0]["author"]["display_name"].split()[-1]
        auth_str = f"{last}"
    elif len(authors) == 2:
        a1 = authors[0]["author"]["display_name"].split()[-1]
        a2 = authors[1]["author"]["display_name"].split()[-1]
        auth_str = f"{a1} & {a2}"
    else:
        a1 = authors[0]["author"]["display_name"].split()[-1]
        auth_str = f"{a1} et al."
    apa = f"{auth_str} ({year}). {title}. {journal_name}."
    if doi:
        apa += f" {doi}"
    return apa, f"({auth_str}, {year})"

def main(limit=5):
    text = TXT_PATH.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    print(f"Total oraciones detectadas: {len(sentences)}")
    # tomar 5 oraciones diversas para demo: 1, 6, 15, 21, 27 (EDM, caja negra, territorio funcional, segmentacion, interseccionalidad)
    indices = [0, 5, 10, 18, 25]  # 0-based
    demo_sentences = [sentences[i] for i in indices if i < len(sentences)]

    log = []
    output_lines = []
    for idx, sent in enumerate(demo_sentences, 1):
        print(f"\n[{idx}/5] Oracion: {sent[:120]}...")
        kw = extract_keywords(sent)
        print(f"  Keywords: {kw}")
        works = openalex_search(kw, n=5)
        if not works:
            # fallback: termino generico
            works = openalex_search("educational data mining", n=5)
        if works:
            best = works[0]
            apa, cita = format_apa(best)
            title = best.get("display_name","")[:80]
            cited = best.get("cited_by_count",0)
            doi = best.get("doi","")
            print(f"  -> Elegido: {title}... ({best.get('publication_year')}) citado {cited} | {doi}")
        else:
            apa, cita = "(No encontrado - revision manual requerida)", ""
            print("  -> Sin resultados")
        sent_citado = sent.rstrip() + f" {cita}" if cita else sent
        output_lines.append(sent_citado)
        log.append({
            "oracion_original": sent,
            "keywords": kw,
            "oracion_citada": sent_citado,
            "apa": apa,
            "doi": best.get("doi","") if works else "",
            "titulo": best.get("display_name","") if works else "",
            "year": best.get("publication_year","") if works else "",
            "cited_by_count": best.get("cited_by_count",0) if works else 0,
            "openalex_id": best.get("id","") if works else ""
        })
        time.sleep(0.6)  # respeto rate limit

    OUTPUT_TXT.write_text("\n\n".join(output_lines) + "\n", encoding="utf-8")
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDemo escrita en: {OUTPUT_TXT}")
    print(f"Log trazable en: {LOG_PATH}")
    for entry in log:
        print(f"\n- {entry['oracion_original'][:70]}... \n  => {entry['apa']}")

if __name__ == "__main__":
    main()
