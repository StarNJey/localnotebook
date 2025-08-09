class AnalyzerAgent:
    """
    • 각 문서(PDF/웹) → relevance 1-10점·핵심 insight 추출
    • 상위 문서들 간 cross-validation(일관/모순) 수행
    """
    def __init__(self, tok, model, dcfg):
        self.tok, self.model, self.dcfg = tok, model, dcfg

    def _ask(self, prompt, max_tok=256):
        inp = self.tok(prompt, return_tensors="pt").to(self.dcfg.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_tok,
                                      do_sample=False, eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def analyse(self, query:str, docs:list[dict]) -> list[dict]:
        for d in docs:
            p = f"Q:{query}\nSnippet:{d['text'][:400]}…\n" \
                "Give relevance 1-10 and a one-line key-insight."
            resp = self._ask(p)
            m = re.search(r"([1-9]|10)", resp)
            d["analysis_score"] = (int(m.group()) if m else 5)/10
            d["key_insight"] = resp.splitlines()[-1][:120]
        return docs

    def cross_validate(self, docs:list[dict]) -> dict:
        if len(docs) < 2:
            return {"consistency":1.0,"conflicts":[],"consensus":[]}
        joined = "\n\n".join(f"[{i}] {d['text'][:250]}…" for i,d in enumerate(docs[:6],1))
        resp = self._ask("Analyse consistent vs conflicting points:\n"+joined)
        consis = re.findall(r"✓\s*(.+)", resp)
        confl  = re.findall(r"✗\s*(.+)", resp)
        ratio  = round(len(consis)/(len(consis)+len(confl)+1e-6),2)
        return {"consistency":ratio,"consensus":consis,"conflicts":confl}
