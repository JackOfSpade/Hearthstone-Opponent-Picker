"""Stdlib HTTP dashboard server.

Endpoints:
* ``GET  /``            - the single-page dashboard (HTML below)
* ``GET  /status``      - JSON status from the EngineController
* ``GET  /screen.png``  - latest phone screencap (throttled by the client)
* ``POST /start``       - start the hunt (optional JSON criteria overrides)
* ``POST /stop``        - request a clean stop
* ``POST /ack``         - acknowledge / silence the target alarm
* ``GET  /criteria``    - current criteria + risk summary (for the risk meter)

The server holds an :class:`~hop.runner.EngineController` and a snapshot of the
loaded :class:`~hop.config.Config` (for the criteria form + pass-rate risk
meter). It binds to localhost only.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import Config
from ..hero_classes import DISPLAY_NAMES, HeroClass, parse_class
from ..runner import EngineController


def _index_html() -> bytes:
    return _INDEX_HTML.encode("utf-8")


class DashboardServer:
    def __init__(self, controller: EngineController, cfg: Config, host: str = "127.0.0.1", port: int = 8765):
        self.controller = controller
        self.cfg = cfg
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None

    def serve_forever(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj: dict):
                self._send(code, json.dumps(obj).encode(), "application/json")

            def _read_body(self) -> dict:
                length = int(self.headers.get("Content-Length", 0) or 0)
                if not length:
                    return {}
                try:
                    return json.loads(self.rfile.read(length).decode() or "{}")
                except Exception:
                    return {}

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    self._send(200, _index_html(), "text/html; charset=utf-8")
                elif self.path.startswith("/status"):
                    self._json(200, server.controller.status())
                elif self.path.startswith("/criteria"):
                    self._json(200, server._criteria_summary())
                elif self.path.startswith("/screen.png"):
                    png = server.controller.latest_png()
                    if png:
                        self._send(200, png, "image/png")
                    else:
                        self._send(503, b"no frame", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                body = self._read_body()
                if self.path.startswith("/start"):
                    started = server.controller.start(body)
                    self._json(200, {"started": started})
                elif self.path.startswith("/stop"):
                    server.controller.stop()
                    self._json(200, {"stopped": True})
                elif self.path.startswith("/ack"):
                    server.controller.ack_alarm()
                    self._json(200, {"acked": True})
                elif self.path.startswith("/bugreport"):
                    try:
                        n = server._bugreport(body.get("description", ""))
                        self._json(200, {"copied": True, "bytes": n})
                    except Exception as e:
                        self._json(500, {"error": str(e)})
                else:
                    self._send(404, b"not found", "text/plain")

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        if self._httpd:
            self._httpd.shutdown()

    def _bugreport(self, description: str) -> int:
        """Assemble a self-improving bug report and put it on the clipboard; return the
        report size in bytes so the UI can confirm something landed.

        Skips the live device probe (a fresh adb connection could contend with the
        hunt's running transport), but DOES attach a live engine-status snapshot from
        the controller (cheap, in-process) so the report says what the hunt was doing.
        The journal, config and logs carry the rest of the diagnosis.
        """
        from pathlib import Path
        from .. import __version__
        from .. import bugreport as br

        home = Path.home()
        paths = br.ReportPaths(
            config=home / ".config" / "hop" / "config.toml",
            runs_root=home / ".config" / "hop" / "runs",
            unknowns_dir=home / ".config" / "hop" / "unknowns",
            app_log=home / "Library" / "Logs" / "hop.log",
            templates=home / ".config" / "hop" / "templates",
        )
        md = br.collect(description, paths, version=__version__, device_probe=None,
                        status_probe=lambda: br.format_status(self.controller.status()))
        if not br.copy_to_clipboard(md):
            raise RuntimeError("could not copy the report to the clipboard (pbcopy failed)")
        return len(md.encode("utf-8"))

    def _criteria_summary(self) -> dict:
        crit = self.cfg.criteria
        pass_rate = crit.pass_rate_estimate()
        concede_rate = 1.0 - pass_rate
        # barcode-risk heuristic: very high concede rate is the risky shape
        if concede_rate >= 0.9:
            risk = "high"
        elif concede_rate >= 0.7:
            risk = "elevated"
        else:
            risk = "moderate"
        return {
            "all_classes": [DISPLAY_NAMES[c] for c in HeroClass],
            "target_classes": [DISPLAY_NAMES[c] for c in crit.target_classes],
            "require_second": crit.require_second,
            "mode": crit.mode,
            "risk_profile": self.cfg.risk_profile,
            "pass_rate": round(pass_rate, 3),
            "concede_rate": round(concede_rate, 3),
            "barcode_risk": risk,
        }


def serve(controller: EngineController, cfg: Config, host: str = "127.0.0.1", port: int = 8765) -> DashboardServer:
    srv = DashboardServer(controller, cfg, host, port)
    srv.serve_forever()
    return srv


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>hop dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{color-scheme:dark light;--bg:#12141a;--card:#1c1f28;--fg:#e6e8ee;--mut:#9aa0ad;--acc:#4da3ff;--warn:#ffb454;--bad:#ff5c6c;--ok:#5cd6a0}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
  header{padding:14px 20px;border-bottom:1px solid #2a2e38;display:flex;align-items:center;gap:12px}
  h1{font-size:16px;margin:0;font-weight:600}
  .pill{font-size:12px;padding:2px 8px;border-radius:10px;background:#2a2e38;color:var(--mut)}
  main{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px;max-width:1100px}
  @media(max-width:800px){main{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid #2a2e38;border-radius:12px;padding:16px}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:0 0 12px}
  button{font:inherit;border:0;border-radius:8px;padding:9px 14px;cursor:pointer;background:var(--acc);color:#06121f;font-weight:600}
  button.sec{background:#2a2e38;color:var(--fg)}
  button.warn{background:var(--warn);color:#241a06}
  button:disabled{opacity:.5;cursor:not-allowed}
  .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
  label.chk{display:inline-flex;align-items:center;gap:5px;font-size:13px;background:#22262f;padding:5px 9px;border-radius:8px;cursor:pointer}
  .stat{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;border-bottom:1px solid #23272f}
  .stat b{font-weight:600}
  .bar{height:8px;background:#23272f;border-radius:5px;overflow:hidden;margin-top:4px}
  .bar>i{display:block;height:100%;background:var(--acc)}
  .risk-high{color:var(--bad)} .risk-elevated{color:var(--warn)} .risk-moderate{color:var(--ok)}
  img#screen{width:100%;border-radius:8px;background:#000;min-height:120px;object-fit:contain}
  .mut{color:var(--mut);font-size:12px}
  #banner{display:none;padding:12px 16px;background:var(--ok);color:#04231a;font-weight:700;border-radius:10px;margin-bottom:12px}
  #banner.err{background:var(--bad);color:#2a0508}
</style></head>
<body>
<header><h1>hop</h1><span class="pill" id="conn">connecting…</span><span class="pill" id="prof"></span></header>
<main>
  <section>
    <div id="banner"></div>
    <div class="card">
      <h2>Criteria</h2>
      <div class="mut">Target classes (empty = any):</div>
      <div class="row" id="classes"></div>
      <div class="row">
        <label class="chk"><input type="checkbox" id="second"> Only when going 2nd</label>
      </div>
      <div class="row">
        <button id="startBtn">Start Search</button>
        <button id="stopBtn" class="sec">Stop</button>
        <button id="ackBtn" class="warn">Silence alarm</button>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Barcode-risk meter</h2>
      <div class="stat"><span>Estimated pass rate</span><b id="passrate">–</b></div>
      <div class="stat"><span>Estimated concede rate</span><b id="concederate">–</b></div>
      <div class="stat"><span>Risk shape</span><b id="risk">–</b></div>
      <div class="mut" style="margin-top:8px">Fewer target classes + require-2nd → higher concede rate → more barcode-like. Widen criteria to reduce risk and hit targets faster.</div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Report a bug</h2>
      <div class="mut">Copies a self-improving report (logs + live state) to your clipboard. Paste it straight into Claude Code.</div>
      <textarea id="bugdesc" rows="3" placeholder="What went wrong?" style="width:100%;margin-top:8px;background:#22262f;color:var(--fg);border:1px solid #2a2e38;border-radius:8px;padding:8px;font:inherit;box-sizing:border-box"></textarea>
      <div class="row"><button id="bugBtn" class="sec">Copy report to clipboard</button><span class="mut" id="bugout"></span></div>
    </div>
  </section>
  <section>
    <div class="card">
      <h2>Observed class distribution</h2>
      <svg id="dist" viewBox="0 0 320 200" width="100%" role="img" aria-label="opponent class distribution"></svg>
      <div class="mut" id="disttot">no games yet this session</div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Session</h2>
      <div class="stat"><span>Games</span><b id="games">0</b></div>
      <div class="stat"><span>Concedes</span><b id="concedes">0</b></div>
      <div class="stat"><span>Concedes until target</span><b id="untiltarget">–</b></div>
      <div class="stat"><span>Coin split (1st / 2nd)</span><b id="coin">0 / 0</b></div>
      <div class="stat"><span>Committing ratio</span><b id="ratio">0</b></div>
      <div class="stat"><span>Actions this run</span><b id="actions">0</b></div>
      <div class="bar"><i id="actbar" style="width:0%"></i></div>
      <div class="stat"><span>Session minutes</span><b id="mins">0</b></div>
      <div class="stat"><span>Last opponent</span><b id="lastopp">–</b></div>
      <div class="stat"><span>Stop reason</span><b id="stopreason">–</b></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>HumanState</h2>
      <div class="stat"><span>Attention</span><b id="attention">–</b></div>
      <div class="stat"><span>Confidence</span><b id="confidence">–</b></div>
      <div class="stat"><span>Fatigue</span><b id="fatigue">–</b></div>
      <div class="stat"><span>Familiarity</span><b id="familiarity">–</b></div>
    </div>
  </section>
</main>
<script>
let CRIT=null;
async function loadCriteria(){
  CRIT=await (await fetch('/criteria')).json();
  document.getElementById('prof').textContent=CRIT.risk_profile+' · '+CRIT.mode;
  const box=document.getElementById('classes'); box.innerHTML='';
  CRIT.all_classes.forEach(c=>{
    const on=CRIT.target_classes.includes(c);
    const l=document.createElement('label'); l.className='chk';
    l.innerHTML=`<input type="checkbox" value="${c}" ${on?'checked':''}> ${c}`;
    box.appendChild(l);
  });
  document.getElementById('second').checked=CRIT.require_second;
  box.onchange=updateRiskFromForm;
  document.getElementById('second').onchange=updateRiskFromForm;
  updateRiskFromForm();
}
function chosenCriteria(){
  const classes=[...document.querySelectorAll('#classes input:checked')].map(i=>i.value);
  return {target_classes:classes, require_second:document.getElementById('second').checked};
}
function updateRiskFromForm(){
  if(!CRIT) return;
  const chosen=chosenCriteria();
  let pass=chosen.target_classes.length?chosen.target_classes.length/Math.max(1,CRIT.all_classes.length):1;
  if(chosen.require_second) pass*=0.5;
  const concede=1-pass;
  let risk='moderate';
  if(concede>=0.9) risk='high';
  else if(concede>=0.7) risk='elevated';
  document.getElementById('passrate').textContent=(pass*100).toFixed(0)+'%';
  document.getElementById('concederate').textContent=(concede*100).toFixed(0)+'%';
  const r=document.getElementById('risk'); r.textContent=risk; r.className='risk-'+risk;
}
async function start(){ await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(chosenCriteria())}); }
async function stop(){ await fetch('/stop',{method:'POST'}); }
async function ack(){ await fetch('/ack',{method:'POST'}); document.getElementById('banner').style.display='none'; }
document.getElementById('startBtn').onclick=start;
document.getElementById('stopBtn').onclick=stop;
document.getElementById('ackBtn').onclick=ack;
async function poll(){
  try{
    const s=await (await fetch('/status')).json();
    document.getElementById('conn').textContent=s.running?'running':'idle';
    document.getElementById('startBtn').disabled=s.running;
    // Surface a Search that died on arrival (e.g. missing vision deps) rather than let it
    // read as "nothing happened": a failed start leaves last_error set and running=false.
    const b=document.getElementById('banner');
    if(s.target_found){
      b.className=''; b.style.display='block';
      b.textContent='🎯 TARGET FOUND — '+(s.last_opponent||'')+' — your turn! (Silence alarm to dismiss)';
    }else if(s.last_error && !s.running){
      b.className='err'; b.style.display='block';
      b.textContent='⚠ Search stopped — '+s.last_error;
    }else if(b.className==='err'){
      b.className=''; b.style.display='none';
    }
    if(s.games!==undefined){
      document.getElementById('games').textContent=s.games;
      document.getElementById('concedes').textContent=s.concedes;
      document.getElementById('untiltarget').textContent=(s.concedes_until_target!=null)?s.concedes_until_target:'–';
      document.getElementById('coin').textContent=(s.going_first||0)+' / '+(s.going_second||0);
      document.getElementById('stopreason').textContent=s.stop_reason||(s.last_error||'–');
      document.getElementById('lastopp').textContent=s.last_opponent||'–';
      drawDistribution(s.class_distribution||{});
      if(s.budget){
        document.getElementById('ratio').textContent=s.budget.committing_ratio;
        document.getElementById('actions').textContent=s.budget.actions_run+' / '+s.budget.actions_run_cap;
        document.getElementById('mins').textContent=s.budget.session_minutes;
        const pct=Math.min(100,100*s.budget.actions_run/Math.max(1,s.budget.actions_run_cap));
        document.getElementById('actbar').style.width=pct+'%';
      }
      if(s.human_state){
        document.getElementById('attention').textContent=s.human_state.attention;
        document.getElementById('confidence').textContent=s.human_state.confidence;
        document.getElementById('fatigue').textContent=s.human_state.fatigue;
        document.getElementById('familiarity').textContent=s.human_state.familiarity;
      }
    }
  }catch(e){ document.getElementById('conn').textContent='offline'; }
}
// Dependency-free horizontal bar chart of the opponent class distribution, built from
// SVG <rect>/<text> so it needs no charting library and no CDN (the server is local).
function drawDistribution(dist){
  const svg=document.getElementById('dist');
  const entries=Object.entries(dist).sort((a,b)=>b[1]-a[1]);
  const total=entries.reduce((n,[,v])=>n+v,0);
  document.getElementById('disttot').textContent=total?(total+' game'+(total==1?'':'s')+' this session'):'no games yet this session';
  const W=320, rowH=22, gap=6, labelW=96, x0=labelW+6, maxW=W-x0-34;
  const max=Math.max(1,...entries.map(([,v])=>v));
  const H=Math.max(40, entries.length*(rowH+gap));
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  let out='';
  entries.forEach(([cls,v],i)=>{
    const y=i*(rowH+gap), w=Math.max(2,maxW*v/max);
    out+=`<text x="0" y="${y+rowH*0.7}" fill="#9aa0ad" font-size="12">${cls}</text>`;
    out+=`<rect x="${x0}" y="${y}" width="${w}" height="${rowH}" rx="4" fill="#4da3ff"/>`;
    out+=`<text x="${x0+w+5}" y="${y+rowH*0.7}" fill="#e6e8ee" font-size="12">${v}</text>`;
  });
  svg.innerHTML=out || '<text x="0" y="20" fill="#9aa0ad" font-size="12">no games yet</text>';
}
document.getElementById('bugBtn').onclick=async()=>{
  const out=document.getElementById('bugout'), btn=document.getElementById('bugBtn');
  btn.disabled=true; out.textContent='copying…';
  try{
    const r=await (await fetch('/bugreport',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({description:document.getElementById('bugdesc').value})})).json();
    out.textContent=r.copied?('✓ copied '+(r.bytes?'('+Math.round(r.bytes/1024)+' KB) ':'')+'— paste into Claude Code'):('error: '+(r.error||'?'));
  }catch(e){ out.textContent='failed: '+e; }
  btn.disabled=false;
};
loadCriteria();
setInterval(poll,1000);
</script>
</body></html>
"""
