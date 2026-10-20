"""Multi-camera MJPEG livestream server (stdlib only, no new deps).

Serves every CameraRig stream over plain HTTP so the audience, the Hermes
chat user, and any browser on the LAN can watch what the robot sees while it
acts -- detections, agent status and world state overlaid live.

Routes:
    /                    dark dashboard: N camera tiles + world-state panel
    /stream/<name>       multipart/x-mixed-replace MJPEG (annotated)
    /snapshot/<name>.jpg one annotated frame
    /state               JSON: beliefs (with colors), cameras, agent status

The handler threads only ever *read* the latest frame slot of each stream,
so any number of viewers can attach without slowing perception or control.
"""

from __future__ import annotations

import json
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .live_view import draw_detections, draw_grasp_pose, draw_hud

_BOUNDARY = "wrcframe"


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Suppress routine browser disconnect noise without hiding real faults."""

    def handle_error(self, request, client_address) -> None:
        # Python 3.10 compatibility: sys.exception() was added in 3.11.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        if isinstance(exc, ssl.SSLError) and "UNEXPECTED_EOF_WHILE_READING" in str(exc):
            return
        super().handle_error(request, client_address)

_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>wrc-demo :: live</title><style>
 :root {{
   --bg:#05070d; --panel-hi:rgba(20,28,44,.92); --panel-lo:rgba(10,15,26,.92);
   --line:rgba(148,163,184,.12); --line-hi:rgba(56,189,248,.25);
   --txt:#e6edf3; --muted:#8b98a9; --dim:#5b6b7d;
   --cyan:#22d3ee; --violet:#8b5cf6; --ok:#34d399; --fail:#f87171; --warn:#fbbf24;
 }}
 * {{ box-sizing:border-box; }}
 html {{ scrollbar-width:thin; scrollbar-color:#22304a transparent; }}
 body {{ background:
     radial-gradient(1100px 520px at 88% -12%, rgba(139,92,246,.11), transparent 60%),
     radial-gradient(950px 480px at -8% -4%, rgba(34,211,238,.09), transparent 55%),
     var(--bg);
   color:var(--txt); font:14px/1.4 system-ui,'Segoe UI',sans-serif; margin:0; }}
 body::before {{  /* faint engineering grid, fades out toward the fold */
   content:''; position:fixed; inset:0; pointer-events:none;
   background:
     linear-gradient(rgba(148,163,184,.045) 1px, transparent 1px),
     linear-gradient(90deg, rgba(148,163,184,.045) 1px, transparent 1px);
   background-size:44px 44px;
   -webkit-mask-image:radial-gradient(ellipse at 50% 0%, #000 25%, transparent 72%);
   mask-image:radial-gradient(ellipse at 50% 0%, #000 25%, transparent 72%);
 }}
 ::-webkit-scrollbar {{ width:8px; height:8px; }}
 ::-webkit-scrollbar-thumb {{ background:#22304a; border-radius:4px; }}
 ::-webkit-scrollbar-thumb:hover {{ background:#31425f; }}

 /* ── command bar ── */
 header {{ position:relative; padding:10px 18px; display:flex; gap:14px; align-items:center;
   background:linear-gradient(180deg, rgba(13,20,34,.95), rgba(9,13,24,.88));
   border-bottom:1px solid transparent;
   border-image:linear-gradient(90deg, var(--cyan), var(--violet), transparent 85%) 1;
   backdrop-filter:blur(6px); }}
 .brand {{ display:flex; align-items:center; gap:11px; user-select:none; }}
 .logo {{ width:24px; height:24px; border-radius:7px; transform:rotate(45deg);
   background:linear-gradient(135deg, var(--cyan), var(--violet));
   box-shadow:0 0 14px rgba(34,211,238,.55); flex:none; }}
 .t1 {{ font-size:14px; font-weight:800; letter-spacing:.14em; }}
 .t2 {{ font-size:8.5px; font-weight:600; letter-spacing:.34em; color:var(--dim);
   text-transform:uppercase; margin-top:1px; }}
 #conn {{ padding:3px 10px; border-radius:999px; border:1px solid var(--line);
   font:700 10.5px ui-monospace,monospace; letter-spacing:.1em; color:var(--muted);
   white-space:nowrap; }}
 #conn::before {{ content:'\\25cf  '; }}
 #conn.online {{ color:var(--ok); border-color:rgba(52,211,153,.4);
   background:rgba(52,211,153,.08); animation:breath 2.2s ease-in-out infinite; }}
 #conn.offline {{ color:var(--fail); border-color:rgba(248,113,113,.45);
   background:rgba(248,113,113,.08); font-weight:800; }}
 @keyframes breath {{
   0%,100% {{ box-shadow:0 0 0 0 rgba(52,211,153,.30); }}
   50% {{ box-shadow:0 0 0 5px rgba(52,211,153,0); }}
 }}
 #task {{ color:var(--warn); font:600 12px system-ui,sans-serif; white-space:nowrap;
   overflow:hidden; text-overflow:ellipsis; max-width:34vw; }}
 #task:not(:empty)::before {{ content:'\\25B8  '; color:var(--dim); }}
 #status {{ margin-left:auto; display:flex; gap:8px; align-items:center; }}
 .pill {{ padding:3px 10px; border-radius:999px; border:1px solid var(--line);
   font:600 10.5px ui-monospace,monospace; letter-spacing:.06em; color:var(--muted);
   white-space:nowrap; }}
 .pill.ok {{ color:var(--ok); border-color:rgba(52,211,153,.35); background:rgba(52,211,153,.07); }}
 .pill.warn {{ color:var(--warn); border-color:rgba(251,191,36,.35); background:rgba(251,191,36,.07); }}
 #st-agent {{ max-width:30vw; overflow:hidden; text-overflow:ellipsis; }}

 /* ── layout ── */
 main {{ position:relative; display:grid; grid-template-columns: 1fr 384px; gap:14px;
   padding:14px; align-items:start; }}
 #cams {{ display:flex; flex-wrap:wrap; gap:14px; align-content:start; }}
 .cam {{ background:linear-gradient(180deg, var(--panel-hi), var(--panel-lo));
   border:1px solid var(--line); border-radius:14px; padding:10px; min-width:0;
   box-shadow:0 14px 34px rgba(0,0,0,.38); }}
 .cam h2 {{ display:flex; align-items:center; font-size:12px; margin:0 0 8px 2px;
   color:var(--muted); font-weight:600; letter-spacing:.08em; text-transform:uppercase; }}
 .camdot {{ display:inline-block; width:7px; height:7px; border-radius:50%;
   background:var(--fail); box-shadow:0 0 7px var(--fail); margin-right:8px;
   animation:blink 1.8s ease-in-out infinite; flex:none; }}
 @keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.35; }} }}
 .frame {{ position:relative; border-radius:8px; }}
 .cam img {{ width:100%; max-width:100%; height:auto; aspect-ratio:16/9;
   border-radius:8px; display:block; object-fit:contain; background:#000;
   transition:opacity .3s; }}
 .cam img.dead {{ opacity:.25; filter:grayscale(1); }}
 .frame .c {{ position:absolute; width:18px; height:18px; pointer-events:none;
   border:0 solid rgba(34,211,238,.8); filter:drop-shadow(0 0 4px rgba(34,211,238,.5)); }}
 .c-tl {{ top:7px;  left:7px;  border-top-width:2px;    border-left-width:2px;
   border-top-left-radius:6px; }}
 .c-tr {{ top:7px;  right:7px; border-top-width:2px;    border-right-width:2px;
   border-top-right-radius:6px; }}
 .c-bl {{ bottom:7px; left:7px;  border-bottom-width:2px; border-left-width:2px;
   border-bottom-left-radius:6px; }}
 .c-br {{ bottom:7px; right:7px; border-bottom-width:2px; border-right-width:2px;
   border-bottom-right-radius:6px; }}

 aside {{ display:flex; flex-direction:column; gap:14px; min-width:300px; }}
 .panel {{ background:linear-gradient(180deg, var(--panel-hi), var(--panel-lo));
   border:1px solid var(--line); border-radius:14px; padding:12px 14px;
   box-shadow:0 14px 34px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.03); }}
 .panel h3 {{ display:flex; align-items:center; gap:9px; margin:0 0 10px;
   font-size:10.5px; color:#7dd3fc; text-transform:uppercase; letter-spacing:.14em;
   font-weight:700; }}
 .panel h3::before {{ content:''; width:8px; height:8px; border-radius:3px; flex:none;
   background:linear-gradient(135deg, var(--cyan), var(--violet));
   box-shadow:0 0 9px rgba(34,211,238,.65); }}

 /* ── chat ── */
 #chat {{ display:flex; gap:8px; }}
 #cmd {{ flex:1; min-width:0; background:rgba(5,9,18,.8); border:1px solid #26344d;
   border-radius:9px; color:var(--txt); padding:8px 12px; font:13px system-ui,sans-serif;
   outline:none; transition:border-color .15s, box-shadow .15s; }}
 #cmd:focus {{ border-color:var(--cyan); box-shadow:0 0 0 3px rgba(34,211,238,.18); }}
 #cmd::placeholder {{ color:var(--dim); }}
 .btn {{ border:0; border-radius:9px; font-weight:800; padding:0 16px; cursor:pointer;
   letter-spacing:.06em; font-size:12.5px; transition:filter .12s, transform .12s; }}
 .btn:hover {{ filter:brightness(1.15); transform:translateY(-1px); }}
 .btn:active {{ transform:translateY(0); }}
 .btn-send {{ background:linear-gradient(135deg, var(--cyan), var(--violet));
   color:#04101a; box-shadow:0 4px 16px rgba(34,211,238,.35); }}
 .btn-stop {{ background:linear-gradient(135deg, #f87171, #dc2626); color:#fff;
   box-shadow:0 4px 16px rgba(220,38,38,.35); }}
 #chatmsg {{ font:11px ui-monospace,monospace; color:var(--muted); margin-top:7px;
   min-height:14px; }}
 .btn-mic {{ flex:none; width:38px; padding:0; display:inline-flex; align-items:center;
   justify-content:center; background:rgba(34,211,238,.10);
   border:1px solid rgba(34,211,238,.35); border-radius:9px; color:var(--cyan);
   cursor:pointer; transition:background .12s; }}
 .btn-mic:hover {{ background:rgba(34,211,238,.18); }}
 .btn-mic:disabled {{ opacity:.3; cursor:not-allowed; }}
 .btn-mic.rec {{ background:rgba(248,113,113,.22); border-color:var(--fail);
   color:var(--fail); animation:micpulse 1.1s ease-in-out infinite; }}
 @keyframes micpulse {{
   0%,100% {{ box-shadow:0 0 0 0 rgba(248,113,113,.45); }}
   50% {{ box-shadow:0 0 0 7px rgba(248,113,113,0); }}
 }}

 /* ── narration ── */
 #feed {{ max-height:36vh; overflow-y:auto; margin:0 -4px; padding:0 4px; }}
 .ev {{ display:grid; grid-template-columns:46px 42px 1fr; gap:8px; padding:5px 7px;
   border-radius:8px; align-items:baseline; }}
 .ev:hover {{ background:rgba(148,163,184,.06); }}
 .ev-t {{ color:#3fb6d8; font:10.5px/1.7 ui-monospace,monospace; text-align:right;
   white-space:nowrap; opacity:.8; }}
 .ev-k {{ font-size:9px; font-weight:800; text-align:center; border-radius:5px;
   padding:2.5px 0; letter-spacing:.06em; }}
 .k-obs {{ background:rgba(56,132,244,.14); color:#7db4ff;
   box-shadow:inset 0 0 0 1px rgba(125,180,255,.25); }}
 .k-act {{ background:rgba(52,211,153,.13); color:var(--ok);
   box-shadow:inset 0 0 0 1px rgba(52,211,153,.25); }}
 .k-out {{ background:rgba(248,113,113,.13); color:var(--fail);
   box-shadow:inset 0 0 0 1px rgba(248,113,113,.25); }}
 .k-note {{ background:rgba(251,191,36,.12); color:var(--warn);
   box-shadow:inset 0 0 0 1px rgba(251,191,36,.25); }}
 .ev-x {{ font:12.5px/1.5 system-ui,sans-serif; overflow-wrap:anywhere; color:var(--txt); }}
 .ev-raw {{ color:var(--muted); font:12px/1.5 ui-monospace,monospace; }}

 /* ── agent panel: LLM task decomposition + skill-call timeline ── */
 #ag-head {{ font:600 13px system-ui,sans-serif; overflow-wrap:anywhere; }}
 .ag-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
   margin-right:7px; flex:none; }}
 .ag-dot.run {{ background:var(--warn); box-shadow:0 0 7px var(--warn);
   animation:breath 2.2s ease-in-out infinite; }}
 .ag-dot.ok {{ background:var(--ok); box-shadow:0 0 7px rgba(52,211,153,.55); }}
 .ag-dot.fail {{ background:var(--fail); box-shadow:0 0 7px rgba(248,113,113,.55); }}
 .ag-meta {{ color:var(--muted); font:600 11px ui-monospace,monospace; }}
 .ag-sub {{ margin-top:10px; font:700 9.5px system-ui,sans-serif; letter-spacing:.14em;
   text-transform:uppercase; color:var(--dim); }}
 #ag-milestones {{ margin:4px 0 0 18px; padding:0; }}
 #ag-milestones li {{ font:12.5px/1.55 system-ui,sans-serif; color:var(--txt);
   overflow-wrap:anywhere; }}
 .ag-step {{ font:11.5px/1.6 ui-monospace,monospace; padding:1px 0;
   overflow-wrap:anywhere; }}
 .ag-step.ok::before {{ content:'\\2713  '; color:var(--ok); }}
 .ag-step.fail {{ color:var(--fail); }}
 .ag-step.fail::before {{ content:'\\2717  '; }}
 .ag-step.run {{ color:var(--warn); }}
 .ag-step.run::before {{ content:'\\25B8  '; }}
 .ag-dim {{ color:var(--dim); font:12px/1.5 ui-monospace,monospace; }}
 #ag-llm {{ margin-top:2px; max-height:88px; overflow-y:auto; white-space:pre-wrap;
   font:11.5px/1.5 ui-monospace,monospace; color:var(--muted); overflow-wrap:anywhere; }}

 /* ── objects ── */
 table {{ width:100%; border-collapse:collapse; font:12px ui-monospace,monospace; }}
 th {{ padding:4px 6px; text-align:left; font-size:9.5px; text-transform:uppercase;
   letter-spacing:.1em; color:var(--dim); border-bottom:1px solid var(--line); }}
 td {{ padding:4px 6px; text-align:left; border-bottom:1px solid rgba(148,163,184,.06); }}
 tbody tr {{ transition:background .12s; }}
 tbody tr:hover {{ background:rgba(34,211,238,.05); }}
 .chip {{ display:inline-block; width:10px; height:10px; border-radius:50%;
   margin-right:7px; vertical-align:baseline; border:1px solid rgba(255,255,255,.25); }}
 .remembered {{ opacity:.5; }}
 .multi {{ font:600 9.5px ui-monospace,monospace; color:var(--warn);
   border:1px solid rgba(251,191,36,.4); border-radius:4px; padding:1px 4px;
   margin-left:4px; vertical-align:1px; }}

 /* ── debug details ── */
 details.panel.dev > summary {{ cursor:pointer; display:flex; gap:8px; align-items:center;
   font-size:10.5px; color:#7dd3fc; text-transform:uppercase; letter-spacing:.14em;
   font-weight:700; user-select:none; list-style:none; }}
 details.panel.dev > summary::-webkit-details-marker {{ display:none; }}
 details.panel.dev > summary::before {{ content:'\\25B8'; color:var(--cyan);
   transition:transform .15s; }}
 details.panel.dev[open] > summary::before {{ transform:rotate(90deg); }}
 details.panel.dev[open] > summary {{ margin-bottom:10px; }}
 .debug-grid {{ display:grid; grid-template-columns:150px 1fr; gap:4px 10px;
   font:12px/1.5 ui-monospace,monospace; }}
 .debug-grid .k {{ color:var(--muted); }}
 .debug-grid > div {{ color:var(--txt); overflow-wrap:anywhere; }}
 .sect {{ grid-column:1 / -1; margin:9px 0 3px; padding-top:7px;
   border-top:1px solid var(--line); font-size:9.5px; color:var(--dim);
   text-transform:uppercase; letter-spacing:.12em; font-weight:700; }}
 .sect:first-child {{ margin-top:0; border-top:0; padding-top:0; }}
 .debug-ok {{ color:var(--ok) !important; font-weight:700; }}
 .debug-fail {{ color:var(--fail) !important; font-weight:700; }}
 .debug-warn {{ color:var(--warn) !important; font-weight:700; }}
 .debug-muted {{ color:var(--muted) !important; }}
 .cand {{ font:11.5px/1.6 ui-monospace,monospace; padding:3px 0;
   border-bottom:1px dashed rgba(148,163,184,.10); }}
 .cand:last-child {{ border-bottom:0; }}
 .cand .dim {{ color:var(--muted); }}
 .cand .rej {{ color:var(--fail); }}

 @media (prefers-reduced-motion: reduce) {{
   * {{ animation:none !important; transition:none !important; }}
 }}
</style></head><body>
<header>
 <div class="brand"><div class="logo"></div>
  <div><div class="t1">WRC&middot;DEMO</div><div class="t2">live control</div></div></div>
 <span id="conn">connecting</span>
 <span id="task"></span>
 <div id="status">
  <span class="pill" id="st-cams">--</span>
  <span class="pill" id="st-holding">--</span>
  <span class="pill" id="st-arm">--</span>
  <span class="pill" id="st-agent">--</span>
 </div>
</header>
<main>
 <div id="cams">{tiles}</div>
 <aside>
  <div class="panel"><h3>talk to the robot</h3>
   <form id="chat">
    <input id="cmd" placeholder="pick and place pink object"
      autocomplete="off">
    <button class="btn btn-mic" type="button" id="micbtn"
      title="hold to talk (voice to text, review then send)">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2.4" stroke-linecap="round"
        stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/>
        <path d="M5 11a7 7 0 0 0 14 0"/><line x1="12" y1="18" x2="12" y2="21"/>
      </svg></button>
    <button class="btn btn-send" type="submit">send</button>
    <button class="btn btn-stop" type="button" id="stopbtn">stop</button>
   </form><div id="chatmsg"></div></div>
  <div class="panel"><h3>robot narration</h3><div id="feed">waiting...</div></div>
  <div class="panel" id="agent-panel" style="display:none">
   <h3>agent &middot; task decomposition &amp; skills</h3>
   <div id="ag-head">--</div>
   <div class="ag-sub">llm milestones</div>
   <ol id="ag-milestones"></ol>
   <div class="ag-sub">skill calls</div>
   <div id="ag-steps"><div class="ag-dim">--</div></div>
   <div class="ag-sub">llm latest note</div>
   <div id="ag-llm">--</div>
  </div>
  <div class="panel"><h3>objects in the world model</h3>
   <table><thead><tr><th>object</th><th>color</th><th>position (m)</th><th>state</th></tr></thead>
   <tbody id="objs"></tbody></table></div>
  <details class="panel dev" id="dev"><summary>grasp &amp; ik debug</summary>
   <div class="debug-grid">
    <div class="sect">candidate &amp; attempt</div>
    <div class="k">Status</div><div id="g-status">--</div>
    <div class="k">Target</div><div id="g-target">--</div>
    <div class="k">Grasp call</div><div id="g-call">--</div>
    <div class="k">Attempt in call</div><div id="g-attempt">--</div>
    <div class="k">Candidate</div><div id="g-candidate">--</div>
    <div class="k">Candidate sequence</div><div id="g-sequence">--</div>
    <div class="k">Source</div><div id="g-source">--</div>
    <div class="sect">scoring</div>
    <div class="k">Candidate score</div><div id="g-score">--</div>
    <div class="k">Tilt from vertical</div><div id="g-tilt">--</div>
    <div class="k">Detection confidence</div><div id="g-conf">--</div>
    <div class="k">Required jaw width</div><div id="g-width">--</div>
    <div class="sect">frame geometry</div>
    <div class="k">Camera XYZ</div><div id="g-camera">--</div>
    <div class="k">Base XYZ</div><div id="g-base">--</div>
    <div class="k">GGX raw origin</div><div id="g-raw-origin">--</div>
    <div class="k">Tip offset</div><div id="g-tip-offset">--</div>
    <div class="k">Approach</div><div id="g-approach">--</div>
    <div class="k">Jaw opening axis</div><div id="g-jaw-axis">--</div>
   </div>
   <div class="debug-grid" style="margin-top:10px">
    <div class="sect">ik</div>
    <div class="k">Pregrasp</div><div id="ik-pre-status">--</div>
    <div class="k">Grasp</div><div id="ik-grasp-status">--</div>
    <div class="k">Position error</div><div id="ik-pos-error">--</div>
    <div class="k">Rotation error</div><div id="ik-rot-error">--</div>
    <div class="k">Total log6 error</div><div id="ik-total-error">--</div>
    <div class="k">q current</div><div id="ik-q-current">--</div>
    <div class="k">Reject / error</div><div id="ik-reason">--</div>
    <div class="k">Candidates</div><div id="g-history">--</div>
   </div>
  </details>
 </aside>
</main>
<script>
 const SWATCH = {{red:'#e5484d', orange:'#f76b15', yellow:'#ffe629', green:'#46a758',
   cyan:'#00a2c7', blue:'#0090ff', purple:'#8e4ec6', pink:'#f76190',
   brown:'#ad7f58', white:'#eee', gray:'#888', black:'#111'}};
 document.getElementById('stopbtn').addEventListener('click', async () => {{
   const msg = document.getElementById('chatmsg');
   try {{
     const r = await (await fetch('/cancel', {{method: 'POST'}})).json();
     msg.textContent = r.cancelled ? 'cancelled - the arm is stopping' : (r.error || 'cancel failed');
   }} catch (e) {{ msg.textContent = 'error: ' + e; }}
 }});
 document.getElementById('chat').addEventListener('submit', async (ev) => {{
   ev.preventDefault();
   const box = document.getElementById('cmd');
   const msg = document.getElementById('chatmsg');
   const task = box.value.trim();
   if (!task) return;
   msg.textContent = 'sending...';
   try {{
     const r = await (await fetch('/task', {{method: 'POST',
       headers: {{'Content-Type': 'application/json'}},
       body: JSON.stringify({{task}})}})).json();
     msg.textContent = r.accepted ? `running: ${{r.task}}` : (r.error || 'rejected');
     if (r.accepted) box.value = '';
   }} catch (e) {{ msg.textContent = 'error: ' + e; }}
 }});
 // ── push-to-talk voice input ──────────────────────────────────────────
 // Hold the mic button -> MediaRecorder captures the mic; release (or the
 // 30 s cap) -> upload the blob to POST /asr -> the transcribed text lands
 // in the input box for the human to review before send (a mis-heard
 // command must never drive the arm unreviewed).
 const mic = document.getElementById('micbtn');
 let rec = null, chunks = [], micTimer = null;
 function micStop() {{
   clearTimeout(micTimer); micTimer = null;
   if (rec && rec.state === 'recording') rec.stop();  // onstop uploads
   mic.classList.remove('rec');
 }}
 mic.addEventListener('pointerdown', (e) => {{
   e.preventDefault();
   const msg = document.getElementById('chatmsg');
   if (rec || mic.disabled) return;
   if (!navigator.mediaDevices || !window.MediaRecorder) {{
     msg.textContent = 'voice: this browser lacks MediaRecorder';
     return;
   }}
   try {{ mic.setPointerCapture(e.pointerId); }} catch (_) {{}}
   navigator.mediaDevices.getUserMedia({{audio: {{
     echoCancellation: true,
     noiseSuppression: true,
     autoGainControl: true,
     channelCount: 1
   }}}}).then((stream) => {{
     rec = new MediaRecorder(stream);
     chunks = [];
     rec.ondataavailable = (ev) => chunks.push(ev.data);
     rec.onstop = () => {{
       stream.getTracks().forEach((t) => t.stop());
       const blob = new Blob(chunks, {{type: rec.mimeType || 'audio/webm'}});
       rec = null;
       if (blob.size < 1200) {{ msg.textContent = '(recording too short)'; return; }}
       msg.textContent = 'transcribing...';
       const fd = new FormData();
       fd.append('audio', blob, 'talk.webm');
       fetch('/asr', {{method: 'POST', body: fd}})
         .then((r) => r.json())
         .then((r) => {{
           if (r.text) {{
             const box = document.getElementById('cmd');
             box.value = r.text; box.focus();
             msg.textContent = 'voice ok - review, then press send';
           }} else {{
             msg.textContent = r.error || 'no speech recognized';
           }}
         }})
         .catch((err) => {{ msg.textContent = 'asr error: ' + err; }});
     }};
     rec.start();
     mic.classList.add('rec');
      micTimer = setTimeout(micStop, 30000);  // hard cap: 30 s
   }}).catch((err) => {{
     msg.textContent = err.name === 'NotAllowedError' ?
       'mic blocked by browser - allow the mic (LAN http may need localhost or a permissive browser)' :
       'mic error: ' + err.name;
   }});
 }});
 mic.addEventListener('pointerup', (e) => {{ e.preventDefault(); micStop(); }});
 mic.addEventListener('pointercancel', micStop);
 mic.addEventListener('contextmenu', (e) => e.preventDefault());

 // MJPEG streams freeze on the last frame when the server restarts; nudge
 // the img to reload (query param busts any cache) and grey it out meanwhile.
 document.querySelectorAll('.cam img').forEach(im => {{
   im.onerror = () => {{
     im.classList.add('dead');
     setTimeout(() => {{ im.src = im.src.split('?')[0] + '?r=' + Date.now(); }}, 2000);
   }};
   im.onload = () => im.classList.remove('dead');
 }});
 function esc(s) {{
   return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
 }}
 function fmtVec(v, digits=3) {{
   if (!Array.isArray(v)) return "--";
   return "[" + v.map(x => Number(x).toFixed(digits)).join(", ") + "]";
 }}
 function setDebugStatus(id, status) {{
   const el = document.getElementById(id);
   el.textContent = status || "--";
   el.className = status === "OK" || status === "succeeded" ? "debug-ok" :
     (status === "FAIL" || status === "failed" ? "debug-fail" : "");
 }}
 // Ages span a whole session (display history is never pruned), so fold
 // anything past two minutes into minutes to keep the column narrow.
 function fmtAge(s) {{
   const n = Number(s);
   return n <= 120 ? n.toFixed(1) + 's' : (n / 60).toFixed(1) + 'm';
 }}
 // Narration rows arrive as "[  3.2s ago] (kind) text" (episodic digest).
 // Split into age + kind badge + text so the eye scans WHAT happened
 // instead of parsing a monospace wall.
 const KINDS = {{observation:['obs','OBS'], action:['act','ACT'],
                 outcome:['out','OUT'], note:['note','NOTE']}};
 function renderFeed(lines) {{
   const feed = document.getElementById('feed');
   const stick = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 12;
   feed.innerHTML = (lines || []).map(l => {{
     const m = l.match(/^\\[\\s*([\\d.]+)s ago\\] \\((\\w+)\\) (.*)$/);
     if (!m) return `<div class="ev"><span class="ev-x ev-raw">${{esc(l)}}</span></div>`;
     const k = KINDS[m[2]] || ['note', m[2].toUpperCase().slice(0, 4)];
     return `<div class="ev"><span class="ev-t">${{fmtAge(m[1])}}</span>` +
       `<span class="ev-k k-${{k[0]}}">${{k[1]}}</span>` +
       `<span class="ev-x">${{esc(m[3])}}</span></div>`;
   }}).join('');
   if (stick) feed.scrollTop = feed.scrollHeight;
 }}
 function renderGraspDebug(s) {{
   const g = s.grasp_debug || {{}};
   setDebugStatus("g-status", g.status || "--");
   document.getElementById("g-target").textContent = g.target || "--";
   document.getElementById("g-call").textContent =
     g.grasp_call_index ? "#" + String(g.grasp_call_index) : "--";
   document.getElementById("g-source").textContent =
     g.source ? g.source + (g.branch ? "/" + g.branch : "") : "--";
   document.getElementById("g-score").textContent =
     g.grasp_score != null ? Number(g.grasp_score).toFixed(3) : "--";
   document.getElementById("g-tilt").textContent =
     g.tilt_deg != null ? Number(g.tilt_deg).toFixed(1) + " deg" : "--";
   document.getElementById("g-attempt").textContent =
     g.pick_attempt ? String(g.pick_attempt) + " / " + String(g.pick_attempt_max || g.pick_attempt) : "--";
   document.getElementById("g-candidate").textContent =
     g.candidate_index ? String(g.candidate_index) + " / " + String(g.candidate_count) : "--";
   document.getElementById("g-sequence").textContent =
     g.candidate_sequence ? String(g.candidate_sequence) : "--";
   document.getElementById("g-conf").textContent =
     g.confidence != null ? Number(g.confidence).toFixed(2) : "--";
   const widthEl = document.getElementById("g-width");
   if (g.width_m != null) {{
     const widthMm = Number(g.width_m) * 1000.0;
     const maxMm = g.max_width_m != null ? Number(g.max_width_m) * 1000.0 : null;
     const tooWide = maxMm != null && widthMm > maxMm;
     widthEl.textContent = widthMm.toFixed(1) + " mm" + (tooWide ? " \\u26a0" : "");
     widthEl.className = tooWide ? "debug-warn" : "";
   }} else {{
     widthEl.textContent = "--"; widthEl.className = "";
   }}
   document.getElementById("g-camera").textContent = fmtVec(g.camera_xyz);
   document.getElementById("g-base").textContent = fmtVec(g.base_xyz);
   document.getElementById("g-raw-origin").textContent = fmtVec(g.raw_origin_xyz);
   document.getElementById("g-tip-offset").textContent =
     g.tip_offset_m != null ? (Number(g.tip_offset_m) * 1000).toFixed(1) + " mm" : "--";
   document.getElementById("g-approach").textContent = fmtVec(g.approach);
   document.getElementById("g-jaw-axis").textContent = fmtVec(g.jaw_opening_axis);
   document.getElementById("ik-q-current").textContent = fmtVec(g.q_current);
   const pre = g.pregrasp || {{}};
   const grasp = g.grasp || {{}};
   setDebugStatus("ik-pre-status", pre.status || "--");
   setDebugStatus("ik-grasp-status", grasp.status || "--");
   const ik = (grasp.status === "FAIL" || grasp.status === "OK") ? grasp :
     ((pre.status === "FAIL" || pre.status === "OK") ? pre : null);
   document.getElementById("ik-pos-error").textContent = ik && ik.position_error_m != null ?
     (Number(ik.position_error_m) * 1000.0).toFixed(1) + " mm" : "--";
   document.getElementById("ik-rot-error").textContent = ik && ik.rotation_error_rad != null ?
     (Number(ik.rotation_error_rad) * 180.0 / Math.PI).toFixed(1) + " deg" : "--";
   document.getElementById("ik-total-error").textContent = ik && ik.total_error != null ?
     Number(ik.total_error).toFixed(5) : "--";
   document.getElementById("ik-reason").textContent =
     g.execution_error || g.reject_reason || "--";
   // Candidate history with per-status colouring: OK green / FAIL red.
   const st = x => x === "OK" ? '<b class="debug-ok">OK</b>' :
     (x === "FAIL" ? '<b class="debug-fail">FAIL</b>' : "--");
   const history = Array.isArray(g.candidates) ? g.candidates.slice(-8) : [];
   document.getElementById("g-history").innerHTML = history.length ?
     history.map(c => {{
       const width = c.width_m != null ? (Number(c.width_m) * 1000).toFixed(0) + "mm" : "--";
       const preStatus = (c.pregrasp || {{}}).status || "--";
       const graspStatus = (c.grasp || {{}}).status || "--";
       const source = (c.source || "?") + (c.branch ? "/" + c.branch : "");
       const tilt = c.tilt_deg != null ? " tilt=" + Number(c.tilt_deg).toFixed(0) + "deg" : "";
       return `<div class="cand">#${{c.sequence}} ${{esc(source)}}` +
         `<span class="dim">${{tilt}} width=${{width}}</span>` +
         ` pre=${{st(preStatus)}} grasp=${{st(graspStatus)}}` +
         (c.reject_reason ? ` <span class="rej">${{esc(c.reject_reason)}}</span>` : "") +
         `</div>`;
     }}).join('') : "--";
 }}
 function renderAgent(s) {{
   const panel = document.getElementById('agent-panel');
   if (!s || !s.agent) {{ panel.style.display = 'none'; return; }}
   panel.style.display = '';
   const a = s.agent;
   const cls = a.state === 'running' ? 'run' : (a.success ? 'ok' : 'fail');
   const meta = [a.state, a.path, a.elapsed_s != null ? a.elapsed_s + 's' : '']
     .filter(Boolean).join(' \\u00b7 ');
   document.getElementById('ag-head').innerHTML =
     `<span class="ag-dot ${{cls}}"></span>${{esc(a.task || '(no task)')}}` +
     `<div class="ag-meta" style="margin:4px 0 0 16px">${{esc(meta)}}</div>` +
     (a.summary ? `<div class="ag-meta" style="margin:4px 0 0 16px">${{esc(a.summary)}}</div>` : '');
   const ms = a.milestones || [];
   document.getElementById('ag-milestones').innerHTML = ms.length
     ? ms.map(m => `<li>${{esc(m)}}</li>`).join('')
     : '<li class="ag-dim">no decomposition (reflex/experience path, or decompose off)</li>';
   const done = (a.steps || []).map(st =>
     `<div class="ag-step ${{st.ok ? 'ok' : 'fail'}}">[${{st.step}}] ${{esc(st.tool)}}(${{esc(st.args)}})` +
     `${{st.ok ? '' : ' \\u2014 ' + esc(st.error || 'failed')}}</div>`);
   const cur = a.current;
   if (cur) done.push(
     `<div class="ag-step run">[${{cur.step}}] ${{esc(cur.tool)}}(${{esc(cur.args)}}) running\\u2026</div>`);
   document.getElementById('ag-steps').innerHTML =
     done.join('') || '<div class="ag-dim">no skill calls yet</div>';
   document.getElementById('ag-llm').textContent = a.last_llm_text || '--';
 }}

 async function tick() {{
   const conn = document.getElementById('conn');
   let s = null;
   try {{
     s = await (await fetch('/state')).json();
     conn.textContent = 'live'; conn.className = 'online';
   }} catch (e) {{
     conn.textContent = 'offline'; conn.className = 'offline';
     setTimeout(tick, 1000);  // retry faster while the server is down
     return;
   }}
   renderGraspDebug(s);
   renderAgent(s);
   renderFeed(s.events);
   // hide the mic button entirely when no ASR is wired server-side
   mic.style.display = s.voice === false ? 'none' : '';
   document.getElementById('task').textContent = s.task ? s.task : '';
   const nCams = Object.keys(s.cameras).length;
   document.getElementById('st-cams').textContent = nCams + ' CAM' + (nCams > 1 ? 'S' : '');
   document.getElementById('st-holding').textContent = 'HOLD: ' + (s.holding || '\\u2014');
   const armEl = document.getElementById('st-arm');
   armEl.textContent = s.arm_connected ? 'ARM UP' : 'ARM STANDBY';
   armEl.classList.toggle('ok', !!s.arm_connected);
   armEl.classList.toggle('warn', !s.arm_connected);
   document.getElementById('st-agent').textContent = s.agent_status || 'idle';
   // Group rows by label (freshest belief first): one physical object can
   // legitimately produce several belief rows (two cameras, drift), which
   // used to render as confusing duplicates. xN badge when it happens.
   const byLabel = {{}};
   for (const o of (s.objects || [])) {{
     const g = byLabel[o.label] || (byLabel[o.label] =
       {{fresh: o, n: 0, remembered: 0}});
     g.n += 1;
     if (o.state === 'remembered') g.remembered += 1;
   }}
   document.getElementById('objs').innerHTML = Object.values(byLabel).map(g => {{
     const o = g.fresh;
     const name = o.label + (g.n > 1 ?
       ' <span class="multi">x' + g.n + '</span>' : '');
     return `<tr class="${{o.state}}"><td>${{name}}</td>` +
       `<td><span class="chip" style="background:${{SWATCH[o.color] || '#333'}}"></span>${{o.color || '?'}}</td>` +
       `<td>[${{o.position.join(', ')}}]</td><td>${{o.state}} (${{o.age_s}}s)</td></tr>`;
   }}).join('');
   setTimeout(tick, 500);
 }}
 tick();
</script></body></html>"""


def lan_ip() -> str:
    """Best-effort LAN address for printing a clickable URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def ensure_tls_cert(cert_path, key_path, host: str | None) -> bool:
    """Generate a self-signed cert/key pair with `openssl` if missing.

    Browsers show a one-time "not secure" warning for self-signed certs --
    that is the accepted price for getting getUserMedia (the mic) on plain
    LAN addresses: Chrome/Edge only grant mic access on https:// or
    localhost. Returns True when a usable pair exists afterwards.
    """
    if cert_path.exists() and key_path.exists():
        return True
    import subprocess

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    ip = host if host and not host.isalpha() and host != "0.0.0.0" else lan_ip()
    san = f"IP:{ip},IP:127.0.0.1,DNS:localhost"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-days", "3650", "-keyout", str(key_path), "-out", str(cert_path),
        "-subj", "/CN=wrc-demo livestream",
        "-addext", f"subjectAltName={san}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except FileNotFoundError:
        print("[stream] openssl not found -- cannot make a TLS cert; "
              "serving plain HTTP (browser mic needs https or localhost)",
              file=sys.stderr)
        return False
    except subprocess.SubprocessError as e:
        print(f"[stream] TLS cert generation failed ({e}); serving "
              f"plain HTTP", file=sys.stderr)
        return False
    print(f"[stream] self-signed TLS cert written to {cert_path}", file=sys.stderr)
    return True


class StreamServer:
    """HTTP front-end over a CameraRig + a state callback."""

    def __init__(
        self,
        rig,
        state_fn=None,
        host: str = "0.0.0.0",
        advertise_host: str | None = None,
        port: int = 8090,
        fps: float = 15.0,
        quality: int = 80,
        tls: bool = False,
        tls_cert=None,
        tls_key=None,
    ):
        self._rig = rig
        self._state_fn = state_fn or (lambda: {})
        self._host = host
        # Binding and advertising are separate concerns: bind to 0.0.0.0 so
        # every LAN interface can serve, but print a client-reachable address
        # instead of whichever VPN/container interface socket heuristics pick.
        self._advertise_host = advertise_host
        self.port = port
        self._fps = fps
        self._quality = quality
        # TLS (self-signed) exists so browsers grant mic access on LAN
        # addresses; degrades to plain HTTP if a cert cannot be made.
        self._tls = bool(tls)
        self._tls_cert = tls_cert
        self._tls_key = tls_key
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        # One JPEG encode per camera per frame regardless of viewer count:
        # cache keyed by (frame id, overlay status) -- N viewers of the same
        # stream share the encode instead of costing O(viewers x fps).
        self._jpeg_cache: dict[str, tuple[tuple, bytes]] = {}
        self._jpeg_lock = threading.Lock()
        # Optional chat: POST /task hands a natural-language command to this
        # callback (the app wires it to its orchestrator). One at a time.
        self._task_fn = None
        self._cancel_fn = None
        self._task_busy = threading.Lock()
        # Optional voice input: POST /asr hands a recorded audio blob to
        # this callback (demo wires faster-whisper). None -> 501 + the
        # dashboard hides the mic button (state() reports voice=False).
        self._asr_fn = None

    def set_task_fn(self, fn) -> None:
        """fn(task_text) runs a command; called from a worker thread."""
        self._task_fn = fn

    def set_cancel_fn(self, fn) -> None:
        """fn() aborts the running command (POST /cancel, the STOP button)."""
        self._cancel_fn = fn

    def set_asr_fn(self, fn) -> None:
        """fn(audio_bytes, mime) -> str transcribes a mic recording."""
        self._asr_fn = fn

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        handler = _make_handler(self)
        self._httpd = _QuietThreadingHTTPServer((self._host, self.port), handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # resolves port=0
        if self._tls:
            from pathlib import Path

            cert = (Path(self._tls_cert) if self._tls_cert
                    else Path.home() / ".wrc_demo" / "stream_tls_cert.pem")
            key = (Path(self._tls_key) if self._tls_key
                   else cert.parent / "stream_tls_key.pem")
            if ensure_tls_cert(cert, key, self._advertise_host):
                import ssl

                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(str(cert), str(key))
                self._httpd.socket = ctx.wrap_socket(
                    self._httpd.socket, server_side=True)
            else:
                self._tls = False  # degrade, don't die
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="stream-server"
        )
        self._thread.start()
        print(f"[stream] live view on {self.url}", file=sys.stderr)
        if self._tls:
            print("[stream] self-signed TLS: accept the browser's certificate "
                  "warning once (Advanced -> proceed)", file=sys.stderr)

    def stop(self) -> None:
        self._stopping = True  # MJPEG handler loops watch this and exit
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        host = self._advertise_host or lan_ip()
        scheme = "https" if self._tls else "http"
        return f"{scheme}://{host}:{self.port}/"

    # ── rendering ────────────────────────────────────────────────────────

    def annotated_jpeg(self, name: str) -> bytes | None:
        stream = self._rig.get(name)
        frame = stream.latest()
        if frame is None:
            return None
        dets, status = stream.overlay()
        projection_state = {}
        try:
            projection_state = self._state_fn() or {}
        except Exception:
            pass
        grasp_debug = projection_state.get("grasp_debug", {})
        key = (frame.frame_id, len(dets), status, grasp_debug.get("updated_at"))
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(name)
            if cached is not None and cached[0] == key:
                return cached[1]
        img = frame.rgb.copy()
        try:
            draw_detections(img, dets)
        except Exception:
            pass
        projection = projection_state.get("grasp_projection", {})
        if projection.get("camera") in (None, name):
            try:
                draw_grasp_pose(img, frame.K, projection["T_base_cam"], grasp_debug)
            except Exception:
                pass
        h, w = img.shape[:2]
        draw_hud(img, [
            f"{stream.name}  {w}x{h}  {stream.fps:4.1f} fps  depth: {frame.depth_source}",
            f"agent: {status}",
        ])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return None
        jpeg = buf.tobytes()
        with self._jpeg_lock:
            self._jpeg_cache[name] = (key, jpeg)
        return jpeg

    def state(self) -> dict:
        cams = self._rig.stats()
        base = {"cameras": cams, "t": time.time()}
        try:
            base.update(self._state_fn() or {})
        except Exception as e:
            base["state_error"] = str(e)
        base.setdefault("agent_status", "idle")
        # Lets the dashboard hide the mic button when no ASR is wired
        # (faster-whisper is an optional post-install dependency).
        base["voice"] = self._asr_fn is not None
        return base


def _make_handler(server: StreamServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # stderr chatter only on errors
            pass

        # ── routes ───────────────────────────────────────────────────────

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                if path == "/":
                    return self._index()
                if path == "/state":
                    return self._json(server.state())
                if path.startswith("/snapshot/"):
                    name = path.split("/", 2)[2].removesuffix(".jpg")
                    return self._snapshot(name)
                if path.startswith("/stream/"):
                    return self._mjpeg(path.split("/", 2)[2])
                self.send_error(404, "unknown path")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab mid-frame
            except KeyError as e:
                self.send_error(404, str(e))

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            # API endpoints always answer JSON (even on error) so the
            # dashboard's fetch().json() never chokes on an HTML error page.
            if self.path.rstrip("/") == "/cancel":
                if server._cancel_fn is None:
                    return self._json({"cancelled": False,
                                       "error": "no cancel handler wired"}, code=501)
                try:
                    server._cancel_fn()
                    return self._json({"cancelled": True})
                except Exception as e:
                    return self._json({"cancelled": False, "error": str(e)[:120]})
            if self.path.rstrip("/") == "/asr":
                return self._asr()
            if self.path.rstrip("/") != "/task":
                return self._json({"error": "unknown path"}, code=404)
            if server._task_fn is None:
                return self._json({"accepted": False,
                                   "error": "no task handler wired (REPL mode drives tasks)"},
                                  code=501)
            try:
                n = int(self.headers.get("Content-Length", 0))
                task = str(json.loads(self.rfile.read(n)).get("task", "")).strip()
            except (ValueError, json.JSONDecodeError):
                return self._json({"accepted": False,
                                   "error": 'body must be JSON {"task": ...}'}, code=400)
            if not task:
                return self._json({"accepted": False, "error": "empty task"}, code=400)
            if not server._task_busy.acquire(blocking=False):
                return self._json({"accepted": False, "error": "busy with another command"})

            def _run():
                try:
                    server._task_fn(task)
                except Exception as e:
                    print(f"[stream] task failed: {e}", file=sys.stderr)
                finally:
                    server._task_busy.release()

            threading.Thread(target=_run, daemon=True, name="dashboard-task").start()
            self._json({"accepted": True, "task": task})

        def _asr(self):
            """POST /asr -- multipart audio blob -> {"text": "..."}.

            The browser records with MediaRecorder and uploads the blob
            untouched; the wired fn (faster-whisper, demo.py) decodes it.
            """
            if server._asr_fn is None:
                return self._json(
                    {"error": "voice input not configured (voice.enabled in "
                              "demo.yaml + faster-whisper installed?)"},
                    code=501,
                )
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                return self._json({"error": "body must be multipart/form-data"},
                                  code=400)
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            if n <= 0 or n > 8 * 1024 * 1024:  # ~80 s of opus; PTT caps at 30 s
                return self._json({"error": "audio too large or empty"}, code=400)
            body = self.rfile.read(n)
            audio, mime = _extract_multipart_audio(ctype, body)
            if audio is None:
                return self._json({"error": "no audio part found"}, code=400)
            try:
                text = server._asr_fn(audio, mime)
            except Exception as e:
                return self._json({"error": str(e)[:200]}, code=500)
            return self._json({"text": text or ""})

        def _index(self):
            tiles = "".join(
                f'<div class="cam"><h2><span class="camdot"></span>{n}</h2>'
                f'<div class="frame"><img src="/stream/{n}" alt="{n}">'
                f'<i class="c c-tl"></i><i class="c c-tr"></i>'
                f'<i class="c c-bl"></i><i class="c c-br"></i></div></div>'
                for n in server._rig.names
            )
            body = _INDEX_HTML.format(tiles=tiles).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200):
            body = json.dumps(payload, default=_np_safe).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _snapshot(self, name: str):
            jpeg = server.annotated_jpeg(name)
            if jpeg is None:
                return self.send_error(503, "no frame yet")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        def _mjpeg(self, name: str):
            server._rig.get(name)  # 404 (KeyError) before headers go out
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            period = 1.0 / max(server._fps, 1.0)
            while not server._stopping:
                t0 = time.monotonic()
                jpeg = server.annotated_jpeg(name)
                if jpeg is not None:
                    self.wfile.write(
                        b"--" + _BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)

    return Handler


def _np_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _extract_multipart_audio(content_type: str, body: bytes):
    """Pull the first file part (name="audio") out of a multipart body.

    stdlib-only via email.parser -- FormData from the browser sends exactly
    one file part, so "first part with a filename" is the recording.
    Returns (audio_bytes, mime) or (None, None).
    """
    from email.parser import BytesParser
    from email.policy import HTTP

    try:
        msg = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + content_type.encode("latin-1", "replace")
            + b"\r\n\r\n" + body
        )
        for part in msg.iter_parts():
            if part.get_filename():
                payload = part.get_payload(decode=True) or b""
                mime = part.get_content_type()
                return payload, mime
    except Exception:
        pass
    return None, None
