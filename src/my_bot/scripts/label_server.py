#!/usr/bin/env python3
"""Hand-correct YOLO label drafts in a browser -- no display on the Jetson.

Objective 2 needs every draft from `detection_accuracy.py prelabel` checked by
a person. labelImg needs a screen; this serves the same job over HTTP, so the
labelling happens in the browser of the laptop you SSH from:

    # on the Jetson
    python3 label_server.py --data ~/eval/insitu2_stationary
    # on the laptop
    ssh -L 8765:localhost:8765 mic-711@<jetson>
    # then open http://localhost:8765

It reads and writes <data>/labels_draft/*.txt in place (YOLO format, class
order from labels_draft/classes.txt), so the plan's next step is unchanged:
`mv labels_draft/* labels/` once every frame is reviewed, then `score`.

Bound to 127.0.0.1 by default: the frames show people, and the SSH tunnel is
the access control. `--host 0.0.0.0` only on a network you trust.

Which frames have been reviewed is kept in <data>/labels_draft/reviewed.json,
so a session can stop and resume. A frame counts as reviewed when it is saved
from the page -- which the page does on every move to another frame.

The deployed model's predictions are deliberately NOT shown: labelling while
looking at the detector under test biases the truth toward it.

Stdlib only.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME_RE = re.compile(r"^[0-9]{6}$")


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def atomic_write(path, text):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class Store:
    def __init__(self, data, labels):
        self.frames_dir = os.path.join(data, "frames")
        self.labels_dir = os.path.join(data, labels)
        if not os.path.isdir(self.frames_dir):
            die("no frames/ in %s -- run extract first" % data)
        if not os.path.isdir(self.labels_dir):
            die("no %s/ in %s -- run prelabel first" % (labels, data))
        cf = os.path.join(self.labels_dir, "classes.txt")
        if not os.path.exists(cf):
            die("no classes.txt in %s" % self.labels_dir)
        with open(cf) as fh:
            self.classes = [l.strip() for l in fh if l.strip()]
        self.frames = sorted(f[:-4] for f in os.listdir(self.frames_dir)
                             if f.endswith(".jpg") and NAME_RE.match(f[:-4]))
        if not self.frames:
            die("no frames in %s" % self.frames_dir)
        self.reviewed_path = os.path.join(self.labels_dir, "reviewed.json")
        self.reviewed = set()
        if os.path.exists(self.reviewed_path):
            with open(self.reviewed_path) as fh:
                self.reviewed = set(json.load(fh))

    def read(self, name):
        path = os.path.join(self.labels_dir, name + ".txt")
        boxes = []
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) >= 5:
                        boxes.append([int(float(p[0]))] + [float(v) for v in p[1:5]])
        return boxes

    def write(self, name, boxes):
        lines = []
        for b in boxes:
            c, cx, cy, w, h = int(b[0]), *(float(v) for v in b[1:5])
            if not 0 <= c < len(self.classes):
                raise ValueError("class index %d out of range" % c)
            if w <= 0 or h <= 0:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (c, cx, cy, w, h))
        # An empty file is meaningful: "looked, nothing there". Always write it.
        atomic_write(os.path.join(self.labels_dir, name + ".txt"),
                     "\n".join(lines) + ("\n" if lines else ""))
        self.reviewed.add(name)
        atomic_write(self.reviewed_path, json.dumps(sorted(self.reviewed)))


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/meta":
                return self.json({"classes": store.classes, "frames": store.frames,
                                  "reviewed": sorted(store.reviewed)})
            m = re.match(r"^/frames/([0-9]{6})\.jpg$", p)
            if m and m.group(1) in store.frames:
                with open(os.path.join(store.frames_dir, m.group(1) + ".jpg"), "rb") as fh:
                    return self.send(200, fh.read(), "image/jpeg")
            m = re.match(r"^/api/labels/([0-9]{6})$", p)
            if m and m.group(1) in store.frames:
                return self.json({"boxes": store.read(m.group(1))})
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            m = re.match(r"^/api/labels/([0-9]{6})$", self.path)
            if not (m and m.group(1) in store.frames):
                return self.send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                boxes = json.loads(self.rfile.read(n))["boxes"]
                store.write(m.group(1), boxes)
            except (ValueError, KeyError, TypeError) as e:
                return self.json({"error": str(e)}, 400)
            self.json({"ok": True, "reviewed": len(store.reviewed)})

    return H


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Label review</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#1b1d21;color:#e6e6e6}
#bar{display:flex;gap:14px;align-items:center;padding:8px 12px;background:#26292f;flex-wrap:wrap}
#bar b{color:#fff} .cls{padding:2px 8px;border-radius:4px;cursor:pointer;border:2px solid transparent}
.cls.on{border-color:#fff} #wrap{padding:10px;overflow:auto}
canvas{display:block;cursor:crosshair;background:#000}
#help{padding:4px 12px 10px;color:#aab;font-size:13px}
#msg{color:#ffb454} button{font:inherit}
.done{color:#7ee787}
</style></head><body>
<div id="bar">
  <span>frame <b id="fname">-</b> (<span id="idx"></span>)</span>
  <span id="rev"></span>
  <span id="classes"></span>
  <label>zoom <select id="zoom"><option>1</option><option selected>1.5</option><option>2</option><option>2.5</option></select></label>
  <button id="nextun">next unreviewed</button>
  <span id="msg"></span>
</div>
<div id="help">
<b>1–4</b> class for new boxes (or change the selected box) · <b>drag on empty</b> = new box ·
<b>click</b> = select · drag inside = move · drag a corner/edge = resize ·
<b>Del/Backspace</b> = delete · <b>d / →</b> save + next · <b>a / ←</b> save + previous ·
<b>s</b> save · <b>h</b> hold to hide boxes · <b>Esc</b> deselect.
Moving to another frame saves this one and marks it reviewed.
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const COLORS=["#ff5c5c","#4da3ff","#ffd33d","#56d364","#d2a8ff","#ff9e64"];
const HANDLE=7;
let meta, idx=0, boxes=[], sel=-1, cur=0, img=new Image(), scale=1.5, hide=false, dirty=false;
let drag=null;
const cv=document.getElementById("c"), ctx=cv.getContext("2d");
const $=id=>document.getElementById(id);

function toXYXY(b){const[c,cx,cy,w,h]=b,W=img.naturalWidth,H=img.naturalHeight;
  return {c,x1:(cx-w/2)*W,y1:(cy-h/2)*H,x2:(cx+w/2)*W,y2:(cy+h/2)*H};}
function fromXYXY(o){const W=img.naturalWidth,H=img.naturalHeight;
  const x1=Math.max(0,Math.min(o.x1,o.x2)),x2=Math.min(W,Math.max(o.x1,o.x2));
  const y1=Math.max(0,Math.min(o.y1,o.y2)),y2=Math.min(H,Math.max(o.y1,o.y2));
  return {c:o.c,x1,y1,x2,y2};}
function toYolo(o){const W=img.naturalWidth,H=img.naturalHeight;
  return [o.c,(o.x1+o.x2)/2/W,(o.y1+o.y2)/2/H,(o.x2-o.x1)/W,(o.y2-o.y1)/H];}

function draw(){
  cv.width=img.naturalWidth*scale; cv.height=img.naturalHeight*scale;
  ctx.drawImage(img,0,0,cv.width,cv.height);
  if(hide) return;
  boxes.forEach((o,i)=>{
    const col=COLORS[o.c%COLORS.length];
    ctx.lineWidth=i===sel?3:2; ctx.strokeStyle=col;
    ctx.strokeRect(o.x1*scale,o.y1*scale,(o.x2-o.x1)*scale,(o.y2-o.y1)*scale);
    ctx.font="bold 13px system-ui"; const t=meta.classes[o.c];
    const tw=ctx.measureText(t).width+6;
    ctx.fillStyle=col; ctx.fillRect(o.x1*scale,o.y1*scale-16,tw,16);
    ctx.fillStyle="#000"; ctx.fillText(t,o.x1*scale+3,o.y1*scale-4);
    if(i===sel){ctx.fillStyle=col;
      for(const[x,y]of corners(o))ctx.fillRect(x*scale-HANDLE/2,y*scale-HANDLE/2,HANDLE,HANDLE);}
  });
  if(drag&&drag.kind==="new"){const o=drag.box;ctx.setLineDash([5,4]);ctx.strokeStyle=COLORS[o.c];
    ctx.strokeRect(o.x1*scale,o.y1*scale,(o.x2-o.x1)*scale,(o.y2-o.y1)*scale);ctx.setLineDash([]);}
}
function corners(o){return[[o.x1,o.y1],[o.x2,o.y1],[o.x1,o.y2],[o.x2,o.y2]];}

function status(){
  $("fname").textContent=meta.frames[idx]; $("idx").textContent=(idx+1)+" / "+meta.frames.length;
  const n=meta.reviewed.size, N=meta.frames.length;
  $("rev").innerHTML=(n===N?'<span class="done">':'')+"reviewed "+n+" / "+N+(n===N?" ✓ all done</span>":"")+
    (meta.reviewed.has(meta.frames[idx])?" · this one ✓":" · this one not yet");
  document.querySelectorAll(".cls").forEach((e,i)=>e.classList.toggle("on",i===cur));
}

async function load(i){
  idx=Math.max(0,Math.min(meta.frames.length-1,i)); sel=-1; dirty=false;
  const name=meta.frames[idx];
  const r=await fetch("/api/labels/"+name); const j=await r.json();
  await new Promise(res=>{img.onload=res; img.src="/frames/"+name+".jpg";});
  boxes=j.boxes.map(toXYXY); status(); draw();
}
async function save(){
  const name=meta.frames[idx];
  const kept=boxes.map(fromXYXY).filter(o=>(o.x2-o.x1)>=2&&(o.y2-o.y1)>=2);
  const r=await fetch("/api/labels/"+name,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({boxes:kept.map(toYolo)})});
  if(!r.ok){$("msg").textContent="SAVE FAILED for "+name+": "+(await r.text());return false;}
  meta.reviewed.add(name); dirty=false; $("msg").textContent="saved "+name; status(); return true;
}
async function go(d){ if(await save()) await load(idx+d); }

function pos(e){const r=cv.getBoundingClientRect();return{x:(e.clientX-r.left)/scale,y:(e.clientY-r.top)/scale};}
function hitHandle(o,p){const t=HANDLE/scale+2;const res={};
  if(Math.abs(p.x-o.x1)<t)res.l=1; if(Math.abs(p.x-o.x2)<t)res.r=1;
  if(Math.abs(p.y-o.y1)<t)res.t=1; if(Math.abs(p.y-o.y2)<t)res.b=1;
  const inX=p.x>o.x1-t&&p.x<o.x2+t, inY=p.y>o.y1-t&&p.y<o.y2+t;
  if(!(inX&&inY))return null; return Object.keys(res).length?res:null;}
function inside(o,p){return p.x>=o.x1&&p.x<=o.x2&&p.y>=o.y1&&p.y<=o.y2;}
function area(o){return(o.x2-o.x1)*(o.y2-o.y1);}

cv.addEventListener("mousedown",e=>{
  const p=pos(e);
  if(sel>=0){const h=hitHandle(boxes[sel],p);
    if(h){drag={kind:"resize",h,start:p,orig:{...boxes[sel]}};return;}}
  // smallest box under the cursor, so a chair inside a person box is reachable
  let best=-1; boxes.forEach((o,i)=>{if(inside(o,p)&&(best<0||area(o)<area(boxes[best])))best=i;});
  if(best>=0){sel=best;drag={kind:"move",start:p,orig:{...boxes[sel]}};draw();return;}
  sel=-1; drag={kind:"new",box:{c:cur,x1:p.x,y1:p.y,x2:p.x,y2:p.y}}; draw();
});
window.addEventListener("mousemove",e=>{
  if(!drag)return; const p=pos(e), dx=p.x-(drag.start?drag.start.x:0), dy=p.y-(drag.start?drag.start.y:0);
  if(drag.kind==="new"){drag.box.x2=p.x;drag.box.y2=p.y;}
  else if(drag.kind==="move"){const o=drag.orig;Object.assign(boxes[sel],{x1:o.x1+dx,x2:o.x2+dx,y1:o.y1+dy,y2:o.y2+dy});dirty=true;}
  else if(drag.kind==="resize"){const o=drag.orig,h=drag.h,b=boxes[sel];
    if(h.l)b.x1=o.x1+dx; if(h.r)b.x2=o.x2+dx; if(h.t)b.y1=o.y1+dy; if(h.b)b.y2=o.y2+dy; dirty=true;}
  draw();
});
window.addEventListener("mouseup",()=>{
  if(!drag)return;
  if(drag.kind==="new"){const o=fromXYXY(drag.box);
    if(o.x2-o.x1>=3&&o.y2-o.y1>=3){boxes.push(o);sel=boxes.length-1;dirty=true;}}
  else if(sel>=0){boxes[sel]=fromXYXY(boxes[sel]);}
  drag=null; draw();
});
window.addEventListener("keydown",e=>{
  if(e.target.tagName==="SELECT")return;
  const k=e.key;
  if(k>="1"&&k<=String(meta.classes.length)){const c=+k-1;
    if(sel>=0){boxes[sel].c=c;dirty=true;} cur=c; status(); draw(); return;}
  if(k==="Delete"||k==="Backspace"){if(sel>=0){boxes.splice(sel,1);sel=-1;dirty=true;draw();}e.preventDefault();return;}
  if(k==="d"||k==="ArrowRight"){go(1);e.preventDefault();return;}
  if(k==="a"||k==="ArrowLeft"){go(-1);e.preventDefault();return;}
  if(k==="s"){save();return;}
  if(k==="h"&&!hide){hide=true;draw();return;}
  if(k==="Escape"){sel=-1;draw();}
});
window.addEventListener("keyup",e=>{if(e.key==="h"){hide=false;draw();}});
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="";}});
$("zoom").onchange=e=>{scale=+e.target.value;draw();};
$("nextun").onclick=async()=>{if(!await save())return;
  const N=meta.frames.length;
  for(let k=1;k<=N;k++){const j=(idx+k)%N;if(!meta.reviewed.has(meta.frames[j])){await load(j);return;}}
  $("msg").textContent="every frame is reviewed";};

(async()=>{
  meta=await (await fetch("/api/meta")).json(); meta.reviewed=new Set(meta.reviewed);
  $("classes").innerHTML=meta.classes.map((c,i)=>
    `<span class="cls" style="background:${COLORS[i%COLORS.length]};color:#000" data-i="${i}">${i+1} ${c}</span>`).join(" ");
  document.querySelectorAll(".cls").forEach(el=>el.onclick=()=>{cur=+el.dataset.i;
    if(sel>=0){boxes[sel].c=cur;dirty=true;draw();} status();});
  const first=meta.frames.findIndex(f=>!meta.reviewed.has(f));
  await load(first<0?0:first);
})();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="dataset dir from `extract`")
    ap.add_argument("--labels", default="labels_draft",
                    help="subdirectory to read and write (default labels_draft)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    store = Store(os.path.abspath(os.path.expanduser(args.data)), args.labels)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print("%d frames, %d reviewed, classes: %s"
          % (len(store.frames), len(store.reviewed), " ".join(store.classes)))
    print("serving on http://%s:%d  -- from the laptop:" % (args.host, args.port))
    print("    ssh -L %d:localhost:%d %s@<jetson>   then open http://localhost:%d"
          % (args.port, args.port, os.environ.get("USER", "mic-711"), args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. reviewed %d / %d" % (len(store.reviewed), len(store.frames)))


if __name__ == "__main__":
    main()
