import re, sys
from PIL import Image, ImageDraw, ImageFont
SGR=re.compile(r'\x1b\[([0-9;]*)m'); DEF=(201,209,217)
PAL={32:(63,185,80),33:(210,153,34),31:(248,81,73),36:(57,197,207),90:(110,118,129)}
BRIGHT={32:(86,211,100),31:(255,123,114)}
WRAP=100
def parse(line):
    parts=[]; pos=0; bold=False; fg=DEF
    for m in SGR.finditer(line):
        if m.start()>pos: parts.append((line[pos:m.start()],fg))
        for c in (int(x) if x else 0 for x in m.group(1).split(';')):
            if c==0: bold=False; fg=DEF
            elif c==1: bold=True
            elif c in PAL: fg=BRIGHT.get(c) if (bold and c in BRIGHT) else PAL[c]
        pos=m.end()
    if pos<len(line): parts.append((line[pos:],fg))
    return parts
def wrap(parts):
    chars=[(ch,col) for t,col in parts for ch in t]; out=[]; cur=[]
    for ch,col in chars:
        cur.append((ch,col))
        if len(cur)>=WRAP:
            sp=max((i for i,(c,_) in enumerate(cur) if c==' '), default=-1)
            if sp>WRAP*0.55: out.append(cur[:sp]); cur=cur[sp+1:]
            else: out.append(cur); cur=[]
    out.append(cur); return out or [[]]
src,out,title=sys.argv[1],sys.argv[2],sys.argv[3]
raw=open(src,encoding='utf-8',errors='replace').read().replace('^D','')
raw=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]','',raw).replace('\r','').replace('\ufe0f','').replace('\ufe0e','')
logical=[l for l in raw.split('\n')]
while logical and logical[-1].strip()=='': logical.pop()
while logical and logical[0].strip()=='': logical.pop(0)
vis=[]
for l in logical:
    for vl in wrap(parse(l)): vis.append(vl)
SZ=24
font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',SZ,index=0); lh=int(SZ*1.5)
def w(line): return font.getlength(''.join(c for c,_ in line))
maxw=max((w(l) for l in vis), default=10); PAD=28; TITLE=44
W=int(maxw+2*PAD); H=int(TITLE+len(vis)*lh+2*PAD)
img=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(img)
d.rounded_rectangle([0,0,W-1,H-1],radius=14,fill=(13,17,23,255))
d.rounded_rectangle([0,0,W-1,TITLE],radius=14,fill=(22,27,34,255)); d.rectangle([0,TITLE-14,W-1,TITLE],fill=(22,27,34,255))
for i,c in enumerate([(255,95,86),(255,189,46),(39,201,63)]): d.ellipse([20+i*22,TITLE//2-7,34+i*22,TITLE//2+7],fill=c)
d.text((W/2,TITLE/2),title,font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',17,index=0),fill=(139,148,158),anchor='mm')
y=TITLE+PAD
for line in vis:
    x=PAD; i=0
    while i<len(line):
        col=line[i][1]; j=i
        while j<len(line) and line[j][1]==col: j+=1
        seg=''.join(c for c,_ in line[i:j]); d.text((x,y),seg,font=font,fill=col+(255,)); x+=font.getlength(seg); i=j
    y+=lh
img.save(out); print('wrote',out.split('/')[-1],img.size)
