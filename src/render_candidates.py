"""Render each hemistich as K seed-candidates (separate files), gold pipeline + BigVGAN-v2.
Pick the clean one per hemistich by ear, then stitch with stitch_hemis.py."""
import os, sys, glob, json, argparse, numpy as np, soundfile as sf, torch
PROD="<PROD>"; sys.path.insert(0, PROD)
import prep_text as PT, bigvgan
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, preprocess_ref_audio_text
from f5_tts.model import DiT
CHAMP=f"{PROD}/CHAMPION_2026-06-11"
ap=argparse.ArgumentParser()
ap.add_argument("--ref_wav",required=True); ap.add_argument("--ref_text_file",required=True)
ap.add_argument("--padas",required=True); ap.add_argument("--outdir",required=True)
ap.add_argument("--tag",required=True); ap.add_argument("--K",type=int,default=6)
ap.add_argument("--speed",type=float,default=0.90); ap.add_argument("--nfe",type=int,default=64)
a=ap.parse_args(); SR=24000; os.makedirs(a.outdir,exist_ok=True)
CFG=dict(dim=1024,depth=22,heads=16,ff_mult=2,text_dim=512,conv_layers=4)
vocab=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"))[0]
cfm=load_model(DiT,CFG,mel_spec_type="vocos",vocab_file=vocab,device="cuda")
ck=torch.load(f"{CHAMP}/voice_armA_ema_2026-06-11.pt",map_location="cpu",weights_only=True)
ema={k.replace("ema_model.",""):v for k,v in ck["ema_model_state_dict"].items() if k not in("initted","step")}
cfm.load_state_dict(ema,strict=False); cfm.eval()
real=load_vocoder("vocos")
class Cap:
    def __init__(s,r): s.r=r; s.last=None
    def decode(s,m): s.last=m.detach().cpu().numpy(); return s.r.decode(m)
cap=Cap(real)
g=bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x",use_cuda_kernel=False)
bsd=torch.load(f"{CHAMP}/voc_bigvgan_EMA_2026-06-11.pth",map_location="cpu"); bsd=bsd.get("model",bsd)
g.load_state_dict(bsd); g.remove_weight_norm(); g=g.cuda().eval()
for p in g.parameters(): p.requires_grad=False
def bvgan(mel):
    m=torch.from_numpy(mel).cuda()
    with torch.no_grad():
        if m.dim()==3 and m.shape[1]!=100 and m.shape[2]==100: m=m.transpose(1,2)
        return g(m).squeeze().cpu().numpy().astype(np.float32)
def gate(au,voice=0.08,sil=0.012,fin=0.015,fout=0.040,lead=0.03,keep=0.06):
    win=int(0.02*SR); r=[float(np.sqrt((au[i:i+win]**2).mean())) for i in range(0,len(au)-win,win)]; n=len(r)
    if n==0: return au
    vs=next((i for i in range(n-1) if r[i]>voice and r[i+1]>sil),int(np.argmax(r))); s=vs
    while s>0 and r[s-1]>sil: s-=1
    start=max(0,s*win-int(lead*SR)); ve=max((i for i in range(n) if r[i]>0.035),default=vs)
    end=min(len(au),ve*win+int(keep*SR)); out=au[start:end].copy()
    fi,fo=int(fin*SR),int(fout*SR)
    if fi and len(out)>fi: out[:fi]*=np.linspace(0,1,fi)
    if fo and len(out)>fo: out[-fo:]*=(np.cos(np.linspace(0,np.pi,fo))*0.5+0.5)
    return out
ref_text=open(a.ref_text_file,encoding="utf-8").read().strip()
ref_audio,ref_t=preprocess_ref_audio_text(a.ref_wav,ref_text)
PIECES=[PT.model_text(p) for p in json.load(open(a.padas))]
for i,p in enumerate(PIECES):
    for k in range(a.K):
        torch.manual_seed(7000+i*131+k*17)
        w,sr,_=infer_process(ref_audio,ref_t,p,cfm,cap,mel_spec_type="vocos",speed=a.speed,nfe_step=a.nfe,cfg_strength=1.2,device="cuda")
        w=np.array(w,dtype=np.float32)
        if np.abs(w).max()>1.5: w=w/32768.0
        y=bvgan(cap.last); mx=np.abs(y).max(); y=y/mx*0.97 if mx>1 else y
        fn=f"{a.outdir}/{a.tag}_hemi{i+1}_s{k}.wav"; sf.write(fn,gate(y),SR)
        print(f"hemi{i+1} s{k}: {len(y)/SR:.1f}s -> {os.path.basename(fn)}",flush=True)
print("CANDIDATES DONE",flush=True)
