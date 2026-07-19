from __future__ import annotations

import hashlib, json, os, random, subprocess, time
from pathlib import Path
import numpy as np

import comfy.model_management as model_management
from comfy.utils import ProgressBar

from ...engine.video_swap_engine import CMKVideoSwapEngine
from ...engine.content_guard import ContentGuardBlocked, GUARD_VERSION
from ...engine.swap_selected_engine import SelectedSwapSettings
from ...engine.enhance_backends import get_available_enhancer_modes, validate_enhancer_mode
from ...models.model_manager import list_detector_models, list_swap_models, resolve_swap_model
from ...pipe.cmk_log_pipe import cmk_add_block
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.tensor_utils import tensor_to_uint8_rgb
from .split_video_segments import _executable, _output_root, _probe_video

_SELECTION_MODES=["Largest","Leftmost","Rightmost","Topmost","Bottommost","Center"]
_SEGMENT_MODES=["Full Path (Batch)","Randomize","Last Used Segment"]
_SCHEMA="cmk.video.face_swap.v1"; _ENGINE_VERSION=7

def _hash_json(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def _atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8"); os.replace(tmp,path)
def _file_id(path):
    st=Path(path).stat(); return {"path":str(Path(path).resolve()),"size":st.st_size,"mtime_ns":st.st_mtime_ns}
def _image_hash(rgb): return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
def _blend(a,b,x): return b if x>=.999 else np.clip(a.astype(np.float32)*(1-x)+b.astype(np.float32)*x,0,255).astype(np.uint8)


def _selection_state_path(segs):
    source_path=Path(str(segs.get("source_path","video"))).resolve()
    key=_hash_json({"source_path":str(source_path),"manifest_path":str(segs.get("manifest_path",""))})[:20]
    root=_output_root()/"video"/"face_swap"/"_selection_state"
    root.mkdir(parents=True,exist_ok=True)
    return root/f"{key}.json"

def _load_last_used_index(segs):
    path=_selection_state_path(segs)
    if not path.exists():
        return None
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
        return int(data["last_used_index"])
    except Exception:
        return None

def _store_last_used_index(segs,index):
    _atomic_json(_selection_state_path(segs),{
        "type":"CMK_VIDEO_FACE_SWAP_SELECTION_STATE",
        "version":1,
        "source_path":str(Path(str(segs.get("source_path",""))).resolve()),
        "manifest_path":str(segs.get("manifest_path","")),
        "last_used_index":int(index),
        "updated_at":time.time(),
    })

class CMKFaceSwapVideo:
    @classmethod
    def IS_CHANGED(cls, **kw):
        enabled=bool(kw.get("FACE SWAP", True))
        if not enabled:
            return None
        mode=str(kw.get("SEGMENT SELECTION",""))
        if mode=="Randomize":
            return float("nan")
        return None
    CATEGORY="CMK/Toolbox/Video"; RETURN_TYPES=("CMK_VIDEO_SEGMENTS","CMK_LOG_PIPE","CMK_DIAGNOSTIC"); RETURN_NAMES=("SEGMENTS","LOG","diagnostic"); FUNCTION="run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "SEGMENTS":("CMK_VIDEO_SEGMENTS",), "IMAGE SOURCE":("IMAGE",), "LOG":("CMK_LOG_PIPE",),
            "FACE SWAP":("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF"}),
            "SEGMENT SELECTION":(_SEGMENT_MODES,{"default":"Full Path (Batch)"}),
            "SWAP MODEL":(list_swap_models(),), "DETECT MODEL":(list_detector_models(),),
            "FACE ENHANCER":(get_available_enhancer_modes(),{"default":"Off"}), "TARGET FACE":(_SELECTION_MODES,{"default":"Largest"}),
            "SOURCE FACE":(_SELECTION_MODES,{"default":"Largest"}), "BLEND":("FLOAT",{"default":1.0,"min":0.0,"max":1.0,"step":0.05}),
            "bbox_dilation":("INT",{"default":0,"min":-512,"max":512,"step":1,"advanced":True}),
            "crop_factor":("FLOAT",{"default":1.5,"min":1.0,"max":3.0,"step":0.1,"advanced":True}),
            "drop_size":("INT",{"default":10,"min":1,"max":8192,"step":1,"advanced":True}),
            "feather":("INT",{"default":0,"min":0,"max":100,"step":1,"advanced":True}),
            "max_missing_frames":("INT",{"default":12,"min":0,"max":300,"step":1,"advanced":True}),
            "tracking_iou_threshold":("FLOAT",{"default":0.08,"min":0.0,"max":1.0,"step":0.01,"advanced":True}),
            "tracking_embedding_threshold":("FLOAT",{"default":0.35,"min":-1.0,"max":1.0,"step":0.01,"advanced":True}),
        }}
    def run(self, **kw):
        segs=kw["SEGMENTS"]; log=kw["LOG"]
        if not isinstance(segs,dict) or segs.get("type")!="CMK_VIDEO_SEGMENTS": raise TypeError("CMK FaceSwap Video: SEGMENTS is not CMK_VIDEO_SEGMENTS")
        if not isinstance(log,dict): raise TypeError("CMK FaceSwap Video: LOG is not CMK_LOG_PIPE")
        enabled=bool(kw.get("FACE SWAP", True))
        source_rgb=tensor_to_uint8_rgb(kw["IMAGE SOURCE"][0]); detector_model=str(kw["DETECT MODEL"]); swap_model=str(kw["SWAP MODEL"])
        if not enabled:
            all_segments=list(segs.get("segments") or [])
            mode=str(kw.get("SEGMENT SELECTION","Full Path (Batch)"))
            log_out=cmk_add_block(log,"FaceSwap Video",70,[f"FACE SWAP         : OFF",f"SEGMENT SELECTION : {mode}",f"SEGMENTS TOTAL    : {len(all_segments)}",f"SEGMENTS OUTPUT   : {len(all_segments)}",f"STATUS            : BYPASSED",f"OUTPUT SOURCE     : original input segments"])
            summary="\n".join(["Face Swap: OFF",f"Selection: {mode}",f"Segments passed through: {len(all_segments)}","Status: bypassed"])
            diagnostic=make_diagnostic_payload(title="FaceSwap Video",node="CMK FaceSwap Video",previews=[],stages=[],summary=summary,details=summary,mode="Video",metadata={"enabled":False,"selection":mode,"segments":len(all_segments)})
            return (segs,log_out,diagnostic)
        kw["FACE ENHANCER"]=validate_enhancer_mode(str(kw["FACE ENHANCER"]))
        kw["crop_factor"]=min(3.0,max(1.0,float(kw["crop_factor"])))
        engine=CMKVideoSwapEngine(detector_model); source_face=engine.select_source(source_rgb,str(kw["SOURCE FACE"]),int(kw["drop_size"]))
        model_path=resolve_swap_model(swap_model)
        settings={k:kw[k] for k in ["SWAP MODEL","DETECT MODEL","FACE ENHANCER","TARGET FACE","SOURCE FACE","BLEND","bbox_dilation","crop_factor","drop_size","feather","max_missing_frames","tracking_iou_threshold","tracking_embedding_threshold"]}
        base_sig=_hash_json({"schema":_SCHEMA,"engine":_ENGINE_VERSION,"content_guard":GUARD_VERSION,"source_manifest":_file_id(segs["manifest_path"]),"source_image":_image_hash(source_rgb),"swap_model":_file_id(model_path),"settings":settings})
        root=_output_root()/"video"/"face_swap"/(Path(segs.get("source_path","video")).stem or "video")/base_sig[:20]; root.mkdir(parents=True,exist_ok=True)
        manifest_path=root/"swap.json"; manifest={"type":"CMK_VIDEO_FACE_SWAP_MANIFEST","version":1,"schema":_SCHEMA,"engine_version":_ENGINE_VERSION,"source_segments_manifest":segs["manifest_path"],"source_image_name":str(segs.get("source_image_name","") or ""),"swap_signature":base_sig,"settings":settings,"segments":{}}
        if manifest_path.exists():
            try:
                old=json.loads(manifest_path.read_text(encoding="utf-8"))
                if old.get("swap_signature")==base_sig: manifest=old
            except Exception: pass
        all_segments=list(segs.get("segments") or [])
        mode=str(kw["SEGMENT SELECTION"])
        if not all_segments: raise RuntimeError("CMK FaceSwap Video: no source segments")
        last_used_index=_load_last_used_index(segs)
        if mode=="Full Path (Batch)":
            selected=all_segments
        elif mode=="Randomize":
            candidates=all_segments
            if len(all_segments)>1 and last_used_index is not None:
                filtered=[seg for seg in all_segments if int(seg.get("index",-1))!=last_used_index]
                if filtered:
                    candidates=filtered
            selected=[random.choice(candidates)]
            _store_last_used_index(segs,int(selected[0]["index"]))
            last_used_index=int(selected[0]["index"])
        else:
            if last_used_index is None:
                raise RuntimeError("CMK FaceSwap Video: Last Used Segment is unavailable because Randomize has not selected a segment yet")
            matches=[seg for seg in all_segments if int(seg.get("index",-1))==last_used_index]
            if not matches:
                raise RuntimeError(f"CMK FaceSwap Video: last used segment {last_used_index} is no longer present in the current SEGMENTS context")
            selected=[matches[0]]
        selected_display="ALL" if mode=="Full Path (Batch)" else f"{int(selected[0]['index'])+1} / {len(all_segments)}"
        ffmpeg=_executable("ffmpeg"); processed=reused=invalidated=failed=frames=missing=0; out_segments=[]; lines=[]; created_indices=[]
        estimated_total=sum(max(1, int(seg.get("estimated_frames") or 1)) for seg in selected)
        progress=ProgressBar(estimated_total)
        progress_value=0
        swap_settings=SelectedSwapSettings(swap_model=swap_model,enhancer_mode=str(kw["FACE ENHANCER"]),bbox_dilation=int(kw["bbox_dilation"]),crop_factor=float(kw["crop_factor"]),feather=int(kw["feather"]))
        for seg in selected:
            idx=int(seg["index"]); src=Path(seg["path"]); out=root/f"segment_{idx:04d}.mp4"; rec=manifest["segments"].get(str(idx),{})
            fingerprint=_hash_json({"base":base_sig,"segment":_file_id(src),"index":idx,"start":seg.get("start"),"end":seg.get("end")})
            if rec.get("status")=="complete" and rec.get("fingerprint")==fingerprint and out.exists() and out.stat().st_size>0:
                reused+=1; stats=rec; lines.append(f"{idx+1:03d}/{len(all_segments):03d} REUSED");
            else:
                if rec: invalidated+=1
                tmp_video=root/f"segment_{idx:04d}.video.tmp.mp4"; tmp_out=root/f"segment_{idx:04d}.tmp.mp4"
                for p in (tmp_video,tmp_out):
                    if p.exists(): p.unlink()
                probe=_probe_video(src); w=int(probe["width"]); h=int(probe["height"]); fps=float(probe["fps"])
                dec=subprocess.Popen([ffmpeg,"-v","error","-i",str(src),"-map","0:v:0","-f","rawvideo","-pix_fmt","rgb24","pipe:1"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                enc=subprocess.Popen([ffmpeg,"-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",f"{fps:.12g}","-i","pipe:0","-an","-c:v",str(segs.get("video_codec","libx264")),"-b:v",str(segs.get("video_bitrate","8000k")),"-preset",str(segs.get("preset","fast")),"-pix_fmt","yuv420p",str(tmp_video)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
                state=None; local_frames=local_missing=0
                try:
                    frame_bytes=w*h*3
                    print(f"[CMK FaceSwap Video] segment {idx+1}/{len(all_segments)}: frame processing started ({w}x{h} @ {fps:.3f} fps)")
                    while True:
                        model_management.throw_exception_if_processing_interrupted()
                        raw=dec.stdout.read(frame_bytes)
                        if not raw: break
                        if len(raw)!=frame_bytes: raise RuntimeError(f"short decoded frame at frame {local_frames}")
                        frame=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3).copy()
                        model_management.throw_exception_if_processing_interrupted()
                        engine.inspect_target_content(frame)
                        faces=engine.detect_filtered(frame,int(kw["drop_size"]))
                        target,state=engine.select_target(frame,faces,str(kw["TARGET FACE"]),state,int(kw["max_missing_frames"]),float(kw["tracking_iou_threshold"]),float(kw["tracking_embedding_threshold"]))
                        result=frame
                        if target is None:
                            local_missing+=1
                        else:
                            model_management.throw_exception_if_processing_interrupted()
                            result=_blend(frame,engine.swap_frame(frame,source_rgb,source_face,target,swap_settings),float(kw["BLEND"]))
                        model_management.throw_exception_if_processing_interrupted()
                        enc.stdin.write(result.tobytes())
                        local_frames+=1
                        progress_value+=1
                        progress.update(1)
                        if local_frames == 1 or local_frames % 10 == 0:
                            print(f"[CMK FaceSwap Video] segment {idx+1}/{len(all_segments)}: frame {local_frames} processed")
                    dec.stdout.close(); dec_rc=dec.wait(); enc.stdin.close(); enc_err=enc.stderr.read().decode("utf-8","replace"); enc_rc=enc.wait()
                    if dec_rc!=0: raise RuntimeError(dec.stderr.read().decode("utf-8","replace") or f"decoder exit {dec_rc}")
                    if enc_rc!=0: raise RuntimeError(enc_err or f"encoder exit {enc_rc}")
                    mux=subprocess.run([ffmpeg,"-y","-v","error","-i",str(tmp_video),"-i",str(src),"-map","0:v:0","-map","1:a?","-c:v","copy","-c:a","copy","-shortest",str(tmp_out)],capture_output=True,text=True)
                    if mux.returncode!=0: raise RuntimeError("audio copy/mux failed: "+mux.stderr.strip())
                    os.replace(tmp_out,out); tmp_video.unlink(missing_ok=True)
                    stats={"index":idx,"status":"complete","fingerprint":fingerprint,"output_path":str(out),"frames_total":local_frames,"frames_without_target":local_missing,"completed_at":time.time()}
                    manifest["segments"][str(idx)]=stats; _atomic_json(manifest_path,manifest); created_indices.append(idx); processed+=1; frames+=local_frames; missing+=local_missing; lines.append(f"{idx+1:03d}/{len(all_segments):03d} PROCESSED {local_frames} frames / {local_missing} without target")
                except Exception as exc:
                    for proc in (dec, enc):
                        try:
                            if proc.poll() is None:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2.0)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait(timeout=2.0)
                        except Exception:
                            pass
                    for p in (tmp_video,tmp_out): p.unlink(missing_ok=True)
                    if isinstance(exc, ContentGuardBlocked):
                        for created_idx in created_indices:
                            (root/f"segment_{created_idx:04d}.mp4").unlink(missing_ok=True)
                            manifest["segments"].pop(str(created_idx),None)
                        manifest["segments"].pop(str(idx),None)
                        _atomic_json(manifest_path,manifest)
                        raise
                    failed+=1; manifest["segments"][str(idx)]={"index":idx,"status":"failed","fingerprint":fingerprint,"error":str(exc),"frame":local_frames,"updated_at":time.time()}; _atomic_json(manifest_path,manifest)
                    if exc.__class__.__name__ == "InterruptProcessingException":
                        print(f"[CMK FaceSwap Video] interrupted at segment {idx+1}/{len(all_segments)}, frame {local_frames}")
                        raise
                    raise RuntimeError(f"CMK FaceSwap Video: segment {idx+1}/{len(all_segments)}, frame {local_frames}: {exc}") from exc
            new_seg=dict(seg)
            new_seg["path"]=str(out)
            new_seg["filename"]=out.name
            out_segments.append(new_seg)

        # Test selections are independent short timelines. The merger requires
        # local contiguous indices, while source identity remains traceable.
        if mode != "Full Path (Batch)":
            normalized_segments=[]
            local_start=0.0
            for local_index,item in enumerate(out_segments):
                normalized=dict(item)
                source_start=float(item.get("start",0.0) or 0.0)
                source_end=float(item.get("end",source_start) or source_start)
                duration=max(0.0,float(item.get("duration",source_end-source_start) or (source_end-source_start)))
                normalized["source_index"]=int(item.get("index",local_index))
                normalized["source_start"]=source_start
                normalized["source_end"]=source_end
                normalized["index"]=local_index
                normalized["start"]=local_start
                normalized["end"]=local_start+duration
                normalized["duration"]=duration
                normalized_segments.append(normalized)
                local_start+=duration
            out_segments=normalized_segments

        out_payload=dict(segs)
        # Keep source_path as the immutable full-video origin. Compare receives
        # the exact original material matching the active output timeline via
        # compare_source_path in the same CMK_VIDEO_SEGMENTS transport.
        compare_source_path=str(Path(segs.get("source_path","")).resolve())
        if mode != "Full Path (Batch)":
            out_payload["duration"]=sum(float(x.get("duration",0.0) or 0.0) for x in out_segments)
            out_payload["frame_count"]=sum(int(x.get("estimated_frames",0) or 0) for x in out_segments)
        if mode != "Full Path (Batch)":
            compare_source_path=str(Path(selected[0]["path"]).resolve())
        out_payload.update({"output_directory":str(root),"manifest_path":str(manifest_path),"segment_paths":tuple(x["path"] for x in out_segments),"segments":tuple(out_segments),"parent_manifest_path":segs.get("manifest_path"),"processing_type":"face_swap","processing_signature":base_sig,"compare_source_path":compare_source_path,"segment_selection":mode})
        log_out=cmk_add_block(log,"FaceSwap Video",70,[f"FACE SWAP         : ON",f"SEGMENT SELECTION : {mode}",f"SEGMENTS TOTAL     : {len(all_segments)}",f"SEGMENTS OUTPUT    : {len(out_segments)}",f"SELECTED SEGMENT   : {selected_display}",f"PROCESSED          : {processed}",f"REUSED             : {reused}",f"INVALIDATED        : {invalidated}",f"FAILED             : {failed}",f"FRAMES PROCESSED   : {frames}",f"WITHOUT TARGET     : {missing}",f"SWAP MODEL         : {swap_model}",f"DETECT MODEL       : {detector_model}",f"FACE ENHANCER      : {kw['FACE ENHANCER']}",f"OUTPUT DIRECTORY   : {root}",f"SWAP MANIFEST      : {manifest_path}"])
        summary="\n".join(["Face Swap: ON",f"Selection: {mode}",f"Selected segment: {selected_display}",f"Source segments: {len(all_segments)}",f"Output segments: {len(out_segments)}",f"Processed: {processed}",f"Reused: {reused}",f"Invalidated: {invalidated}",f"Failed: {failed}",*lines])
        diagnostic=make_diagnostic_payload(title="FaceSwap Video",node="CMK FaceSwap Video",previews=[],stages=[],summary=summary,details=summary,mode="Video",metadata={"selection":mode,"processed":processed,"reused":reused,"invalidated":invalidated,"failed":failed,"manifest":str(manifest_path)})
        return (out_payload,log_out,diagnostic)
