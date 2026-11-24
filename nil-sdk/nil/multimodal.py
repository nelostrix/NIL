"""Bounded timed GIF/TIFF clips and explicit audio/visual/text-feature fusion."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
from nil.runtime import require
from nil.vision import ImageTransform


@dataclass(frozen=True)
class Clip:
    frames: object
    timestamps: tuple
    durations: tuple


def load_clip(path, *, size=(64,64), max_frames=64, max_bytes=32*1024*1024, max_pixels=4*1024*1024, default_duration_ms=100, max_decoded_bytes=64*1024*1024):
    """Decode timed GIF or multipage TIFF; no implicit FFmpeg/network/backend.

    Full clips exceeding max_frames fail rather than silently truncate. Missing
    duration uses the explicit default. MP4/WebM require a future codec adapter.
    """
    image=require('PIL.Image','vision');torch=require('torch','torch')
    transform=ImageTransform(size=size,max_bytes=max_bytes,max_pixels=max_pixels)
    if type(max_frames) is not int or not 1<=max_frames<=256 or type(default_duration_ms) is not int or not 1<=default_duration_ms<=60000:raise ValueError('Invalid frame/duration budget')
    with Path(path).open('rb') as stream:encoded=stream.read(max_bytes+1)
    if len(encoded)>max_bytes:raise ValueError('Clip exceeds byte budget')
    if type(max_decoded_bytes) is not int or not 1<=max_decoded_bytes<=256*1024*1024:raise ValueError('Invalid decoded budget')
    frames=[];times=[];durations=[];cursor=0.
    with image.open(BytesIO(encoded)) as source:
        if source.format not in {'GIF','TIFF'}:raise ValueError('Only GIF/TIFF timed clips are supported')
        for index in range(max_frames+1):
            try:source.seek(index)
            except EOFError:break
            if index==max_frames:raise ValueError('Clip exceeds frame budget')
            if source.width*source.height>max_pixels:raise ValueError('Clip exceeds pixel budget')
            duration=source.info.get('duration',default_duration_ms)
            if type(duration) not in (int,float) or not math.isfinite(duration) or not 0<duration<=60000:raise ValueError('Invalid frame duration')
            if (len(frames)+1)*3*transform.size[0]*transform.size[1]*4>max_decoded_bytes:raise ValueError('Decoded clip exceeds memory budget')
            rgba=source.convert('RGBA');background=image.new('RGBA',rgba.size,(255,255,255,255))
            rgb=image.alpha_composite(background,rgba).convert('RGB').resize((transform.size[1],transform.size[0]),image.Resampling.BILINEAR)
            tensor=torch.frombuffer(bytearray(rgb.tobytes()),dtype=torch.uint8).reshape(*transform.size,3).permute(2,0,1).to(torch.float32).div_(255).contiguous()
            frames.append(tensor);times.append(cursor);durations.append(duration/1000);cursor+=duration/1000
    if not frames:raise ValueError('Empty clip')
    return Clip(torch.stack(frames),tuple(times),tuple(durations))


def visual_features(clip):
    """Per-channel time/spatial mean, std and mean absolute inter-frame change."""
    torch=require('torch','torch')
    if not isinstance(clip,Clip) or not isinstance(clip.frames,torch.Tensor) or clip.frames.ndim!=4 or clip.frames.shape[1]!=3 or not len(clip.frames) or len(clip.timestamps)!=len(clip.frames) or not torch.isfinite(clip.frames).all():raise ValueError('Invalid TCHW clip')
    frames=clip.frames
    motion=(frames[1:]-frames[:-1]).abs().mean((0,2,3)) if len(frames)>1 else torch.zeros(3,dtype=frames.dtype,device=frames.device)
    return torch.cat([frames.mean((0,2,3)),frames.std((0,2,3),unbiased=False),motion]).detach().cpu().numpy()


def aligned_audio(clip, waveform):
    """Return a waveform slice for every frame; reject incomplete audio coverage."""
    from nil.audio import segment
    if not isinstance(clip,Clip) or len(clip.timestamps)!=len(clip.durations):raise ValueError('Invalid timeline')
    slices=[];previous=0.
    for start,duration in zip(clip.timestamps,clip.durations):
        if not math.isfinite(start) or not math.isfinite(duration) or start<previous or duration<=0:raise ValueError('Invalid/nonmonotonic timeline')
        end=start+duration
        if end>waveform.seconds+1e-9:raise ValueError('Audio does not cover the clip timeline')
        slices.append(segment(waveform,start,min(end,waveform.seconds)));previous=end
    return slices


def fuse_features(values, dimensions, *, missing='error'):
    """Sorted named feature concatenation plus presence bits, with fixed dimensions.

    Text/other embeddings can be provided explicitly; no encoder is downloaded.
    Fused vectors use the same declared dimensions/order at train and inference.
    """
    np=require('numpy','audio')
    if not isinstance(values,dict) or not isinstance(dimensions,dict) or not 1<=len(dimensions)<=16 or set(values)-set(dimensions) or missing not in {'error','zeros'}:raise ValueError('Invalid modalities or missing policy')
    parts=[];present=[]
    for name,width in sorted(dimensions.items()):
        if not isinstance(name,str) or not name or type(width) is not int or not 1<=width<=4096:raise ValueError('Invalid modality dimensions')
        value=values.get(name)
        if value is None:
            if missing=='error':raise ValueError(f'Missing modality: {name}')
            parts.append(np.zeros(width,dtype=np.float32));present.append(0.)
        else:
            vector=np.asarray(value,dtype=np.float32)
            if vector.shape!=(width,) or not np.isfinite(vector).all():raise ValueError('Feature shape/value mismatch')
            parts.append(vector);present.append(1.)
    return np.concatenate([*parts,np.asarray(present,dtype=np.float32)])


def _decode_video(path, size, fps, max_frames, max_pixels, timeout, preserve_timestamps=False):
    """Runs inside NIL's owned process tree; external codecs are explicit opt-ins."""
    import json
    import subprocess
    probe=subprocess.run(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-select_streams','v:0','-show_entries',
        'stream=width,height','-of','json',path],capture_output=True,check=True,timeout=timeout)
    metadata=json.loads(probe.stdout)
    if not metadata.get('streams'):raise ValueError('Video has no visual stream')
    stream=metadata['streams'][0]
    if stream.get('width',0)*stream.get('height',0)>max_pixels:raise ValueError('Video exceeds source pixel budget')
    command=['ffmpeg','-v','fatal','-nostdin','-threads','1','-protocol_whitelist','file,pipe','-i',path,'-map','0:v:0','-an',
             '-vf',(f'scale={size[1]}:{size[0]}' if preserve_timestamps else f'fps={fps},scale={size[1]}:{size[0]}'),'-fps_mode','passthrough','-threads','1','-frames:v',str(max_frames+1),
             '-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=timeout)
    if not preserve_timestamps:return result.stdout
    timeline=subprocess.run(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-select_streams','v:0',
        '-read_intervals',f'%+#{max_frames+64}','-show_frames','-show_entries',
        'frame=best_effort_timestamp_time,pkt_duration_time,duration_time','-of','json',path],capture_output=True,check=True,timeout=timeout)
    rows=json.loads(timeline.stdout).get('frames',[])
    count=len(result.stdout)//(3*size[0]*size[1]);rows=rows[:count]
    if len(rows)!=count:raise ValueError('Cannot align original frame timestamps')
    times=[float(row['best_effort_timestamp_time']) for row in rows]
    if any(not math.isfinite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Unsupported missing/nonmonotonic source timestamps')
    durations=[b-a for a,b in zip(times,times[1:])]
    if rows:
        last=rows[-1].get('duration_time',rows[-1].get('pkt_duration_time'))
        if last is None:raise ValueError('Source omits last-frame duration; use explicit CFR sampling')
        duration=float(last)
        if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid last-frame duration')
        durations.append(duration)
    return result.stdout,times,durations


def load_video(path, *, size=(64,64), fps=4., max_frames=64, max_bytes=64*1024*1024,
               max_pixels=4*1024*1024, max_decoded_bytes=64*1024*1024, timeout=30., job_root='./nil_video_jobs', preserve_timestamps=False):
    """Opt-in FFmpeg MP4/WebM/local-video decoding with owned deadlines.

    Output timestamps are the requested constant-rate sampling grid starting at0,
    not original variable-frame timestamps, unless preserve_timestamps=True.
    Preservation refuses missing/nonmonotonic timestamps or last duration.
    No audio is implicitly decoded.
    External ffmpeg/ffprobe must already be installed; NIL never installs them.
    """
    import shutil
    from nil.execution import run,Limits
    torch=require('torch','torch')
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:raise RuntimeError('Explicit FFmpeg/ffprobe installation required for load_video')
    if type(preserve_timestamps) is not bool:raise ValueError('preserve_timestamps must be bool')
    transform=ImageTransform(size=size,max_bytes=max_bytes,max_pixels=max_pixels)
    if type(fps) not in (int,float) or not math.isfinite(fps) or not .1<=fps<=120 or type(max_frames) is not int or not 1<=max_frames<=256:raise ValueError('Invalid sampling/frame budget')
    if type(max_decoded_bytes) is not int or not 1<=max_decoded_bytes<=256*1024*1024:raise ValueError('Invalid decode memory budget')
    path=Path(path).resolve()
    if not path.is_file() or path.stat().st_size>max_bytes:raise ValueError('Local video exceeds file budget')
    frame_bytes=3*transform.size[0]*transform.size[1]
    if max_frames*frame_bytes*4>max_decoded_bytes:raise ValueError('Requested decoded output exceeds memory budget')
    limits=Limits(timeout=timeout,payload_mb=max(1,math.ceil((max_frames+1)*frame_bytes/1024**2)+1),log_bytes=8192)
    raw=run(_decode_video,str(path),transform.size,float(fps),max_frames,max_pixels,timeout,preserve_timestamps,limits=limits,job_root=job_root)
    if preserve_timestamps:raw,times,durations=raw
    if not raw or len(raw)%frame_bytes:raise ValueError('Empty or malformed decoded video')
    count=len(raw)//frame_bytes
    if count>max_frames:raise ValueError('Video exceeds frame budget')
    frames=torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(count,*transform.size,3).permute(0,3,1,2).to(torch.float32).div_(255).contiguous()
    return Clip(frames,tuple(times),tuple(durations)) if preserve_timestamps else Clip(frames,tuple(i/fps for i in range(count)),tuple(1/fps for _ in range(count)))


def load_encoder(path, **options):
    """Load an explicitly supplied local native CLIP model and processor."""
    from nil._multimodal_encoder import Encoder
    return Encoder(path,**options)
