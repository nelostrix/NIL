"""Actual opt-in MP4 codec acceptance, separate from base/optional backend installs."""
import shutil
import subprocess
import pytest
pytest.importorskip('torch')
import nil
pytestmark=pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),reason='Actual explicit FFmpeg backend required')


def test_owned_mp4_decode_grid_and_budgets(tmp_path):
    path=tmp_path/'test.mp4'
    subprocess.run(['ffmpeg','-v','fatal','-f','lavfi','-i','color=c=red:s=32x24:r=8:d=1','-threads','1','-pix_fmt','yuv420p','-y',str(path)],check=True,timeout=10)
    clip=nil.multimodal.load_video(path,size=(8,12),fps=4,max_frames=8,job_root=tmp_path/'jobs')
    assert clip.frames.shape==(4,3,8,12) and clip.timestamps==(0.,.25,.5,.75)
    assert clip.frames[:,0].mean()>.95 and clip.frames[:,1:].mean()<.05
    with pytest.raises(ValueError,match='frame budget'):nil.multimodal.load_video(path,max_frames=2,job_root=tmp_path/'overflow')
    with pytest.raises(ValueError,match='memory'):nil.multimodal.load_video(path,max_decoded_bytes=10,job_root=tmp_path/'memory')


def test_original_vfr_timestamps_match_independent_probe(tmp_path):
    import json
    path=tmp_path/'variable.mp4'
    subprocess.run(['ffmpeg','-v','fatal','-f','lavfi','-i','testsrc=size=32x24:rate=10:duration=1',
        '-vf',"select='eq(n,0)+eq(n,1)+eq(n,4)+eq(n,9)'",'-fps_mode','vfr','-threads','1','-pix_fmt','yuv420p',str(path)],check=True,timeout=10)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],timeout=10))
    expected=tuple(float(row['best_effort_timestamp_time']) for row in probe['frames'])
    clip=nil.multimodal.load_video(path,size=(8,12),max_frames=8,preserve_timestamps=True,job_root=tmp_path/'jobs')
    assert clip.timestamps==expected==pytest.approx((0.,.1,.4,.9))
    assert clip.durations[:-1]==pytest.approx((.1,.3,.5)) and clip.durations[-1]>0
