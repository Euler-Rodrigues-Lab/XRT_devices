import pytest
from xrt_devices.video.gstreamer import pipeline

def test_stereo_pipeline_low_latency():
    cmd=pipeline(1280,480,30,'udp://127.0.0.1:9000')
    assert 'width=1280' in cmd and 'height=480' in cmd
    assert 'max-size-buffers=1' in cmd and 'bframes=0' in cmd
    assert cmd.index('rawvideoparse') < cmd.index('leaky=downstream')
    assert 'mpegtsmux' in cmd and 'host=127.0.0.1' in cmd

def test_rtsp_and_validation():
    assert 'rtspclientsink' in pipeline(640,480,30,'rtsp://localhost:8554/vr')
    with pytest.raises(ValueError):pipeline(641,480,30,'udp://localhost:9000')
    with pytest.raises(ValueError):pipeline(640,480,30,'http://localhost')


def test_binocular_pipeline_crops_each_eye_from_shared_frames():
    cmd=pipeline(1280,480,30,'udp://127.0.0.1:9000',
                 eye_destinations=('udp://127.0.0.1:9002','udp://127.0.0.1:9004'))
    assert cmd.count('rawvideoparse')==1
    assert cmd.count('tee')==1 and cmd.count('stereo.')==3
    assert cmd.count('videocrop')==2
    assert 'right=640' in cmd and 'left=640' in cmd
    assert cmd.index('right=640')<cmd.index('port=9002')<cmd.index('left=640')<cmd.index('port=9004')
    assert cmd.count('x264enc')==3
    with pytest.raises(ValueError):
        pipeline(642,480,30,'udp://127.0.0.1:9000',eye_destinations=('udp://localhost:9002','udp://localhost:9004'))
    with pytest.raises(ValueError):
        pipeline(1280,480,30,'udp://127.0.0.1:9000',eye_destinations=('udp://localhost:9002',))
    with pytest.raises(ValueError):
        pipeline(1280,480,30,'udp://127.0.0.1:9000',eye_destinations=('udp://localhost:9002','udp://localhost:9002'))
