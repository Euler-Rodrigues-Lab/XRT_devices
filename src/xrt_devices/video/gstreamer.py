"""Bounded, latest-frame GStreamer publisher for RGB simulator/camera images.

Publishes H.264 MPEG-TS over UDP to MediaMTX, or RTSP when rtspclientsink is
installed. One SBS frame carries both eyes with a shared timestamp.
"""
from pathlib import Path
from urllib.parse import urlparse
import os
import queue
import select
import shutil
import subprocess
import threading
import time
import numpy as np


def _sink(destination):
    parsed=urlparse(destination)
    if parsed.scheme=='udp' and parsed.hostname and parsed.port:
        return ['!','mpegtsmux','alignment=7','!','udpsink',f'host={parsed.hostname}',f'port={parsed.port}','sync=false','async=false']
    if parsed.scheme=='rtsp' and parsed.hostname:
        return ['!','rtspclientsink',f'location={destination}','protocols=tcp','latency=0']
    if parsed.scheme=='file':
        return ['!','mpegtsmux','!','filesink',f'location={parsed.path}']
    raise ValueError('destination must be udp://host:port, rtsp://host/path or file:///path.ts')


def _branch(fps, destination, bitrate, crop=None):
    args=['queue','max-size-buffers=1','max-size-bytes=0','max-size-time=0','leaky=downstream']
    if crop:args+=['!','videocrop',crop]
    args+=['!','videoconvert','!','video/x-raw,format=I420',
           '!','x264enc','tune=zerolatency','speed-preset=ultrafast',f'bitrate={bitrate}',
           f'key-int-max={max(1,fps//2)}','bframes=0','!','h264parse','config-interval=-1']
    return args+_sink(destination)


def pipeline(width, height, fps, destination, bitrate=12000, eye_destinations=None):
    if min(width,height,fps,bitrate)<=0 or width%2 or height%2:
        raise ValueError('positive fps/bitrate and even positive dimensions required')
    args=['gst-launch-1.0','-q','fdsrc','fd=0',f'blocksize={width*height*3}',
          '!','rawvideoparse','format=rgb',f'width={width}',f'height={height}',f'framerate={fps}/1']
    if eye_destinations is None:
        return args+['!']+_branch(fps,destination,bitrate)
    if width%4 or len(eye_destinations)!=2:
        raise ValueError('split stereo requires width divisible by 4 and two eye destinations')
    if len({destination,*eye_destinations})!=3:
        raise ValueError('SBS, left, and right destinations must be distinct')
    # Split the same raw frame and timestamps before encoding. Do not send the
    # full SBS image to each eye: crop opposite halves for true binocular video.
    args+=['!','tee','name=stereo','stereo.','!']+_branch(fps,destination,bitrate)
    for target,crop in zip(eye_destinations,[f'right={width//2}',f'left={width//2}']):
        args+=['stereo.','!']+_branch(fps,target,max(1,bitrate//2),crop)
    return args


def check_plugins(destination='udp://127.0.0.1:9000', split_stereo=False):
    if not shutil.which('gst-launch-1.0') or not shutil.which('gst-inspect-1.0'):
        raise RuntimeError('Install GStreamer tools and base/good/bad/ugly plugins')
    plugins=['fdsrc','rawvideoparse','queue','videoconvert','x264enc','h264parse']
    if split_stereo:plugins+=['tee','videocrop']
    plugins+=['rtspclientsink'] if destination.startswith('rtsp:') else ['mpegtsmux','udpsink' if destination.startswith('udp:') else 'filesink']
    missing=[p for p in plugins if subprocess.run(['gst-inspect-1.0',p],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode]
    if missing: raise RuntimeError('Missing GStreamer elements: '+', '.join(missing))


class GStreamerPublisher:
    def __init__(self,width,height,fps=30,destination='udp://127.0.0.1:9000',bitrate=12000,log_path=None,*,eye_destinations=None):
        self.width,self.height=width,height
        self.eye_destinations=tuple(eye_destinations) if eye_destinations is not None else None
        self.args=pipeline(width,height,fps,destination,bitrate,self.eye_destinations)
        self.destination,self.log_path=destination,log_path
        self.frames=queue.Queue(maxsize=1);self.stop=threading.Event();self.error=None
        self.submitted=self.written=self.dropped=0
    def __enter__(self):
        check_plugins(self.destination, self.eye_destinations is not None)
        for destination in self.eye_destinations or ():
            check_plugins(destination)
        self.log=open(self.log_path,'wb') if self.log_path else None
        try:
            self.process=subprocess.Popen(self.args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
            os.set_blocking(self.process.stdin.fileno(),False)
            self.thread=threading.Thread(target=self._worker,name='gstreamer-rgb-writer',daemon=True)
            self.thread.start()
            return self
        except BaseException:
            if self.log:self.log.close()
            raise
    def check(self):
        if self.error: raise RuntimeError('GStreamer writer failed') from self.error
        if self.process.poll() is not None: raise RuntimeError(f'GStreamer exited {self.process.returncode}; inspect streaming log')
    def send_frame(self,frame):
        self.check()
        frame=np.asarray(frame)
        if frame.shape!=(self.height,self.width,3) or frame.dtype!=np.uint8:
            raise ValueError(f'expected uint8 RGB {(self.height,self.width,3)}')
        payload=frame.tobytes(order='C');self.submitted+=1
        try:self.frames.put_nowait(payload)
        except queue.Full:
            try:self.frames.get_nowait();self.dropped+=1
            except queue.Empty:pass
            self.frames.put_nowait(payload)
    def _worker(self):
        try:
            while not self.stop.is_set() or not self.frames.empty():
                try:payload=memoryview(self.frames.get(timeout=.05))
                except queue.Empty:continue
                deadline=time.monotonic()+2
                while payload:
                    if time.monotonic()>deadline:raise TimeoutError('GStreamer input stalled')
                    if self.process.poll() is not None:raise BrokenPipeError('GStreamer stopped')
                    if select.select([], [self.process.stdin], [], .05)[1]:
                        try:count=os.write(self.process.stdin.fileno(),payload)
                        except BlockingIOError:continue
                        payload=payload[count:]
                self.written+=1
        except BaseException as exc:self.error=exc
    def __exit__(self,*exc):
        self.stop.set();self.thread.join(3)
        if self.thread.is_alive():self.process.terminate();self.thread.join(1)
        self.process.stdin.close()
        try:code=self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait();code=-9
        if self.log:self.log.close()
        if not exc[0]:
            if self.error:raise RuntimeError('GStreamer writer failed') from self.error
            if code:raise RuntimeError(f'GStreamer exited {code}')
