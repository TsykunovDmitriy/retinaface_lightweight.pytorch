import os
import io
import av
import cv2
import ffmpeg
import subprocess
import numpy as np
from PIL import Image

def vidread(path):
    print(f"Start reading {path}")
    assert os.path.isfile(path), f"Not found video file {path}"

    cap = cv2.VideoCapture(path)
    fps = np.ceil(cap.get(cv2.CAP_PROP_FPS))

    video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()

    audio_path = f"{os.path.dirname(path)}/audio_{os.path.splitext(os.path.basename(path))[0]}.m4a"
    command = f"ffmpeg -y -i {path} -vn -c:a copy {audio_path}"
    subprocess.call(command, shell=True)
    
    assert os.path.isfile(audio_path), f"Not found audio file {audio_path}"
    return video, audio_path, fps

def vidwrite(frames, audio_path, rate=30):
    buff = io.BytesIO()
    height, width, channels = frames[0].shape

    output = av.open(buff, 'w', 'mp4')
    media_container = av.open(audio_path, 'r', 'm4a')

    media_container_stream = media_container.streams.audio[0]
    media_stream = output.add_stream('aac', media_container_stream.rate)
    media_stream.format = media_container_stream.format.name
    media_stream.channels = media_container_stream.channels

    vstream = output.add_stream('h264', rate=rate)
    vstream.width = int(np.ceil(float(width)/2)*2)
    vstream.height = int(np.ceil(float(height)/2)*2)
    vstream.pix_fmt = 'yuv420p'

    for packet in media_container.demux():
        for frame in packet.decode():
            frame.pts = None
            for packet in media_stream.encode(frame):
                output.mux(packet)

    for image in frames:
        pil_im = Image.fromarray(image)
        frame = av.VideoFrame.from_image(pil_im)
        for packet in vstream.encode(frame):
            output.mux(packet)

    # Flush stream
    for packet in vstream.encode():
        output.mux(packet)

    # Close the output
    output.close()
    os.remove(audio_path)
    return buff

def write_bytesio_to_file(output_path, bytesio):
    print("Saving result!")
    with open(output_path, "wb") as outfile:
        outfile.write(bytesio.getbuffer())