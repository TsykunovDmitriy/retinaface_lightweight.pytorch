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
    media_container = av.open(path)
    frames = [np.array(frame.to_image()) for frame in media_container.decode(video=0)]
    media_container.seek(0)
    return frames, media_container


def vidwrite(frames, media_container):
    buff = io.BytesIO()
    height, width, channels = frames[0].shape

    output = av.open(buff, "w", "mp4")

    media_container_stream = media_container.streams.audio[0]
    media_stream = output.add_stream("aac", media_container_stream.rate)
    media_stream.format = media_container_stream.format.name
    media_stream.channels = media_container_stream.channels

    vstream = output.add_stream("h264", rate=media_container.streams.video[0].rate)
    vstream.width = int(np.ceil(float(width)/2)*2)
    vstream.height = int(np.ceil(float(height)/2)*2)
    vstream.pix_fmt = "yuv420p"

    for frame in media_container.decode(media_container_stream):
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
    return buff

def write_bytesio_to_file(output_path, bytesio):
    print("Saving result!")
    with open(output_path, "wb") as outfile:
        outfile.write(bytesio.getbuffer())