"""
Video data structure
"""
import cv2
import numpy as np
import os

class Video(object):
    """
    Video wrapper on top of cv2.VideoCapture and some functionalities.
    """
    def __init__(self, videofile):
        if not os.path.isfile(videofile):
            raise IOError('video {} not found'.format(videofile))
        self.cap = cv2.VideoCapture(videofile)
        self.path = os.path.dirname(videofile)
        self.name = os.path.basename(videofile)

    def __getitem__(self, index):
        """
        Get a specific frame at index

        WARNING: this is very slow because it performs a seek for every frame
        index, thus it has to decode from the new location, discarding previous
        decoded frames---even if you are running it sequentially.
        """
        if not self.cap.isOpened():
            raise IOError('Video ' + self.path + ' is not open')
        count = self.length
        if index >= count:
            raise ValueError('index must be less than video total frames')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, data = self.cap.read()
        if not ret:
            raise RuntimeError('frame not read')
        return data

    @property
    def length(self): 
        """Returns the video size"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self):
        """Returns the video width"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        """Returns the video height"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def shape(self):
        """Returns the video shape as (nframes, height, width)

        Does not return the number of color channels.
        """
        return self.length, self.height, self.width

    def next(self, cvtgray=False):
        """
        Read next frame using lazy evaluation
        """
        if not self.cap.isOpened():
            raise IOError('Video ' + self.path + ' is not open')
        ret, data = self.cap.read()
        # if not ret:
        #     raise RuntimeError('frame not read')
        if cvtgray:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return ret, data

    def __iter__(self, cvtgray=False):
        """
        Read next frame using lazy evaluation
        """
        if not self.cap.isOpened():
            raise IOError('Video ' + self.path + ' is not open')
        ret, data = self.cap.read()
        while ret:
            if cvtgray:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            yield data
            ret, data = self.cap.read()
        self.reset()

    def snippet(self, start=0, final=0, cvtgray=False):
        "Get a set of frames"
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if final == 0:
            final = count
        if start >= count:
            raise ValueError('start must be less than video total frames')
        if start < 0 or final < 0:
            raise ValueError('start and final must be positive')
        if final < start:
            raise ValueError('start must be less than final')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        append = frames.append
        while self.cap.isOpened() and start <= final:
            ret, data = self.cap.read()
            if not ret:
                break
            if cvtgray:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            append(data)
            start = start + 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frames

    def reset(self):
        """Reset stream"""
        if not self.cap.isOpened():
            raise IOError('Video ' + self.path + ' is not open')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek(self, frameid):
        """Seek frame"""
        if not self.cap.isOpened():
            raise IOError('Video ' + self.path + ' is not open')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameid)

    def show(self, start=0, final=0, delay=10, cvtgray=False):
        "Show video"
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if final == 0:
            final = count
        if start >= count:
            raise ValueError('start must be less than video total frames')
        if start < 0 or final < 0:
            raise ValueError('start and final must be positive')
        if final < start:
            raise ValueError('start must be less than final')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        while self.cap.isOpened() and start <= final:
            ret, frame = self.cap.read()
            if not ret:
                break
            if cvtgray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            start = start + 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cv2.destroyAllWindows()

    def astensor(self):
        nframes, height, width = self.shape
        tensor = np.array((nframes, height, width, 3))
        for i, frame in enumerate(self.iterframes()):
            tensor[i] = frame
        return tensor
