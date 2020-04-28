import glob

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os


class PlaybackSettings:
    DEFAULT_TIME = 1
    SAMPLING_INTERVAL = 1000  # TODO: The paper says this should be user defined!

    def __init__(self, timelapse):
        """
        Defines settings that might be relevant when the Timelapse is ready to be written to disk.

        :param timelapse: the relevant Timelapse object
        """

        self.time_per_frame = np.ones(timelapse.frame_length, dtype=np.int32)
        self.timelapse = timelapse


class Timelapse:
    def __init__(self, frames, frame_length, rows, cols, channels):
        """

        :param frames:
        :param frame_length:
        :param rows:
        :param cols:
        :param channels:
        """
        self.frames = frames
        self.frame_length = frame_length
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.settings = PlaybackSettings(self)


class TimelapseProcessor:
    UPLOAD_FILES = "uploads/custom/*"

    OUTPUT_FILE = "static/output/output.mp4"

    OUTPUT_RESOLUTION = (480, 640)
    OUTPUT_FPS = 15

    VIRTUAL_SHUTTER_LEN = 100

    def __init__(self, config):
        """
        Constructs the TimelapseProcessor object with a given configuration.

        :param config: must be of the following: "uniform-sampling", "non-uniform-sampling", "uniform-sampling-wmt" or
        "non-uniform-sampling-wmt"!
        """

        if config == "uniform-sampling" or config == "non-uniform-sampling" or config == "uniform-sampling-wmt" or \
                config == "non-uniform-sampling-wmt":
            self.config = config
        else:
            raise AttributeError("Invalid config. See docs.")

        print("Starting the Timelapse Processor, config: " + config)

    def __load_timelapse(self, files_src):
        """
        Loads the files at the given destination and constructs a Timelapse object.

        timelapse.frames is defined as: [frame_index, x_pixel, y_pixel, channel (R, B, G)]

        :param files_src: the location of the timelapse files (png, jpg or jpeg)
        :return: a Timelapse object with the relevant data
        """

        print("Loading files...")

        files = glob.glob(files_src)

        print(*files)

        frames = np.array([cv2.normalize(cv2.imread(file, cv2.IMREAD_COLOR),
                                         None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                           for file in files])

        frame_length = frames.shape[0]
        rows = frames.shape[1]
        cols = frames.shape[2]
        channels = frames.shape[3]

        return Timelapse(frames, frame_length, rows, cols, channels)

    def __save_timelapse(self, timelapse):
        """
        Saves the Timelapse object in static/output/output.mp4

        :param timelapse: the given Timelapse object
        """

        # First, remove the existing file

        if os.path.exists(self.OUTPUT_FILE):
            os.remove(self.OUTPUT_FILE)

        # Begin writing the new file

        print("Saving timelapse...")

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')

        out = cv2.VideoWriter(self.OUTPUT_FILE, fourcc, self.OUTPUT_FPS,
                              (self.OUTPUT_RESOLUTION[1], self.OUTPUT_RESOLUTION[0]))

        for i in range(timelapse.frame_length):
            # Normalising the frames from 0 to 1 will mess up the output severely.

            # Thankfully, I found this:
            # https://stackoverflow.com/questions/54209466/how-to-write-a-float32-picture-to-video-file-in-opencv2

            normalized_array = (timelapse.frames[i, :, :, :] - np.min(timelapse.frames[i, :, :, :])) / (np.max(timelapse.frames[i, :, :, :]) - np.min(timelapse.frames[i, :, :, :]))

            img_array = (normalized_array * 255).astype(np.uint8)

            # If you are marking this then here is where we are sampling higher for large error values!!
            # Sorry this code is so messy!!

            for j in range(timelapse.settings.time_per_frame[i]):
                out.write(img_array)

        out.release()

    def add_motion_trails(self, timelapse):
        """
        Modifies the frames in the Timelapse and adds motion trails using the 'Virtual Shutter' as defined in the paper.

        :param timelapse: the relevant Timelapse object
        """

        print("Adding motion trails...")

        new_frames = np.zeros(timelapse.frames.shape)

        for i in range(timelapse.frame_length):
            new_frame = np.copy(timelapse.frames[i, :, :, :])

            for j in range(1, 4):
                dist = 40

                if i > dist * j:
                    alpha = (j / 4.0)
                    new_frame += timelapse.frames[dist * j, :, :, :] * alpha

            new_frames[i, :, :, :] = new_frame

        timelapse.frames = new_frames

    def perform_non_uniform_sampling(self, timelapse):
        """
        Performs non-uniform sampling on the given Timelapse object

        :param timelapse: the Timelapse to the relevant file
        """

        ssd = np.zeros(timelapse.frame_length)

        for i in range(timelapse.frame_length - 1):
            ssd[i] = np.sum((timelapse.frames[i, :, :, :] - timelapse.frames[i + 1, :, :, :]) ** 2)

        ssd /= (timelapse.rows * timelapse.cols)

        # plt.plot(ssd)
        # plt.xlabel("Frame")
        # plt.ylabel("Min-Error")
        # plt.show()

        for i in range(timelapse.frame_length):
            min_change_error = int(ssd[i] * PlaybackSettings.SAMPLING_INTERVAL)

            if min_change_error < PlaybackSettings.DEFAULT_TIME:
                min_change_error = PlaybackSettings.DEFAULT_TIME

            # print(str(i) + "=" + str(min_change_error))

            timelapse.settings.time_per_frame[i] = min_change_error

    def process_files(self):
        """
        Loads, processes and saves the files based on the given configuration on __init__.
        """

        timelapse = self.__load_timelapse(self.UPLOAD_FILES)

        if self.config == "uniform-sampling":
            pass

        elif self.config == "non-uniform-sampling":
            self.perform_non_uniform_sampling(timelapse)

        elif self.config == "uniform-sampling-wmt":
            self.add_motion_trails(timelapse)

        elif self.config == "non-uniform-sampling-wmt":
            self.perform_non_uniform_sampling(timelapse)
            self.add_motion_trails(timelapse)

        self.__subsample(timelapse)
        self.__save_timelapse(timelapse)

    def __subsample(self, timelapse):
        new_frames = np.zeros((timelapse.frame_length,) + self.OUTPUT_RESOLUTION + (3,))

        for i in range(timelapse.frame_length):
            new_frames[i, :, :, :] = cv2.resize(timelapse.frames[i, :, :, :],
                                                      (self.OUTPUT_RESOLUTION[1], self.OUTPUT_RESOLUTION[0]))
        timelapse.frames = new_frames
        timelapse.rows, timelapse.cols = self.OUTPUT_RESOLUTION


    def play_timelapse(self, timelapse):
        """
        Utility method for playing a timelapse

        :param timelapse: the Timelapse object
        """

        for i in range(timelapse.frame_length):
            cv2.imshow('Timelapse', timelapse.frames[i, :, :, :])
            cv2.waitKey(100)

    def process_hyperlapse(self, urls):
        """

        :param urls:
        :return:
        """

        imgs = []

        for url in urls:
            try:
                image_data = requests.get(url)
                image_data.raise_for_status()

                # Standard image, NOT in matrix form
                image = Image.open(BytesIO(image_data.content)).convert('RGB')
                imgs.append(image)

                print("Loaded in image -> " + url)

            except Exception:
                print("Skipping image -> " + url)
                pass

        frame_length = len(imgs)
        frames = np.zeros((frame_length,) + self.OUTPUT_RESOLUTION + (3,))

        for i in range(frame_length):
            # Convert to OpenCV form...
            open_cv_image = cv2.normalize(np.array(imgs[i]), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)

            # Need to swap RGB
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

            frames[i, :, :, :] = cv2.resize(open_cv_image, (self.OUTPUT_RESOLUTION[1], self.OUTPUT_RESOLUTION[0]))

        timelapse = Timelapse(frames, frame_length, self.OUTPUT_RESOLUTION[0], self.OUTPUT_RESOLUTION[1], 3)

        self.__save_timelapse(timelapse)


# if __name__ == '__main__':
    # processor = TimelapseProcessor("non-uniform-sampling")

    # processor = TimelapseProcessor("uniform-sampling-wmt")
    # processor.process_files()
