import math
import numpy as np
import cv2

class StructuredLight:
    def __init__(self, projector_shape):
        self.projector_shape = projector_shape
    def light_patterns(self):
        raise NotImplementedError()
    def process_images(self, images):
        raise NotImplementedError()
    def process_color(self, images, gamma_correction = 2.2):
        return np.mean([np.min(images, axis=0), np.max(images, axis=0)], axis = 0) ** (1 / gamma_correction)
    def capture(self, projector_callback, capture_callback, wait_time = 1000, flush_frames = 3, gamma_correction = 2.2):
        # full white
        projector_callback(255 * np.ones(self.projector_shape, np.uint8))
        # camera preview
        while True:
            frame = capture_callback()
            # normalization and gamma correction
            frame = (frame / np.iinfo(frame.dtype).max) ** gamma_correction
            # show image
            cv2.imshow("Camera", (255 * frame).astype(np.uint8))
            if(cv2.waitKey(10) != -1):
                break
        # capture images
        images = []
        for pattern in self.light_patterns():
            # project pattern
            projector_callback(pattern)
            # capture image
            cv2.waitKey(max(1,wait_time))
            for _ in range(flush_frames):
                frame = capture_callback()
            # normalization and gamma correction
            frame = (frame / np.iinfo(frame.dtype).max) ** gamma_correction
            # show image
            cv2.imshow("Camera", (255 * frame).astype(np.uint8))
            cv2.waitKey(1)
            # add to images list
            images.append(frame.astype(np.float32))
        # full black
        projector_callback(np.zeros(self.projector_shape, np.uint8))
        # return captured images
        return images
    @staticmethod
    def _rgb_2_gray(images):
        return np.asarray([np.sum((0.2989, 0.5870, 0.1140) * img, axis = -1) for img in images])


class BinaryStructuredLight(StructuredLight):
    def __init__(self, *args, **kwargs):
        super(BinaryStructuredLight, self).__init__(*args)
        self.resolution_limit = kwargs.get('resolution_limit', 1)

    def light_patterns(self):
        n = math.ceil(math.log(max(self.projector_shape))/math.log(2))
        patterns_x = [255 * (np.indices(self.projector_shape)[1]//((2**i)) % 2 > 0).astype(np.uint8) for i in range(n - 1, self.resolution_limit - 1, -1)]
        patterns_y = [255 * (np.indices(self.projector_shape)[0]//((2**i)) % 2 > 0).astype(np.uint8) for i in range(n - 1, self.resolution_limit - 1, -1)]
        return [255* np.ones(self.projector_shape, np.uint8), *patterns_x, *patterns_y, np.zeros(self.projector_shape, np.uint8)]
        
    def process_images(self, images, valid_threshold = 0.1):
        bw_images = StructuredLight._rgb_2_gray(np.asarray(images)[:,:,:,::-1])
        pattern_encoding_color = np.moveaxis(np.asarray(list(zip(bw_images[1:len(bw_images)//2], bw_images[len(bw_images)//2:-1]))),[0,1],[-1,-2])
        pattern_threshold = np.mean([np.min(pattern_encoding_color, axis=-1), np.max(pattern_encoding_color, axis=-1)], axis = 0)
        valid_projection = np.abs(np.min(bw_images, axis=0) - np.max(bw_images, axis=0)) > valid_threshold
        pattern_encoding_binary = pattern_encoding_color > pattern_threshold.reshape((*pattern_threshold.shape,1))
        # convert binary to value
        pattern_encoding = 2 ** ((pattern_encoding_binary.shape[-1] - 1  + self.resolution_limit) - np.indices(pattern_encoding_binary.shape)[-1])
        pattern_encoding = np.sum(pattern_encoding_binary * pattern_encoding, axis = -1) * valid_projection.reshape((*valid_projection.shape,1))
        # append valid mask
        projector_map = np.concatenate([pattern_encoding, valid_projection.reshape((*valid_projection.shape,1))], axis = -1)
        return projector_map


class GrayCodeStructuredLight(StructuredLight):
    def __init__(self, *args, **kwargs):
        super(GrayCodeStructuredLight, self).__init__(*args)
        self.resolution_limit = kwargs.get('resolution_limit', 0)
    
    def light_patterns(self):
        n = math.ceil(math.log(max(self.projector_shape))/math.log(2))
        patterns_x = [255 * (np.indices(self.projector_shape)[1]//((2**i)) % 2 > 0).astype(np.uint8) for i in range(n - 1, self.resolution_limit - 1, -1)]
        patterns_x_shr = np.roll(patterns_x, 1, axis = 0)
        patterns_x_shr[0] = 0
        patterns_x = np.bitwise_xor(patterns_x, patterns_x_shr)
        patterns_y = [255 * (np.indices(self.projector_shape)[0]//((2**i)) % 2 > 0).astype(np.uint8) for i in range(n - 1, self.resolution_limit - 1, -1)]
        patterns_y_shr = np.roll(patterns_y, 1, axis = 0)
        patterns_y_shr[0] = 0
        patterns_y = np.bitwise_xor(patterns_y, patterns_y_shr)
        return [255* np.ones(self.projector_shape, np.uint8), *patterns_x, *patterns_y, np.zeros(self.projector_shape, np.uint8)]

    def process_images(self, images, valid_threshold = 0.1):
        bw_images = StructuredLight._rgb_2_gray(np.asarray(images)[:,:,:,::-1])
        pattern_encoding_color = np.moveaxis(np.asarray(list(zip(bw_images[1:len(bw_images)//2], bw_images[len(bw_images)//2:-1]))),[0,1],[-1,-2])
        pattern_threshold = np.mean([np.min(pattern_encoding_color, axis=-1), np.max(pattern_encoding_color, axis=-1)], axis = 0)
        valid_projection = np.abs(np.min(bw_images, axis=0) - np.max(bw_images, axis=0)) > valid_threshold
        pattern_encoding_binary = pattern_encoding_color > pattern_threshold.reshape((*pattern_threshold.shape,1))
        # convert gray code to binary
        for i in range(1, pattern_encoding_binary.shape[-1]):
            pattern_encoding_binary[:,:,:,i] = np.logical_xor(pattern_encoding_binary[:,:,:,i], pattern_encoding_binary[:,:,:,i-1])
        # convert binary to value
        pattern_encoding = 2 ** ((pattern_encoding_binary.shape[-1] - 1  + self.resolution_limit) - np.indices(pattern_encoding_binary.shape)[-1])
        pattern_encoding = np.sum(pattern_encoding_binary * pattern_encoding, axis = -1) * valid_projection.reshape((*valid_projection.shape,1))
        # append valid mask
        projector_map = np.concatenate([pattern_encoding, valid_projection.reshape((*valid_projection.shape,1))], axis = -1)
        return projector_map


class PhaseStructuredLight(StructuredLight):
    def __init__(self, *args, **kwargs):
        super(PhaseStructuredLight, self).__init__(*args)
        self.period_resolution = kwargs.get('period_resolution', 32)
    
    def light_patterns(self):
        return [
            255 * np.ones(self.projector_shape, np.uint8),
            (255 * (0.5 + np.cos(math.pi * np.indices(self.projector_shape)[0] / (4 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.sin(math.pi * np.indices(self.projector_shape)[0] / (4 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.cos(math.pi * np.indices(self.projector_shape)[1] / (4 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.sin(math.pi * np.indices(self.projector_shape)[1] / (4 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.cos(math.pi * np.indices(self.projector_shape)[0] / (2 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.sin(math.pi * np.indices(self.projector_shape)[0] / (2 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.cos(math.pi * np.indices(self.projector_shape)[1] / (2 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.sin(math.pi * np.indices(self.projector_shape)[1] / (2 * self.period_resolution))/2)).astype(np.uint8),
            (255 * (0.5 + np.cos(math.pi * np.indices(self.projector_shape)[0] / self.period_resolution)/2)).astype(np.uint8),
            (255 * (0.5 + np.sin(math.pi * np.indices(self.projector_shape)[0] / self.period_resolution)/2)).astype(np.uint8),
            (255 * (0.5 + np.cos(math.pi * np.indices(self.projector_shape)[1] / self.period_resolution)/2)).astype(np.uint8),
            (255 * (0.5 + np.sin(math.pi * np.indices(self.projector_shape)[1] / self.period_resolution)/2)).astype(np.uint8),
            np.zeros(self.projector_shape, np.uint8)
        ]

    def process_images(self, images, valid_threshold = 0.05):
        bw_images = StructuredLight._rgb_2_gray(np.asarray(images)[:,:,:,::-1])
        bw_images = np.asarray(bw_images) ** (1.0/2.2)
        cos = np.asarray([bw_images[1 + i * 2] - np.mean([bw_images[0],bw_images[-1]],axis=0) for i in range((len(bw_images)-2)//2)])
        sin = np.asarray([bw_images[2 + i * 2] - np.mean([bw_images[0],bw_images[-1]],axis=0) for i in range((len(bw_images)-2)//2)])
        valid_projection = np.std(bw_images, axis = 0) > valid_threshold
        phase = np.arctan2(sin, cos) * valid_projection
        #TODO phase unwrapping
        #intensity calibration needed
        from matplotlib import pyplot as plt
        for p in phase:
            plt.imshow(p % 0.5)
            plt.show()



if __name__ == "__main__":
    import os
    import pickle
    from camera import Camera
    cast = __import__('opencv-chromecast')

    sl = GrayCodeStructuredLight((480, 854))

    if(os.path.exists('images.pickle') and input('Load pickle (Y/n)?') != 'n'):
        pattern_images = pickle.load(open('images.pickle','rb'))
    else:
        camera = Camera()
        chromecast = cast.Chromecast("10.0.0.195")

        pattern_images = sl.capture(chromecast.imshow, camera.capture)
        pickle.dump(pattern_images, open('images.pickle','wb'))

    projector_map = sl.process_images(pattern_images)
    color_map = sl.process_color(pattern_images)

    # show results
    map = ((255,255,0) * projector_map / projector_map.max()).astype(np.uint8)[:,:,::-1]
    val = (255 * projector_map[:,:,-1]).astype(np.uint8)
    col = (255 * color_map / color_map.max()).astype(np.uint8)
    cv2.imshow("map", map)
    cv2.imshow("val", val)
    cv2.imshow("col", col)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # img = cv2.imread("lama.jpg")
    # img = cv2.remap(img, projector_map[:,:,:2].astype(np.float32), (), cv2.INTER_LINEAR)

    # if chromecast is not None:
    #   chromecast.imshow(img)