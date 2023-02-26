import os


class DatasetCatalog(object):
    default_data_dir = os.path.expanduser('~/Datasets')
    DATA_DIR = os.environ.get('DATASET_HOME', default_data_dir)

    @staticmethod
    def get(name: str):
        if 'kinectrobot' in name:
            return get_kinectrobot(name)
        raise RuntimeError("Dataset not available: {}".format(name))


def get_kinectrobot(name):
    data_dir = 'data/kinect'
    items = name.split('_')[1:]
    scene = '_'.join(items[:-4])
    skip, firstend, split, training_factor = items[-4:]
    first_frame, last_frame = map(int, firstend.split('to'))
    training_factor = float(training_factor[2:])
    skip = int(skip[4:])
    return dict(
        factory='KinectRobot',
        args={'data_dir': data_dir,
              'scene': scene,
              'skip': skip,
              'split': split,
              'first_frame': first_frame,
              'last_frame': last_frame,
              'training_factor': training_factor
              }
    )
