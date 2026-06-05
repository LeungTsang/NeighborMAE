import glob

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    import nvidia.dali.math as dmath
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import numpy as np
import os
import glob
import random

# fmow_mean = [0.418 * 255, 0.421 * 255, 0.399 * 255]
# fmow_std = [0.288 * 255, 0.275 * 255, 0.276 * 255]
#
fmow_meta = {'mean': [0.418 * 255, 0.421 * 255, 0.399 * 255], 'std': [0.288 * 255, 0.275 * 255, 0.276 * 255],
             'device_memory_padding': 26378240, 'host_memory_padding': 17568064,
             'preallocate_width_hint': 3693, 'preallocate_height_hint': 1290}


class ExternalInputCallable:
    def __init__(self, args, actual_batch_size):
        self.base_path = args.data_path
        self.sample_num = args.sample_num
        self.actual_batch_size = actual_batch_size

        self.files = sorted(glob.glob(os.path.join(args.data_path, '**', '*.' + args.img_ext), recursive=True))
        self.image_id_in_folder = []
        self.image_id_to_folder = []
        cur_folder = None
        for i, p in enumerate(self.files):
            folder = os.path.dirname(p)
            if folder != cur_folder:
                cur_folder = folder
                self.image_id_in_folder.append([i,])
            else:
                self.image_id_in_folder[-1].append(i)
            self.image_id_to_folder.append(len(self.image_id_in_folder)-1)

        self.shard_id = args.local_rank
        self.num_shards = args.local_world_size
        # If the dataset size is not divisibvle by number of shards,
        # the trailing samples will be omitted.
        self.dataset_size = len(self.files)
        self.shard_size = len(self.files) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # If the shard size is not divisible by the batch size, the last
        # incomplete batch will be omitted.
        self.full_iterations = self.shard_size // self.actual_batch_size
        self.perm = np.random.default_rng(seed=42)
        self.perm = self.perm.permutation(len(self.files))
        self.last_seen_epoch = (
            # so that we don't have to recompute the `self.perm` for every sample
            None
        )

    def read_encoded_img(self, filename):
        with open(filename, "rb") as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        return encoded_img

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx)
            self.perm = self.perm.permutation(len(self.files))

        if self.sample_num == 1:
            sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
            return self.read_encoded_img(self.files[sample_idx]),
        else:
            sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
            sample_neighbor_idx = random.sample(self.image_id_in_folder[self.image_id_to_folder[sample_idx]], 1)[0]
            return self.read_encoded_img(self.files[sample_idx]), self.read_encoded_img(self.files[sample_neighbor_idx])


@pipeline_def
def create_dali_pipeline(args, ext):

    dali_device = 'cpu' if args.dali_cpu else 'gpu'
    decoder_device = 'cpu' if args.dali_cpu else 'mixed'
    
    data = fn.external_source(
        source=ext,
        num_outputs=args.sample_num,
        batch=False,
        parallel=True,
        dtype=[types.UINT8 for _ in range(args.sample_num)],
    )
    #args.sample_num = 2#args.sample_num * args.crop_num_per_sample
    img_size = args.img_size * 2 if args.use_satmaepp else args.img_size

    shape = fn.stack(*[fn.peek_image_shape(data[i]) for i in range(args.sample_num)], axis=0)
    h_original = fn.stack(*[shape[i, 0] for i in range(args.sample_num)])
    h_original = fn.cast(h_original, dtype=types.FLOAT)
    w_original = fn.stack(*[shape[i, 1] for i in range(args.sample_num)])
    w_original = fn.cast(w_original, dtype=types.FLOAT)

    img = [fn.decoders.image(data[i], device=decoder_device, output_type=types.RGB,
                             jpeg_fancy_upsampling=True) for i in range(args.sample_num)]

    ratio = (args.aspect_ratio, 1 / args.aspect_ratio)
    aspect_ratio = dmath.exp(fn.random.uniform(range=dmath.log(ratio), shape=(args.sample_num,)))
    max_h = w_original / aspect_ratio
    target_area = fn.random.uniform(range=args.crop_scale, shape=(args.sample_num,))
    scale_ratio = dmath.sqrt(target_area)
    h_target_size = scale_ratio * h_original
    h_target_size = fn.reductions.max(fn.stack(h_target_size, axis=1), axes=1)
    h_target_size = fn.reductions.min(fn.stack(h_target_size, max_h, h_original, axis=1), axes=1)
    w_target_size = h_target_size * aspect_ratio
    p = fn.random.uniform(range=(0, 1), shape=(args.sample_num, 2))

    img = [fn.crop(img[i], crop_h=h_target_size[i], crop_w=w_target_size[i],
                   crop_pos_x=p[i, 1], crop_pos_y=p[i, 0]) for i in range(args.sample_num)]
    img = [fn.resize(img[i], device=dali_device, resize_x=img_size, resize_y=img_size,
                     interp_type=types.INTERP_CUBIC, antialias=True) for i in range(args.sample_num)]

    MEAN = types.Constant([x * 255. for x in args.mean], shape=(1, 1, 3))
    STD = types.Constant([x * 255. for x in args.std], shape=(1, 1, 3))

    img = [fn.normalize(img_i, mean=MEAN, stddev=STD, dtype=types.FLOAT) for img_i in img]
    img = [fn.transpose(img_i, perm=[2, 0, 1]) for img_i in img]

    left = (w_original - w_target_size) * p[:, 1]
    top = h_original - (h_original - h_target_size) * p[:, 0]
    right = left + w_target_size
    bottom = top - h_target_size
    meta = fn.stack(left / h_original, top / h_original, right / h_original, bottom / h_original, axis=1)

    return *img, meta


def build_dali_loader(args, loader_batch_size):
    #dali_params = setup_dali_params(args)

    ext_reader = ExternalInputCallable(args, loader_batch_size)
    dataset_size = ext_reader.dataset_size
    loader_size = ext_reader.full_iterations

    pipe = create_dali_pipeline(batch_size=loader_batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=13 + args.rank,
                                py_num_workers=4,
                                py_start_method='spawn',
                                args=args,
                                ext=ext_reader)
    pipe.build()
    output_map = ['img_{}'.format(i) for i in range(args.sample_num)] + ['meta']
    loader = DALIGenericIterator(pipe, output_map,
                                 last_batch_policy=LastBatchPolicy.DROP,
                                 auto_reset=True)

    return loader, dataset_size, loader_size


