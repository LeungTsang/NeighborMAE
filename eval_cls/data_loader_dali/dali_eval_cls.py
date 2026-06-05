try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    import nvidia.dali.math as dmath
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import os

@pipeline_def
def create_dali_pipeline_cls(args, file_list, mean, std, dali_device, decoder_device,
                         device_memory_padding, host_memory_padding,
                         preallocate_width_hint, preallocate_height_hint):

    img, target = fn.readers.file(file_root=args.data_path,
                             file_list=file_list,
                             shard_id=args.rank,
                             num_shards=args.world_size,
                             random_shuffle=False,
                             shuffle_after_epoch=True,
                             pad_last_batch=False,
                             prefetch_queue_depth=4,
                             seed=args.seed,
                             name="reader_img")

    shape = fn.peek_image_shape(img)

    img = fn.decoders.image(img,
                            device=decoder_device, output_type=types.RGB,
                            device_memory_padding=device_memory_padding,
                            host_memory_padding=host_memory_padding,
                            preallocate_width_hint=preallocate_width_hint,
                            preallocate_height_hint=preallocate_height_hint,
                            jpeg_fancy_upsampling=True,
                            )

    height = fn.cast(shape[0], dtype=types.FLOAT)
    width = fn.cast(shape[1], dtype=types.FLOAT)

    area = height * width

    target_area = area * fn.random.uniform(range=args.crop_scale)
    aspect_ratio = dmath.exp(fn.random.uniform(range=dmath.log((args.aspect_ratio, 1 / args.aspect_ratio))))

    h = dmath.sqrt(target_area / aspect_ratio)
    w = dmath.sqrt(target_area * aspect_ratio)

    h = fn.reductions.min(fn.stack(h, height, axis=0), axes=0)
    w = fn.reductions.min(fn.stack(w, width, axis=0), axes=0)

    i = fn.random.uniform(range=(0, 1))
    j = fn.random.uniform(range=(0, 1))

    img = fn.crop(img, crop_h=h, crop_w=w, crop_pos_x=j, crop_pos_y=i)
    img = fn.resize(img, device=dali_device, resize_x=args.img_size, resize_y=args.img_size, interp_type=types.INTERP_CUBIC, antialias=True)

    MEAN = types.Constant(mean, shape=(1, 1, 3))
    STD = types.Constant(std, shape=(1, 1, 3))

    img = fn.normalize(img, mean=MEAN, stddev=STD, dtype=types.FLOAT)
    img = fn.transpose(img, perm=[2, 0, 1])

    target = fn.squeeze(target, axes=[0])

    return img, target


def setup_dali_params(args, mode):

    dali_params = {}

    dali_params['mean'] = [x * 255. for x in args.mean]
    dali_params['std'] = [x * 255. for x in args.std]

    dali_params['device_memory_padding'] = 6400000
    dali_params['host_memory_padding'] = 6400000
    dali_params['preallocate_width_hint'] = 1280
    dali_params['preallocate_height_hint'] = 1280

    dali_params['dali_device'] = 'cpu' if args.dali_cpu else 'gpu'
    dali_params['decoder_device'] = 'cpu' if args.dali_cpu else 'mixed'

    return dali_params


def build_dali_loader_cls(args, mode='train'):
    dali_params = setup_dali_params(args, mode)

    file_list = os.path.join(args.data_path, 'split', args.split)

    output_map = ['img', 'target']

    pipe = create_dali_pipeline_cls(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=13 + args.rank,
                                args=args,
                                file_list=file_list,
                                **dali_params)

    pipe.build()
    loader = DALIGenericIterator(pipe, output_map,
                                 reader_name='reader_img',
                                 last_batch_policy=LastBatchPolicy.DROP,
                                 last_batch_padded=False,
                                 auto_reset=True)

    with open(file_list) as f:
        dataset_size = len(f.readlines())
    loader_size = len(loader)

    return loader, dataset_size, loader_size